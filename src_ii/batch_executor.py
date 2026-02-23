"""SCATTER -> PACKSOLVE -> EXECUTE -> DENOISE pipeline for /batch_forward.

Client submits fork specs + literal tensors. We apply mutations, bin-pack
entries into one or more packed forwards at REFERENCE_TOTAL_LEN, run each
bin as a separate packed forward, denoise, and return tagged (query_id,
entry_id) results.

PACKSOLVE: entries are assigned to bins via FFD bin packing so that every
bin's total sequence length fits within REFERENCE_TOTAL_LEN. All bins are
padded to the same REFERENCE_TOTAL_LEN -- zero recompiles regardless of
how many bins or what the content is. The server handles arbitrary workloads.
"""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F

from src_ii.forward_packed import prepare_packed_forward, packed_forward
from src_ii.bin_packer import REFERENCE_TOTAL_LEN
from src_ii.inference_packing import pack_for_inference, compute_entry_seq_len
from src_ii.sigma_schedule import resolution_shift
from src_ii.triumphant_future_reduction_ops import denoise_all, latent_padded


class BatchExecutor:
    """Execute batches of fork specs through the model.

    Created once at server startup. Caches packing plans across Euler steps
    -- fork spec structure is constant; only latent values change per step.

    Entries are bin-packed into one or more forward calls, each padded to
    REFERENCE_TOTAL_LEN. One compiled graph, zero recompiles. The number of
    bins adjusts to the workload -- small batches use one bin, large batches
    use as many bins as needed.
    """

    def __init__(self, model, device: torch.device, max_total_len: int = REFERENCE_TOTAL_LEN):
        self.model = model
        self.device = device
        self.max_total_len = max_total_len
        self._plan_cache: dict[str, dict] = {}  # structure_hash -> plan

    def execute(self, queries: list[dict]) -> list[dict]:
        """Execute fork specs, return tagged denoised results.

        Query schema: query_id, base_latent (1,16,H,W), base_cond (1,seq,2560),
        base_cap_len, base_resolution (W,H), sigma, forks, adapter_scales.
        Fork schema: entry_id, cond (None=identity), resolution (None=identity).
        Returns: [{query_id, entry_id, denoised (1,16,H,W), scores}].
        """
        entries = _scatter(queries)
        key = _structure_hash(entries)
        if key not in self._plan_cache:
            self._plan_cache[key] = _build_plan(
                entries, self.model, self.device, self.max_total_len
            )
        return _execute_plan(entries, self._plan_cache[key], self.model, self.device)

    def invalidate_plan_cache(self):
        """Call after model weight changes (new LoRA, recompile, etc.)."""
        self._plan_cache.clear()


# ------------------------------------------------------------------
# SCATTER
# ------------------------------------------------------------------

def _scatter(queries: list[dict]) -> list[dict]:
    entries = []
    for q in queries:
        base_latent = q["base_latent"]
        base_cond = q["base_cond"]
        base_cap_len = q["base_cap_len"]
        base_res = q["base_resolution"]  # (W, H)
        sigma = q["sigma"]
        adapter_scales = q.get("adapter_scales")
        # Normalize to 1D (n_adapters,) for per-entry storage.
        # Queries pass (1, n_adapters) or (n_adapters,); _execute_plan
        # stacks entries to (n_entries_in_bin, n_adapters) for packed forward.
        if adapter_scales is not None and adapter_scales.dim() == 2:
            adapter_scales = adapter_scales.squeeze(0)

        # Precompute the base resolution's shift alpha so forks can derive
        # their own resolution-dependent sigma from the client's base sigma.
        alpha_base = resolution_shift(base_res[0], base_res[1])

        for fork in q["forks"]:
            cond = fork.get("cond")
            if cond is None:
                cond = base_cond
            resolution = fork.get("resolution")
            if resolution is None:
                resolution = base_res

            lh, lw = latent_padded(resolution[0], resolution[1])
            base_lh, base_lw = base_latent.shape[2], base_latent.shape[3]
            if lh == base_lh and lw == base_lw:
                x = base_latent.clone()
            else:
                x = F.interpolate(
                    base_latent, size=(lh, lw),
                    mode="bilinear", align_corners=False,
                )

            # Per-entry sigma shifting: the client's sigma was computed with
            # alpha_base (the base resolution's shift).  Each fork's resolution
            # has its own alpha -- rescale sigma by the ratio so that smaller
            # images see higher effective sigma (more noise) per SD3 Eq.23.
            alpha_entry = resolution_shift(resolution[0], resolution[1])
            sigma_entry = sigma * (alpha_entry / alpha_base)

            entries.append({
                "query_id": q["query_id"],
                "entry_id": fork["entry_id"],
                "x": x,
                "cond": cond,
                "cap_len": cond.shape[1],
                "sigma": sigma_entry,
                "adapter_scales": adapter_scales,
            })
    return entries


# ------------------------------------------------------------------
# PACKSOLVE (bin-pack entries into REFERENCE_TOTAL_LEN-sized bins)
# ------------------------------------------------------------------

def _structure_hash(entries: list[dict]) -> str:
    key_parts = [
        (e["query_id"], e["entry_id"], e["x"].shape[2], e["x"].shape[3], e["cap_len"])
        for e in entries
    ]
    return hashlib.md5(json.dumps(key_parts).encode()).hexdigest()


def _build_plan(entries, model, device, max_total_len):
    """Bin-pack entries into max_total_len-sized bins and prepare state.

    Uses FFD bin packing from inference_packing. Each bin is padded to
    max_total_len by prepare_packed_forward, so every forward call at the
    same max_total_len hits the same compiled graph.
    """
    # Compute seq_len per entry (latent dims, not pixel dims)
    entry_seq_lens = [
        compute_entry_seq_len(e["x"].shape[2], e["x"].shape[3], e["cap_len"])
        for e in entries
    ]

    # PACKSOLVE: bin-pack into REFERENCE_TOTAL_LEN-sized bins
    bins = pack_for_inference(entry_seq_lens, max_total_len=max_total_len)

    # Prepare CONSTANT state for each bin (all padded to max_total_len)
    per_bin = []
    for bin_indices in bins:
        context_list = [entries[i]["cond"] for i in bin_indices]
        img_sizes = [(entries[i]["x"].shape[2], entries[i]["x"].shape[3]) for i in bin_indices]
        cap_lens = [entries[i]["cap_len"] for i in bin_indices]
        prepared = prepare_packed_forward(
            model, context_list, img_sizes, cap_lens, device,
            target_len=max_total_len,
        )
        per_bin.append({"indices": bin_indices, "prepared": prepared})

    return {"bins": per_bin}


# ------------------------------------------------------------------
# EXECUTE + DENOISE (loop over bins)
# ------------------------------------------------------------------

def _execute_plan(entries, plan, model, device):
    results = {}

    for bin_info in plan["bins"]:
        indices = bin_info["indices"]
        state = bin_info["prepared"]

        x_list = [entries[i]["x"].to(device) for i in indices]
        sigmas = [entries[i]["sigma"] for i in indices]
        timesteps = [torch.tensor([s], device=device, dtype=torch.float32) for s in sigmas]

        # Build per-entry adapter_scales for this bin
        any_has_scales = any(entries[i].get("adapter_scales") is not None for i in indices)
        if any_has_scales:
            scale_rows = []
            for i in indices:
                s = entries[i].get("adapter_scales")
                if s is not None:
                    scale_rows.append(s.to(device))
                else:
                    # Entry has no adapter_scales — use zeros (no adapter contribution)
                    # Infer n_adapters from a neighbor
                    n_adapters = next(
                        entries[j]["adapter_scales"].shape[-1]
                        for j in indices if entries[j].get("adapter_scales") is not None
                    )
                    scale_rows.append(torch.zeros(n_adapters, device=device))
            adapter_scales = torch.stack(scale_rows)  # (n_entries_in_bin, n_adapters)
        else:
            adapter_scales = None

        fields, scores = packed_forward(
            model, x_list, timesteps,
            state["refined_caps"], state["packing_info"],
            state["block_mask"], state["packed_rope"],
            adapter_scales=adapter_scales,
        )

        # denoised = x - field * sigma
        sigma_tensors = [
            torch.tensor(s, device=device, dtype=x_list[j].dtype)
            for j, s in enumerate(sigmas)
        ]
        denoised_list = denoise_all(x_list, fields, sigma_tensors)

        for local_idx, global_idx in enumerate(indices):
            e = entries[global_idx]
            results[global_idx] = {
                "query_id": e["query_id"],
                "entry_id": e["entry_id"],
                "denoised": denoised_list[local_idx].cpu(),
                "scores": scores[local_idx].cpu() if scores is not None else None,
            }

    return [results[i] for i in range(len(entries))]
