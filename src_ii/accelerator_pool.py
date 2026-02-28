"""Multi-device dispatch for bin-packed forward passes.

Wraps K model replicas (one per device). Queries are bin-packed once,
bins are distributed across devices by FLOPS cost, and results are
gathered in submission order.

K=1 is the default single-device case — same code path as K=8.
One stream, one device in the assignment list, one gather.

Two-level parallelism:
  Within-node: AcceleratorPool manages K replicas on K local GPUs.
    Bins assigned by FLOPS-balanced greedy scheduling. Parallel execution
    via per-device CUDA streams. Gradients summed across replicas.
  Cross-node: distributed.py all-reduces gradients across processes.
    Called AFTER within-node gathering but BEFORE optimizer.step().

Import constraints:
  - IMPORTS from src_ii.batch_executor: _scatter, _structure_hash
  - IMPORTS from src_ii.inference_packing: compute_entry_seq_len, pack_for_inference
  - IMPORTS from src_ii.forward_packed: prepare_packed_forward, packed_forward
  - IMPORTS from src_ii.triumphant_future_reduction_ops: denoise_all
  - IMPORTS from src_ii.bin_packer: REFERENCE_TOTAL_LEN
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from src_ii.bin_packer import REFERENCE_TOTAL_LEN


class AcceleratorPool:
    """Multi-device dispatch for bin-packed forward passes.

    Wraps K model replicas (one per device). Queries are bin-packed once,
    bins are distributed across devices by FLOPS cost, and results are
    gathered in submission order.

    K=1 is the default single-device case — same behavior as BatchExecutor.
    """

    def __init__(
        self,
        model_factory: Callable[[torch.device], nn.Module],
        devices: list[torch.device] | None = None,
        max_total_len: int = REFERENCE_TOTAL_LEN,
    ):
        """
        Args:
            model_factory: fn(device) → model. Called K times, once per device.
                For K=1, called once with the current device.
                For K>1, must return independent replicas (deepcopy or fresh load).
            devices: List of devices. len(devices) = K. None → [cuda:0].
            max_total_len: Max packed sequence length per bin.
        """
        if devices is None:
            devices = [torch.device("cuda:0")]
        self.devices = devices
        self.max_total_len = max_total_len
        self._models = [model_factory(d) for d in devices]
        self._streams = [torch.cuda.Stream(device=d) for d in devices]
        self._plan_cache: dict[str, dict] = {}

        # Query adapter capacity from primary model
        from src_ii.multi_lora import adapter_capacity
        cap = adapter_capacity(self._models[0])
        self._n_adapter_slots = cap["max_adapters"]

    @property
    def primary_model(self) -> nn.Module:
        """The model on devices[0]. Used for optimizer, parameter access."""
        return self._models[0]

    # -- BatchExecutor compatibility properties --
    @property
    def model(self) -> nn.Module:
        return self._models[0]

    @property
    def device(self) -> torch.device:
        return self.devices[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, queries: list[dict]) -> list[dict]:
        """Same interface as BatchExecutor.execute().

        Queries → scatter → bin-pack → assign bins to devices →
        parallel execute via CUDA streams → gather in submission order.
        """
        from src_ii.batch_executor import _scatter, _structure_hash

        entries = _scatter(queries)
        if not entries:
            return []

        key = _structure_hash(entries)
        if key not in self._plan_cache:
            self._plan_cache[key] = self._build_distributed_plan(entries)

        return self._execute_distributed(entries, self._plan_cache[key])

    def broadcast_params(self) -> None:
        """Copy primary_model params to all replicas. Call after optimizer.step()."""
        K = len(self._models)
        if K <= 1:
            return
        primary = self._models[0]
        for replica in self._models[1:]:
            for p_src, p_dst in zip(primary.parameters(), replica.parameters()):
                p_dst.data.copy_(p_src.data)

    def gather_gradients(self) -> None:
        """Sum .grad from all replicas into primary_model.

        The bins processed by each replica contribute independent gradient
        fragments. Summing them reconstructs the full gradient that a
        single-device execution would have computed. Replica grads are
        cleared after gathering to prevent double-counting.

        Call AFTER backward, BEFORE optimizer.step().
        """
        K = len(self._models)
        if K <= 1:
            return
        for params in zip(*(m.parameters() for m in self._models)):
            p_primary = params[0]
            for p_replica in params[1:]:
                if p_replica.grad is not None:
                    grad_on_primary = p_replica.grad.to(p_primary.device)
                    if p_primary.grad is None:
                        p_primary.grad = grad_on_primary.clone()
                    else:
                        p_primary.grad.add_(grad_on_primary)
                    p_replica.grad = None

    def invalidate_plan_cache(self) -> None:
        """Invalidate all cached packing plans."""
        self._plan_cache.clear()

    # ------------------------------------------------------------------
    # Internal: plan construction
    # ------------------------------------------------------------------

    def _build_distributed_plan(self, entries: list[dict]) -> dict:
        """Bin-pack entries, assign bins to devices, prepare per-device state."""
        from src_ii.inference_packing import compute_entry_seq_len, pack_for_inference
        from src_ii.forward_packed import prepare_packed_forward

        # Compute seq_lens and bin-pack
        entry_seq_lens = [
            compute_entry_seq_len(
                e["x"].shape[2], e["x"].shape[3], e["cap_len"],
            )
            for e in entries
        ]
        bin_index_lists = pack_for_inference(
            entry_seq_lens, max_total_len=self.max_total_len,
        )

        # FLOPS per bin ≈ sum(seq_len² for entries in bin)
        bin_flops = [
            sum(entry_seq_lens[i] ** 2 for i in indices)
            for indices in bin_index_lists
        ]

        # Greedy FLOPS-balanced assignment: largest bins first
        K = len(self.devices)
        sorted_bins = sorted(
            range(len(bin_index_lists)),
            key=lambda i: bin_flops[i],
            reverse=True,
        )
        device_loads = [0] * K
        device_bins: list[list[int]] = [[] for _ in range(K)]
        for bin_idx in sorted_bins:
            min_dev = min(range(K), key=lambda d: device_loads[d])
            device_bins[min_dev].append(bin_idx)
            device_loads[min_dev] += bin_flops[bin_idx]

        # Prepare packed forward state for each bin on its target device
        per_device: list[list[dict]] = []
        for dev_idx in range(K):
            device = self.devices[dev_idx]
            model = self._models[dev_idx]
            bins_for_device = []
            for bin_idx in device_bins[dev_idx]:
                indices = bin_index_lists[bin_idx]
                context_list = [entries[i]["cond"] for i in indices]
                img_sizes = [
                    (entries[i]["x"].shape[2], entries[i]["x"].shape[3])
                    for i in indices
                ]
                cap_lens = [entries[i]["cap_len"] for i in indices]
                prepared = prepare_packed_forward(
                    model, context_list, img_sizes, cap_lens, device,
                    target_len=self.max_total_len,
                )
                bins_for_device.append({
                    "indices": indices,
                    "prepared": prepared,
                })
            per_device.append(bins_for_device)

        return {"per_device": per_device}

    # ------------------------------------------------------------------
    # Internal: distributed execution
    # ------------------------------------------------------------------

    def _execute_distributed(
        self, entries: list[dict], plan: dict,
    ) -> list[dict]:
        """Execute all bins across devices using CUDA streams."""
        from src_ii.batch_executor import execute_single_bin

        results: dict[int, dict] = {}

        for dev_idx, bins_for_device in enumerate(plan["per_device"]):
            device = self.devices[dev_idx]
            model = self._models[dev_idx]
            stream = self._streams[dev_idx]

            with torch.cuda.stream(stream):
                for bin_info in bins_for_device:
                    bin_results = execute_single_bin(
                        entries, bin_info["indices"], bin_info["prepared"],
                        model, device, self._n_adapter_slots,
                    )
                    for r, idx in zip(bin_results, bin_info["indices"]):
                        results[idx] = r

        for stream in self._streams:
            stream.synchronize()

        return [results[i] for i in range(len(entries))]
