"""Inference server: thin ZeroMQ RPC dispatch.

Model lifecycle delegated to ModelManager (model_manager.py).
Sampling algorithms in sampling.py.
Training utilities in training_utils.py.
"""

import argparse
import time
import traceback

import torch
import zmq

from .attention import set_attention_backend
from .model_manager import ModelManager
from .protocol import pack_response, unpack_request
from .sampling import (
    run_trajectory,
    run_trajectory_packed,
    warmup_diffusion,
    warmup_packed,
)
from .training_utils import (
    accumulate_policy_gradients,
    policy_optimizer_step,
    run_backbone_hidden,
    train_btrm_step,
)


class InferenceServer:
    """GPU-owning inference server -- thin RPC dispatch layer."""

    def __init__(
        self,
        fp8_diff_path: str,
        te_path: str,
        vae_path: str,
        tokenizer_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        fp8_block_size: int = 128,
    ):
        self.device = torch.device(device)
        self.dtype = {"float32": torch.float32, "float16": torch.float16,
                      "bfloat16": torch.bfloat16}[dtype]
        self._mm = ModelManager(
            fp8_diff_path=fp8_diff_path, te_path=te_path,
            vae_path=vae_path, tokenizer_path=tokenizer_path,
            device=self.device, dtype=self.dtype,
            fp8_block_size=fp8_block_size,
        )

    # ------------------------------------------------------------------
    # RPC handlers
    # ------------------------------------------------------------------

    def handle_encode_prompt(self, params, tensors):
        """Encode text prompt to conditioning tensor."""
        self._mm.ensure_te()
        from .text_encoder import encode_prompt
        with torch.inference_mode():
            cond = encode_prompt(
                self._mm.te_model, self._mm.tokenizer, params["prompt"],
                device=self.device, layer_idx=params.get("layer_idx", -2),
            )
        return pack_response("ok", {"conditioning": cond})

    def handle_sample_trajectory(self, params, tensors):
        """Run a diffusion sampling trajectory with optional inline BTRM scoring."""
        self._mm.ensure_diffusion()
        self._mm.configure_sage_if_needed(params.get("attention_backend", "sdpa"))
        set_attention_backend(params.get("attention_backend", "sdpa"))
        result_tensors, metadata = run_trajectory(
            self._mm.diff_compiled, self._mm.diff_model,
            self.device, self.dtype, params, tensors,
            btrm_head=self._mm.btrm_head,
        )
        return pack_response("ok", result_tensors, metadata)

    def handle_sample_trajectory_packed(self, params, tensors):
        """Run N packed diffusion trajectories via FlexAttention."""
        self._mm.ensure_diffusion()
        self._mm.configure_sage_if_needed(params.get("attention_backend", "sdpa"))
        set_attention_backend(params.get("attention_backend", "sdpa"))
        result_tensors, metadata = run_trajectory_packed(
            self._mm.diff_compiled_packed, self._mm.diff_model,
            self.device, self.dtype, params, tensors,
        )
        return pack_response("ok", result_tensors, metadata)

    def handle_vae_encode(self, params, tensors):
        """Encode image to latent."""
        self._mm.ensure_vae()
        from .vae import vae_encode
        image = tensors["image"].to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            latent = vae_encode(self._mm.vae_model, image)
        return pack_response("ok", {"latent": latent.cpu()})

    def handle_vae_decode(self, params, tensors):
        """Decode latent to image."""
        self._mm.ensure_vae()
        from .vae import vae_decode
        latent = tensors["latent"].to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            image = vae_decode(self._mm.vae_model, latent)
        return pack_response("ok", {"image": image.cpu()})

    def handle_warmup(self, params, tensors):
        """Warmup diffusion model with a short trajectory."""
        self._mm.ensure_diffusion()
        attn_backend = params.get("attention_backend", "sdpa")
        self._mm.configure_sage_if_needed(attn_backend)
        set_attention_backend(attn_backend)
        warmup_diffusion(
            self._mm.diff_compiled, self._mm.diff_model,
            self.device, self.dtype,
            width=params.get("width", 1280), height=params.get("height", 832),
        )
        print(f"  [warmup] {attn_backend} done")
        return pack_response("ok")

    def handle_warmup_packed(self, params, tensors):
        """Warmup packed FlexAttention forward path."""
        self._mm.ensure_diffusion()
        n = params.get("n_images", 2)
        elapsed = warmup_packed(
            self._mm.diff_compiled_packed, self._mm.diff_model,
            self.device, self.dtype, n_images=n,
        )
        print(f"  [warmup_packed] n={n}, {elapsed:.1f}s (includes compilation)")
        return pack_response("ok")

    def handle_status(self, params, tensors):
        """Return server status."""
        return pack_response("ok", metadata=self._mm.get_status())

    def handle_free(self, params, tensors):
        """Free specified model(s)."""
        target = params.get("model", "all")
        free_fn = {
            "all": self._mm.free_all, "te": self._mm.free_te,
            "diffusion": self._mm.free_diffusion, "vae": self._mm.free_vae,
        }
        if target not in free_fn:
            return pack_response("error", metadata={
                "error": f"Unknown model: {target!r}. Valid: all, te, diffusion, vae"})
        free_fn[target]()
        print(f"  [lifecycle] Freed {target}")
        return pack_response("ok")

    def handle_inject_lora(self, params, tensors):
        """Inject LoRA adapter into diffusion model (legacy: allocate+init+recompile)."""
        self._mm.ensure_diffusion()
        metadata = self._mm.inject_lora_adapter(params)
        print(f"  [inject_lora] {metadata['adapter_name']}: "
              f"{metadata['n_adapters']} adapters, {metadata['n_params']:,} params")
        return pack_response("ok", metadata=metadata)

    def handle_allocate_adapter(self, params, tensors):
        """Allocate adapter slots (graph-mutating, no recompile)."""
        self._mm.ensure_diffusion()
        metadata = self._mm.allocate_adapter_rpc(params)
        print(f"  [allocate_adapter] {metadata['adapter_name']}: "
              f"{metadata['n_adapters']} slots, {metadata['n_params']:,} params (silent)")
        return pack_response("ok", metadata=metadata)

    def handle_init_adapter_weights(self, params, tensors):
        """Initialize adapter weights (graph-invariant, safe after compile)."""
        self._mm.ensure_diffusion()
        metadata = self._mm.init_adapter_weights_rpc(params)
        print(f"  [init_adapter_weights] {metadata['adapter_name']}: "
              f"{metadata['n_modules_initialized']} modules initialized")
        return pack_response("ok", metadata=metadata)

    def handle_update_lora_weights(self, params, tensors):
        """Update LoRA weights in-place."""
        self._mm.ensure_diffusion()
        metadata = self._mm.update_lora_weights_rpc(tensors)
        print(f"  [update_lora_weights] {metadata['n_tensors']} tensors updated")
        return pack_response("ok", metadata=metadata)

    def handle_set_adapter_config(self, params, tensors):
        """Set adapter scale and/or freeze state."""
        self._mm.ensure_diffusion()
        metadata = self._mm.set_adapter_config_rpc(params)
        print(f"  [set_adapter_config] {metadata['adapter_name']}: "
              f"scale={params.get('scale')}, frozen={params.get('frozen')} "
              f"({metadata['n_frozen']} adapters)")
        return pack_response("ok", metadata=metadata)

    def handle_get_lora_state_dict(self, params, tensors):
        """Retrieve current LoRA weights."""
        self._mm.ensure_diffusion()
        from .lora import lora_state_dict
        sd = lora_state_dict(self._mm.diff_model,
                             adapter_name=params.get("adapter_name"))
        return pack_response("ok", sd, {"n_tensors": len(sd)})

    def handle_dump_all_loras(self, params, tensors):
        """Emergency dump all LoRA adapters to disk."""
        from .lora import dump_all_loras
        if self._mm.diff_model is None:
            return pack_response("ok", metadata={
                "files": [], "note": "no diffusion model loaded"})
        result = dump_all_loras(
            self._mm.diff_model, params.get("output_dir", "lora_dumps"),
            btrm_head=self._mm.btrm_head, btrm_config=self._mm.btrm_config,
        )
        return pack_response("ok", metadata=result)

    def handle_inject_btrm_head(self, params, tensors):
        """Create BTRM scoring head on server."""
        metadata = self._mm.inject_btrm_head_rpc(params)
        print(f"  [inject_btrm_head] {metadata['n_heads']} heads, "
              f"{metadata['n_params']:,} params"
              f"{', optimizer' if metadata.get('has_optimizer') else ''}")
        return pack_response("ok", metadata=metadata)

    def handle_score_btrm(self, params, tensors):
        """Score via backbone + BTRM head."""
        self._mm.ensure_diffusion()
        assert self._mm.btrm_head is not None, "BTRM head not injected"
        self._mm.configure_sage_if_needed(params.get("attention_backend", "sdpa"))
        set_attention_backend(params.get("attention_backend", "sdpa"))
        hidden = run_backbone_hidden(
            self._mm.diff_model, tensors["latent"], tensors["sigma"],
            tensors["conditioning"], self.device, self.dtype,
            multiplier=params.get("multiplier", 1.0),
        )
        with torch.no_grad():
            scores = self._mm.btrm_head(hidden)
        return pack_response("ok", metadata={
            "scores": scores.detach().cpu().tolist()})

    def handle_train_btrm_step(self, params, tensors):
        """One BTRM optimizer step from labeled examples."""
        self._mm.ensure_diffusion()
        assert self._mm.btrm_head is not None, "BTRM head not injected"
        assert self._mm.btrm_optimizer is not None, \
            "BTRM optimizer not created (pass lr to inject_btrm_head)"
        self._mm.configure_sage_if_needed(params.get("attention_backend", "sdpa"))
        set_attention_backend(params.get("attention_backend", "sdpa"))
        metadata = train_btrm_step(
            self._mm.diff_model, self._mm.btrm_head, self._mm.btrm_optimizer,
            self.device, self.dtype, params, tensors,
        )
        print(f"  [train_btrm_step] loss={metadata['loss']:.4f} "
              f"bt={metadata['bt_loss']:.4f} logsq={metadata['logsq_loss']:.4f}")
        return pack_response("ok", metadata=metadata)

    def handle_accumulate_policy_gradients(self, params, tensors):
        """Accumulate REINFORCE gradients on server-side LoRA params."""
        self._mm.ensure_diffusion()
        metadata = accumulate_policy_gradients(
            self._mm.diff_model, self.device, self.dtype, params, tensors,
        )
        return pack_response("ok", metadata=metadata)

    def handle_policy_optimizer_step(self, params, tensors):
        """Clip gradients, step policy optimizer, zero gradients."""
        self._mm.ensure_diffusion()
        metadata = policy_optimizer_step(
            self._mm.diff_model, self._mm.policy_optimizers,
            self.device, self.dtype, params,
        )
        print(f"  [policy_optimizer_step] {params['adapter_name']}: "
              f"grad_norm={metadata['grad_norm']:.3e}")
        return pack_response("ok", metadata=metadata)

    # ------------------------------------------------------------------
    # Dispatch and main loop
    # ------------------------------------------------------------------

    _HANDLERS = {
        "encode_prompt": "handle_encode_prompt",
        "sample_trajectory": "handle_sample_trajectory",
        "sample_trajectory_packed": "handle_sample_trajectory_packed",
        "vae_encode": "handle_vae_encode",
        "vae_decode": "handle_vae_decode",
        "warmup": "handle_warmup",
        "warmup_packed": "handle_warmup_packed",
        "status": "handle_status",
        "free": "handle_free",
        "inject_lora": "handle_inject_lora",
        "allocate_adapter": "handle_allocate_adapter",
        "init_adapter_weights": "handle_init_adapter_weights",
        "update_lora_weights": "handle_update_lora_weights",
        "set_adapter_config": "handle_set_adapter_config",
        "get_lora_state_dict": "handle_get_lora_state_dict",
        "dump_all_loras": "handle_dump_all_loras",
        "inject_btrm_head": "handle_inject_btrm_head",
        "score_btrm": "handle_score_btrm",
        "train_btrm_step": "handle_train_btrm_step",
        "accumulate_policy_gradients": "handle_accumulate_policy_gradients",
        "policy_optimizer_step": "handle_policy_optimizer_step",
    }

    def dispatch(self, method, params, tensors):
        handler_name = self._HANDLERS.get(method)
        if handler_name is None:
            return pack_response("error", metadata={
                "error": f"Unknown method: {method}"})
        return getattr(self, handler_name)(params, tensors)

    def serve(self, endpoint: str = "tcp://*:5555"):
        """Run the server main loop."""
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(endpoint)
        print(f"Inference server listening on {endpoint}")
        print(f"  Models: diff={self._mm.fp8_diff_path}")
        print(f"          te={self._mm.te_path}")
        print(f"          vae={self._mm.vae_path}")
        print(f"  Device: {self.device}, dtype: {self.dtype}")

        while True:
            try:
                frames = socket.recv_multipart()
                method, params, tensors = unpack_request(frames)
                print(f"  [{method}] ...", end="", flush=True)
                t0 = time.perf_counter()
                response_frames = self.dispatch(method, params, tensors)
                print(f" {time.perf_counter() - t0:.2f}s")
                socket.send_multipart(response_frames)
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception:
                tb = traceback.format_exc()
                print(f"\n  ERROR:\n{tb}")
                try:
                    socket.send_multipart(
                        pack_response("error", metadata={"error": tb}))
                except Exception:
                    pass  # socket may be in bad state

        socket.close()
        ctx.term()
        self._mm.free_all()


def main():
    parser = argparse.ArgumentParser(
        description="futudiffu inference server (ZeroMQ)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--fp8-diff", required=True,
                        help="Path to FP8 blockwise diffusion model")
    parser.add_argument("--te", required=True,
                        help="Path to text encoder safetensors")
    parser.add_argument("--vae", required=True,
                        help="Path to VAE safetensors")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    server = InferenceServer(
        fp8_diff_path=args.fp8_diff, te_path=args.te,
        vae_path=args.vae, tokenizer_path=args.tokenizer,
        device=args.device, dtype=args.dtype,
    )
    server.serve(f"tcp://*:{args.port}")


if __name__ == "__main__":
    main()
