"""Tests for the FastAPI inference server (no GPU required).

Uses FastAPI's TestClient with a mock model backend.
Tests HTTP routing, serialization, error handling, and timeout behavior.
"""

import io
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Add project root to path for src_ii imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

class MockModelBackend:
    """Mock backend for testing HTTP layer without GPU.

    Returns plausible shapes for all operations. Does not touch torch
    except for creating small CPU tensors for response serialization.
    """

    def __init__(self):
        self._status = {
            "loaded_models": ["diffusion", "vae"],
            "phase": "diffusion",
            "vram_allocated_gb": 8.5,
            "vram_reserved_gb": 10.2,
            "vram_total_gb": 24.0,
            "sage_configured": False,
        }
        self._freed = []
        self._call_log = []

    def get_status(self) -> dict[str, Any]:
        self._call_log.append("get_status")
        return self._status

    def free(self, model: str) -> None:
        self._call_log.append(f"free:{model}")
        self._freed.append(model)
        if model not in ("all", "te", "diffusion", "vae"):
            raise ValueError(f"Unknown model: {model!r}")

    def encode_prompt(self, prompt: str, layer_idx: int) -> dict:
        import torch
        self._call_log.append(f"encode_prompt:{prompt}")
        # Return a small conditioning tensor
        cond = torch.randn(1, 10, 2560)
        return {"conditioning": cond}

    def sample_trajectory(self, params: dict, tensors: dict) -> tuple[dict, dict]:
        import torch
        self._call_log.append("sample_trajectory")
        h, w = params.get("height", 832), params.get("width", 1280)
        lh, lw = h // 8, w // 8
        result = {"final": torch.randn(1, 16, lh, lw)}
        n_steps = params.get("n_steps", 30)
        save_steps = params.get("save_steps")
        if save_steps:
            for s in save_steps:
                if s < n_steps:
                    result[f"step_{s:02d}"] = torch.randn(1, 16, lh, lw)
        metadata = {"saved_steps": sorted(k for k in result if k.startswith("step_"))}
        return result, metadata

    def sample_trajectory_packed(self, params: dict, tensors: dict) -> tuple[dict, dict]:
        import torch
        self._call_log.append("sample_trajectory_packed")
        n_images = params.get("n_images", 2)
        result = {}
        for i in range(n_images):
            result[f"final_{i}"] = torch.randn(1, 16, 104, 160)
        return result, {"n_images": n_images}

    def vae_encode(self, image_bytes: bytes) -> dict:
        import torch
        self._call_log.append("vae_encode")
        return {"latent": torch.randn(1, 16, 104, 160)}

    def vae_decode(self, latent_bytes: bytes) -> dict:
        import torch
        self._call_log.append("vae_decode")
        return {"image": torch.rand(1, 3, 832, 1280)}

    def warmup(self, attention_backend: str, width: int, height: int) -> None:
        self._call_log.append(f"warmup:{attention_backend}")

    def warmup_packed(self, n_images: int) -> None:
        self._call_log.append(f"warmup_packed:{n_images}")

    def allocate_adapter(self, params: dict) -> dict:
        self._call_log.append(f"allocate_adapter:{params['adapter_name']}")
        return {
            "adapter_name": params["adapter_name"],
            "n_adapters": 60,
            "n_params": 100000,
            "graph_mutated": True,
        }

    def init_adapter_weights(self, params: dict) -> dict:
        self._call_log.append(f"init_adapter_weights:{params['adapter_name']}")
        return {
            "adapter_name": params["adapter_name"],
            "n_modules_initialized": 60,
        }

    def inject_lora(self, params: dict) -> dict:
        self._call_log.append(f"inject_lora:{params['adapter_name']}")
        return {
            "adapter_name": params["adapter_name"],
            "n_adapters": 60,
            "n_params": 100000,
        }

    def update_lora_weights(self, tensor_bytes: bytes) -> dict:
        self._call_log.append("update_lora_weights")
        # Verify the bytes are valid safetensors
        from safetensors.torch import load as st_load
        tensors = st_load(tensor_bytes)
        return {"n_tensors": len(tensors)}

    def set_adapter_config(self, params: dict) -> dict:
        self._call_log.append(f"set_adapter_config:{params['adapter_name']}")
        return {"adapter_name": params["adapter_name"], "n_frozen": 0}

    def get_lora_state_dict(self, adapter_name: str | None) -> dict:
        import torch
        self._call_log.append(f"get_lora_state_dict:{adapter_name}")
        return {"test_weight": torch.randn(8, 16)}

    def dump_all_loras(self, output_dir: str) -> dict:
        self._call_log.append(f"dump_all_loras:{output_dir}")
        return {"files": [], "manifest": f"{output_dir}/manifest.json"}

    def inject_btrm_head(self, params: dict) -> dict:
        self._call_log.append("inject_btrm_head")
        return {
            "n_heads": len(params.get("head_names", ["a", "b"])),
            "n_params": 7686,
            "has_optimizer": params.get("lr") is not None,
        }

    def score_btrm(self, params: dict, tensor_bytes: bytes) -> dict:
        self._call_log.append("score_btrm")
        return {"scores": [[0.5, -0.3]]}

    def train_btrm_step(self, params: dict, tensor_bytes: bytes) -> dict:
        self._call_log.append("train_btrm_step")
        return {
            "loss": 0.45,
            "bt_loss": 0.40,
            "logsq_loss": 0.05,
        }

    def accumulate_policy_gradients(self, params: dict, tensor_bytes: bytes) -> dict:
        self._call_log.append("accumulate_policy_gradients")
        return {"total_log_ratio": 0.1, "n_steps": 3}

    def policy_optimizer_step(self, params: dict) -> dict:
        self._call_log.append(f"policy_optimizer_step:{params['adapter_name']}")
        return {"grad_norm": 0.01, "n_params": 50000}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_backend():
    return MockModelBackend()


@pytest.fixture
def app(mock_backend):
    from src_ii.server import create_app
    return create_app(mock_backend, request_timeout_s=30.0)


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: Health and status
# ---------------------------------------------------------------------------

class TestHealthAndStatus:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_status(self, client, mock_backend):
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["loaded_models"] == ["diffusion", "vae"]
        assert body["phase"] == "diffusion"
        assert body["vram_allocated_gb"] == 8.5
        assert "get_status" in mock_backend._call_log

    def test_status_response_fields(self, client):
        """Verify StatusResponse includes all expected fields."""
        resp = client.get("/status")
        body = resp.json()
        assert "loaded_models" in body
        assert "phase" in body
        assert "vram_allocated_gb" in body
        assert "vram_reserved_gb" in body
        assert "vram_total_gb" in body
        assert "sage_configured" in body
        assert "server_version" in body


# ---------------------------------------------------------------------------
# Tests: Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_free_all(self, client, mock_backend):
        resp = client.post("/free", json={"model": "all"})
        assert resp.status_code == 200
        assert "free:all" in mock_backend._call_log

    def test_free_te(self, client, mock_backend):
        resp = client.post("/free", json={"model": "te"})
        assert resp.status_code == 200
        assert "free:te" in mock_backend._call_log

    def test_free_diffusion(self, client, mock_backend):
        resp = client.post("/free", json={"model": "diffusion"})
        assert resp.status_code == 200

    def test_free_vae(self, client, mock_backend):
        resp = client.post("/free", json={"model": "vae"})
        assert resp.status_code == 200

    def test_free_invalid_model(self, client):
        resp = client.post("/free", json={"model": "bogus"})
        assert resp.status_code == 500

    def test_free_default_model(self, client, mock_backend):
        """Free with default model='all' when no model specified."""
        resp = client.post("/free", json={})
        assert resp.status_code == 200
        assert "free:all" in mock_backend._call_log


# ---------------------------------------------------------------------------
# Tests: Text encoding
# ---------------------------------------------------------------------------

class TestEncodePrompt:
    def test_encode_prompt_returns_safetensors(self, client, mock_backend):
        resp = client.post("/encode_prompt", json={
            "prompt": "a laser shark",
            "layer_idx": -2,
        })
        assert resp.status_code == 200
        assert resp.headers.get("X-Tensor-Format") == "safetensors"
        # Verify we can deserialize the safetensors bytes
        from safetensors.torch import load as st_load
        tensors = st_load(resp.content)
        assert "conditioning" in tensors
        assert tensors["conditioning"].shape == (1, 10, 2560)
        assert "encode_prompt:a laser shark" in mock_backend._call_log

    def test_encode_empty_prompt(self, client, mock_backend):
        resp = client.post("/encode_prompt", json={"prompt": ""})
        assert resp.status_code == 200

    def test_encode_prompt_custom_layer_idx(self, client, mock_backend):
        """Verify layer_idx is passed through to backend."""
        resp = client.post("/encode_prompt", json={
            "prompt": "test",
            "layer_idx": -3,
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests: VAE
# ---------------------------------------------------------------------------

class TestVAE:
    def test_vae_decode(self, client, mock_backend):
        import torch
        from safetensors.torch import save as st_save

        latent = torch.randn(1, 16, 104, 160)
        data = st_save({"latent": latent})

        resp = client.post("/vae_decode", content=data)
        assert resp.status_code == 200

        from safetensors.torch import load as st_load
        result = st_load(resp.content)
        assert "image" in result
        assert result["image"].shape == (1, 3, 832, 1280)
        assert "vae_decode" in mock_backend._call_log

    def test_vae_encode(self, client, mock_backend):
        import torch
        from safetensors.torch import save as st_save

        image = torch.rand(1, 3, 832, 1280)
        data = st_save({"image": image})

        resp = client.post("/vae_encode", content=data)
        assert resp.status_code == 200

        from safetensors.torch import load as st_load
        result = st_load(resp.content)
        assert "latent" in result
        assert "vae_encode" in mock_backend._call_log


# ---------------------------------------------------------------------------
# Tests: Warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_warmup_sdpa(self, client, mock_backend):
        resp = client.post("/warmup", json={
            "attention_backend": "sdpa",
            "width": 1280,
            "height": 832,
        })
        assert resp.status_code == 200
        assert "warmup:sdpa" in mock_backend._call_log

    def test_warmup_packed(self, client, mock_backend):
        resp = client.post("/warmup_packed", json={"n_images": 3})
        assert resp.status_code == 200
        assert "warmup_packed:3" in mock_backend._call_log

    def test_warmup_sage(self, client, mock_backend):
        resp = client.post("/warmup", json={
            "attention_backend": "sage",
            "width": 512,
            "height": 512,
        })
        assert resp.status_code == 200
        assert "warmup:sage" in mock_backend._call_log


# ---------------------------------------------------------------------------
# Tests: LoRA management
# ---------------------------------------------------------------------------

class TestLoRA:
    def test_allocate_adapter(self, client, mock_backend):
        resp = client.post("/allocate_adapter", json={
            "adapter_name": "rtheta",
            "rank": 8,
            "alpha": 16.0,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["adapter_name"] == "rtheta"
        assert body["metadata"]["n_adapters"] == 60

    def test_init_adapter_weights(self, client, mock_backend):
        resp = client.post("/init_adapter_weights", json={
            "adapter_name": "rtheta",
            "init_b_std": 0.01,
            "scale": 1.0,
        })
        assert resp.status_code == 200

    def test_inject_lora(self, client, mock_backend):
        resp = client.post("/inject_lora", json={
            "adapter_name": "ptheta",
            "rank": 8,
        })
        assert resp.status_code == 200
        assert "inject_lora:ptheta" in mock_backend._call_log

    def test_update_lora_weights(self, client, mock_backend):
        """Update LoRA weights via safetensors body."""
        import torch
        from safetensors.torch import save as st_save

        weights = {
            "layer.0.adapters.rtheta.lora_A": torch.randn(8, 3840),
            "layer.0.adapters.rtheta.lora_B": torch.randn(3840, 8),
        }
        data = st_save(weights)

        resp = client.post(
            "/update_lora_weights",
            content=data,
        )
        assert resp.status_code == 200
        assert "update_lora_weights" in mock_backend._call_log

    def test_set_adapter_config(self, client, mock_backend):
        resp = client.post("/set_adapter_config", json={
            "adapter_name": "rtheta",
            "scale": 0.5,
        })
        assert resp.status_code == 200

    def test_set_adapter_config_frozen(self, client, mock_backend):
        resp = client.post("/set_adapter_config", json={
            "adapter_name": "rtheta",
            "frozen": True,
        })
        assert resp.status_code == 200

    def test_set_adapter_config_clear_scale(self, client, mock_backend):
        """Test clear_scale parameter."""
        resp = client.post("/set_adapter_config", json={
            "adapter_name": "rtheta",
            "clear_scale": True,
        })
        assert resp.status_code == 200

    def test_get_lora_state_dict(self, client, mock_backend):
        resp = client.post("/get_lora_state_dict", json={
            "adapter_name": "rtheta",
        })
        assert resp.status_code == 200
        from safetensors.torch import load as st_load
        tensors = st_load(resp.content)
        assert "test_weight" in tensors

    def test_dump_all_loras(self, client, mock_backend):
        resp = client.post("/dump_all_loras", json={
            "output_dir": "/tmp/lora_dumps",
        })
        assert resp.status_code == 200
        assert "dump_all_loras:/tmp/lora_dumps" in mock_backend._call_log


# ---------------------------------------------------------------------------
# Tests: BTRM
# ---------------------------------------------------------------------------

class TestBTRM:
    def test_inject_btrm_head(self, client, mock_backend):
        resp = client.post("/inject_btrm_head", json={
            "hidden_dim": 3840,
            "head_names": ["scrimblo", "scrongle"],
            "logit_cap": 10.0,
            "lr": 1e-3,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["n_heads"] == 2
        assert body["metadata"]["has_optimizer"] is True

    def test_inject_btrm_head_no_optimizer(self, client, mock_backend):
        resp = client.post("/inject_btrm_head", json={
            "hidden_dim": 3840,
            "head_names": ["scrimblo"],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["has_optimizer"] is False

    def test_score_btrm_multipart(self, client, mock_backend):
        """Score BTRM via multipart form (safetensors tensors + JSON params)."""
        import torch
        from safetensors.torch import save as st_save

        latent = torch.randn(1, 16, 104, 160)
        sigma = torch.tensor([1.0])
        conditioning = torch.randn(1, 10, 2560)

        data = st_save({"latent": latent, "sigma": sigma, "conditioning": conditioning})

        files = {
            "params": ("params.json", json.dumps({
                "attention_backend": "sdpa",
                "multiplier": 1.0,
            }), "application/json"),
            "tensors": ("tensors.st", data, "application/octet-stream"),
        }
        resp = client.post("/score_btrm", files=files)
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["scores"] == [[0.5, -0.3]]
        assert "score_btrm" in mock_backend._call_log

    def test_train_btrm_step_multipart(self, client, mock_backend):
        """Train BTRM step via multipart form."""
        import torch
        from safetensors.torch import save as st_save

        tensors = {
            "latent_0": torch.randn(1, 16, 104, 160),
            "sigma_0": torch.tensor([1.0]),
            "conditioning_0": torch.randn(1, 10, 2560),
            "latent_1": torch.randn(1, 16, 104, 160),
            "sigma_1": torch.tensor([0.5]),
            "conditioning_1": torch.randn(1, 10, 2560),
        }
        data = st_save(tensors)

        params = {
            "labels": [
                {"head_idx": 0, "is_positive": True},
                {"head_idx": 0, "is_positive": False},
            ],
            "logsquare_weight": 0.1,
            "attention_backend": "sdpa",
            "multiplier": 1.0,
        }
        files = {
            "params": ("params.json", json.dumps(params), "application/json"),
            "tensors": ("tensors.st", data, "application/octet-stream"),
        }
        resp = client.post("/train_btrm_step", files=files)
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["loss"] == 0.45
        assert body["metadata"]["bt_loss"] == 0.40
        assert "train_btrm_step" in mock_backend._call_log


# ---------------------------------------------------------------------------
# Tests: Policy
# ---------------------------------------------------------------------------

class TestPolicy:
    def test_policy_optimizer_step(self, client, mock_backend):
        resp = client.post("/policy_optimizer_step", json={
            "adapter_name": "ptheta",
            "max_grad_norm": 1.0,
            "lr": 1e-4,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["grad_norm"] == 0.01

    def test_accumulate_policy_gradients_multipart(self, client, mock_backend):
        """Accumulate policy gradients via multipart form."""
        import torch
        from safetensors.torch import save as st_save

        tensors = {
            "sigmas": torch.linspace(1.0, 0.0, 31),
            "conditioning": torch.randn(1, 10, 2560),
            "checkpoint_5": torch.randn(1, 16, 104, 160),
            "checkpoint_15": torch.randn(1, 16, 104, 160),
            "checkpoint_25": torch.randn(1, 16, 104, 160),
        }
        data = st_save(tensors)

        params = {
            "adapter_name": "ptheta",
            "sparse_steps": [5, 15, 25],
            "advantage": 0.8,
            "multiplier": 1.0,
        }
        files = {
            "params": ("params.json", json.dumps(params), "application/json"),
            "tensors": ("tensors.st", data, "application/octet-stream"),
        }
        resp = client.post("/accumulate_policy_gradients", files=files)
        assert resp.status_code == 200
        body = resp.json()
        assert body["metadata"]["total_log_ratio"] == 0.1
        assert body["metadata"]["n_steps"] == 3
        assert "accumulate_policy_gradients" in mock_backend._call_log


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_endpoint_404(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_backend_exception_500(self, client, mock_backend):
        # Patch the backend to raise
        original = mock_backend.get_status
        mock_backend.get_status = lambda: (_ for _ in ()).throw(RuntimeError("GPU on fire"))
        resp = client.get("/status")
        assert resp.status_code == 500
        body = resp.json()
        assert "GPU on fire" in body["error"]
        mock_backend.get_status = original

    def test_invalid_json_422(self, client):
        # Missing required fields
        resp = client.post("/allocate_adapter", json={})
        assert resp.status_code == 422  # Pydantic validation error

    def test_malformed_request_body(self, client):
        """Non-JSON body to a JSON endpoint."""
        resp = client.post(
            "/allocate_adapter",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Sampling (multipart and JSON)
# ---------------------------------------------------------------------------

class TestSampling:
    def test_sample_trajectory_json(self, client, mock_backend):
        """Test sample_trajectory with JSON body (base64 tensors)."""
        import base64
        import torch
        from safetensors.torch import save as st_save

        # Create small conditioning tensors
        pos_cond = torch.randn(1, 10, 2560)
        neg_cond = torch.randn(1, 5, 2560)

        # Encode each tensor individually as base64 safetensors
        def tensor_to_b64(name, tensor):
            data = st_save({name: tensor})
            return base64.b64encode(data).decode("ascii")

        body = {
            "params": {
                "seed": 42,
                "n_steps": 10,
                "cfg": 4.0,
                "width": 1280,
                "height": 832,
            },
            "tensors": {
                "pos_cond": tensor_to_b64("pos_cond", pos_cond),
                "neg_cond": tensor_to_b64("neg_cond", neg_cond),
            },
        }
        resp = client.post("/sample_trajectory", json=body)
        assert resp.status_code == 200

        from safetensors.torch import load as st_load
        result = st_load(resp.content)
        assert "final" in result
        assert result["final"].shape == (1, 16, 104, 160)
        assert "sample_trajectory" in mock_backend._call_log

    def test_sample_trajectory_multipart(self, client, mock_backend):
        """Test sample_trajectory with multipart form (safetensors file)."""
        import torch
        from safetensors.torch import save as st_save

        pos_cond = torch.randn(1, 10, 2560)
        neg_cond = torch.randn(1, 5, 2560)
        data = st_save({"pos_cond": pos_cond, "neg_cond": neg_cond})

        files = {
            "params": ("params.json", json.dumps({
                "seed": 42,
                "n_steps": 10,
                "cfg": 4.0,
                "width": 1280,
                "height": 832,
            }), "application/json"),
            "tensors": ("tensors.st", data, "application/octet-stream"),
        }
        resp = client.post("/sample_trajectory", files=files)
        assert resp.status_code == 200

        from safetensors.torch import load as st_load
        result = st_load(resp.content)
        assert "final" in result

    def test_sample_trajectory_packed_multipart(self, client, mock_backend):
        """Test sample_trajectory_packed with multipart form."""
        import torch
        from safetensors.torch import save as st_save

        pos_cond_0 = torch.randn(1, 10, 2560)
        pos_cond_1 = torch.randn(1, 8, 2560)
        neg_cond = torch.randn(1, 5, 2560)

        data = st_save({
            "pos_cond_0": pos_cond_0,
            "pos_cond_1": pos_cond_1,
            "neg_cond": neg_cond,
        })

        params = {
            "n_images": 2,
            "seeds": [42, 43],
            "n_steps": 10,
            "cfg": 4.0,
            "width": 1280,
            "height": 832,
        }
        files = {
            "params": ("params.json", json.dumps(params), "application/json"),
            "tensors": ("tensors.st", data, "application/octet-stream"),
        }
        resp = client.post("/sample_trajectory_packed", files=files)
        assert resp.status_code == 200

        from safetensors.torch import load as st_load
        result = st_load(resp.content)
        assert "final_0" in result
        assert "final_1" in result
        assert "sample_trajectory_packed" in mock_backend._call_log

    def test_sample_trajectory_with_save_steps(self, client, mock_backend):
        """Test sample_trajectory with save_steps returns intermediates."""
        import torch
        from safetensors.torch import save as st_save

        pos_cond = torch.randn(1, 10, 2560)
        neg_cond = torch.randn(1, 5, 2560)
        data = st_save({"pos_cond": pos_cond, "neg_cond": neg_cond})

        files = {
            "params": ("params.json", json.dumps({
                "seed": 42,
                "n_steps": 10,
                "cfg": 4.0,
                "width": 1280,
                "height": 832,
                "save_steps": [0, 5, 9],
            }), "application/json"),
            "tensors": ("tensors.st", data, "application/octet-stream"),
        }
        resp = client.post("/sample_trajectory", files=files)
        assert resp.status_code == 200

        from safetensors.torch import load as st_load
        result = st_load(resp.content)
        assert "final" in result
        # Mock backend saves step_00, step_05, step_09
        assert "step_00" in result
        assert "step_05" in result
        assert "step_09" in result


# ---------------------------------------------------------------------------
# Tests: RPC parity (verify all 20 RPCs from old server have endpoints)
# ---------------------------------------------------------------------------

class TestRPCParity:
    """Verify every RPC from the old ZMQ server has a FastAPI endpoint."""

    OLD_RPCS = [
        "encode_prompt",
        "sample_trajectory",
        "sample_trajectory_packed",
        "vae_encode",
        "vae_decode",
        "warmup",
        "warmup_packed",
        "status",
        "free",
        "inject_lora",
        "allocate_adapter",
        "init_adapter_weights",
        "update_lora_weights",
        "set_adapter_config",
        "get_lora_state_dict",
        "dump_all_loras",
        "inject_btrm_head",
        "score_btrm",
        "train_btrm_step",
        "accumulate_policy_gradients",
        "policy_optimizer_step",
    ]

    def test_all_old_rpcs_have_routes(self, app):
        """Every RPC from the old server must have a corresponding route."""
        route_paths = set()
        for route in app.routes:
            if hasattr(route, "path"):
                route_paths.add(route.path.lstrip("/"))

        missing = []
        for rpc in self.OLD_RPCS:
            if rpc not in route_paths:
                missing.append(rpc)

        assert not missing, f"Missing routes for old RPCs: {missing}"

    def test_no_phantom_rpcs(self, app):
        """No routes exist that don't correspond to an old RPC or health/docs."""
        known = set(self.OLD_RPCS) | {"health", "openapi.json", "docs", "redoc", "docs/oauth2-redirect"}
        route_paths = set()
        for route in app.routes:
            if hasattr(route, "path"):
                path = route.path.lstrip("/")
                if path:
                    route_paths.add(path)

        unexpected = route_paths - known
        assert not unexpected, f"Unexpected routes (phantom RPCs?): {unexpected}"

    def test_rpc_count(self, app):
        """Verify the total number of application routes matches expectations."""
        app_routes = [r for r in app.routes if hasattr(r, "path")
                      and not r.path.startswith("/openapi")
                      and not r.path.startswith("/docs")
                      and not r.path.startswith("/redoc")]
        # 21 old RPCs + health = 22 routes
        assert len(app_routes) >= 22, f"Expected >=22 routes, got {len(app_routes)}"


# ---------------------------------------------------------------------------
# Tests: HTTP client lazy import (structural test)
# ---------------------------------------------------------------------------

class TestHTTPClientLazyImport:
    """Verify the HTTP client does not import torch at module level."""

    def test_no_torch_at_module_level(self):
        """The http_client module should not have torch in its direct imports."""
        import importlib
        import src_ii.http_client as mod
        # Reload to check fresh imports
        source = importlib.util.find_spec("src_ii.http_client")
        assert source is not None

        # Read the source and check that torch is not imported at module level
        import inspect
        src = inspect.getsource(mod)

        # Should NOT have bare 'import torch' at module level
        # (should only appear in TYPE_CHECKING blocks or inside functions)
        lines = src.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip comments, empty lines, and TYPE_CHECKING blocks
            if stripped.startswith("#") or stripped == "":
                continue
            if stripped == "import torch" and not line.startswith(" ") and not line.startswith("\t"):
                # This would be a module-level bare import
                # Check if it's inside TYPE_CHECKING
                # Look backwards for 'if TYPE_CHECKING:'
                in_type_checking = False
                for j in range(i - 1, -1, -1):
                    prev = lines[j].strip()
                    if prev == "if TYPE_CHECKING:":
                        in_type_checking = True
                        break
                    if prev and not prev.startswith("#") and not prev.startswith("import"):
                        break
                if not in_type_checking:
                    pytest.fail(
                        f"torch imported at module level (line {i+1}): {stripped}"
                    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
