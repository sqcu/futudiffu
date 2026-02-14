"""ZeroMQ serialization protocol for inference server communication.

Message format (ZMQ multipart frames):
    Request:  [metadata_json, tensor_0_bytes, tensor_1_bytes, ...]
    Response: [metadata_json, tensor_0_bytes, tensor_1_bytes, ...]

Metadata JSON schema:
    {
        "method": "encode_prompt",           # request only
        "params": {"prompt": "...", ...},    # request only
        "status": "ok",                      # response only
        "error": "...",                      # response only (when status="error")
        "tensors": [
            {"name": "pos_cond", "shape": [1, 128, 2560], "dtype": "bfloat16"},
            ...
        ]
    }

No pickle anywhere. Tensors are raw bytes with shape/dtype in the JSON envelope.
"""

import json

import numpy as np
import torch

# Maps between torch dtype names and numpy dtypes for serialization.
# Only dtypes that actually appear in the pipeline are included.
_TORCH_TO_NP = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": None,  # no numpy equivalent, handled specially
    "uint8": np.uint8,
    "int64": np.int64,
}

_DTYPE_NAMES = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.uint8: "uint8",
    torch.int64: "int64",
}


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Serialize a tensor to raw bytes. Always contiguous, CPU."""
    t = t.detach().contiguous().cpu()
    # bfloat16 has no numpy dtype -- view as uint16 for byte-level transfer
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy().tobytes()
    return t.numpy().tobytes()


def bytes_to_tensor(b: bytes, shape: list[int], dtype_name: str) -> torch.Tensor:
    """Deserialize raw bytes to a torch tensor on CPU."""
    if dtype_name == "bfloat16":
        arr = np.frombuffer(b, dtype=np.uint16).reshape(shape)
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    np_dtype = _TORCH_TO_NP[dtype_name]
    arr = np.frombuffer(b, dtype=np_dtype).reshape(shape)
    return torch.from_numpy(arr.copy())


def _tensor_descriptor(name: str, t: torch.Tensor) -> dict:
    """Build a JSON-serializable descriptor for a tensor."""
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": _DTYPE_NAMES[t.dtype],
    }


def pack_request(
    method: str,
    params: dict | None = None,
    tensors: dict[str, torch.Tensor] | None = None,
) -> list[bytes]:
    """Serialize a request into ZMQ multipart frames.

    Args:
        method: RPC method name (e.g. "encode_prompt", "sample_trajectory").
        params: JSON-serializable parameter dict.
        tensors: Named tensors to include as binary frames.

    Returns:
        List of bytes frames: [metadata_json, tensor_0_bytes, ...].
    """
    tensors = tensors or {}
    descriptors = []
    frames = []
    for name, t in tensors.items():
        descriptors.append(_tensor_descriptor(name, t))
        frames.append(tensor_to_bytes(t))

    metadata = {
        "method": method,
        "params": params or {},
        "tensors": descriptors,
    }
    return [json.dumps(metadata).encode("utf-8")] + frames


def unpack_request(frames: list[bytes]) -> tuple[str, dict, dict[str, torch.Tensor]]:
    """Deserialize a request from ZMQ multipart frames.

    Returns:
        (method, params, tensors) where tensors is {name: Tensor}.
    """
    metadata = json.loads(frames[0])
    tensors = {}
    for i, desc in enumerate(metadata.get("tensors", [])):
        tensors[desc["name"]] = bytes_to_tensor(
            frames[1 + i], desc["shape"], desc["dtype"]
        )
    return metadata["method"], metadata.get("params", {}), tensors


def pack_response(
    status: str = "ok",
    tensors: dict[str, torch.Tensor] | None = None,
    metadata: dict | None = None,
) -> list[bytes]:
    """Serialize a response into ZMQ multipart frames.

    Args:
        status: "ok" or "error".
        tensors: Named result tensors.
        metadata: Additional JSON-serializable metadata (e.g. error message, stats).

    Returns:
        List of bytes frames: [metadata_json, tensor_0_bytes, ...].
    """
    tensors = tensors or {}
    extra = metadata or {}
    descriptors = []
    frames = []
    for name, t in tensors.items():
        descriptors.append(_tensor_descriptor(name, t))
        frames.append(tensor_to_bytes(t))

    envelope = {
        "status": status,
        "tensors": descriptors,
        **extra,
    }
    return [json.dumps(envelope).encode("utf-8")] + frames


def unpack_response(
    frames: list[bytes],
) -> tuple[str, dict[str, torch.Tensor], dict]:
    """Deserialize a response from ZMQ multipart frames.

    Returns:
        (status, tensors, metadata) where tensors is {name: Tensor}
        and metadata is the full JSON envelope (minus tensor descriptors).
    """
    envelope = json.loads(frames[0])
    tensors = {}
    for i, desc in enumerate(envelope.get("tensors", [])):
        tensors[desc["name"]] = bytes_to_tensor(
            frames[1 + i], desc["shape"], desc["dtype"]
        )
    status = envelope.pop("status")
    envelope.pop("tensors", None)
    return status, tensors, envelope
