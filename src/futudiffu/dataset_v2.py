"""Dataset V2: Parquet index + safetensors blob storage for BTRM trajectories.

Replaces the v1 per-directory format (traj_NNNNNN/ with .pt files) with a
compact blob-based layout. Designed for datasets up to ~50K trajectories
(~200 GB) on a single NVMe.

Layout:
    dataset_v2/
        index.parquet                  # One row per trajectory
        blobs/
            blob_000.safetensors       # Sealed (immutable) blobs
            blob_001.safetensors       # ~1 GB each, sequential naming
        _write_lock                    # Advisory lockfile

Write model: tensors accumulate in memory. When the buffer hits max_blob_bytes,
the entire blob is written once, sealed, and the parquet index is updated
atomically. No WIP blob on disk = no crash corruption, no write amplification.

Requires pyarrow for parquet I/O. If pyarrow is not installed, import of
this module will raise an ImportError with install instructions.
"""

from __future__ import annotations

import os
import random
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError(
        "pyarrow is required for dataset_v2 but is not installed.\n"
        "Install it into the project venv from Windows PowerShell:\n"
        "    uv add pyarrow>=14.0\n"
        "Or add 'pyarrow>=14.0' to [project.dependencies] in pyproject.toml "
        "and run 'uv sync'."
    )

from safetensors import safe_open
from safetensors.torch import save_file


# ---------------------------------------------------------------------------
# Parquet schema
# ---------------------------------------------------------------------------

INDEX_SCHEMA = pa.schema([
    ("traj_id", pa.int64()),
    ("prompt", pa.large_utf8()),
    ("prompt_idx", pa.int32()),
    ("seed", pa.uint64()),
    ("cfg", pa.float32()),
    ("width", pa.int32()),
    ("height", pa.int32()),
    ("n_steps", pa.int32()),
    ("attention_backend", pa.utf8()),
    ("batch_type", pa.utf8()),
    ("denoise", pa.float32()),          # nullable
    ("image_file", pa.utf8()),          # nullable
    ("is_gold", pa.bool_()),
    ("batch_idx", pa.int32()),
    ("packed", pa.bool_()),
    ("step_indices", pa.list_(pa.int32())),
    ("has_final", pa.bool_()),
    ("latent_channels", pa.int32()),
    ("latent_height", pa.int32()),
    ("latent_width", pa.int32()),
    ("latent_dtype", pa.utf8()),
    ("blob_file", pa.utf8()),
    ("key_prefix", pa.utf8()),
    ("n_tensors", pa.int32()),
    ("bytes_total", pa.int64()),
    ("sampling_shift", pa.float32()),    # nullable: SD3 Eq.23 alpha, null = 1.0 legacy
    ("timing_seconds", pa.float32()),   # nullable
    ("created_at", pa.timestamp("us", tz="UTC")),
    ("parent_traj_id", pa.int64()),       # nullable: null for non-i2i2i
    ("parent_step", pa.utf8()),           # nullable: e.g. "step_14"
    ("parent_denoise", pa.float32()),     # nullable: denoise used in i2i2i pass
    # --- Sampling state identity hashes (nullable; backfilled for existing data) ---
    ("model_state_hash", pa.utf8()),      # nullable: hex digest of model state
    ("base_model_hash", pa.utf8()),       # nullable: hex digest of base model weights
    ("adapter_set_hash", pa.utf8()),      # nullable: hex digest of active adapter set ("" = none)
    ("trajectory_hash", pa.utf8()),       # nullable: hex digest of full trajectory identity
    ("active_adapters", pa.utf8()),       # nullable: JSON list of {"name", "strength", "param_hash"}
])

# Columns that get dictionary encoding (low cardinality).
_DICT_ENCODE_COLS = [
    "attention_backend",
    "batch_type",
    "latent_dtype",
    "blob_file",
    "parent_step",
]

# Parquet write settings per spec.
_PARQUET_WRITE_KWARGS = dict(
    compression="zstd",
    compression_level=3,
    write_statistics=True,
    use_dictionary=_DICT_ENCODE_COLS,
    row_group_size=10_000,
)

# WIP blob filename constant.
_WIP_BLOB = "blob_wip.safetensors"


# ---------------------------------------------------------------------------
# Platform-adaptive file locking
# ---------------------------------------------------------------------------

def _acquire_lock(lock_path: Path) -> Any:
    """Acquire an advisory file lock. Returns the open file descriptor."""
    fd = open(lock_path, "w")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, IOError) as exc:
        fd.close()
        raise RuntimeError(
            f"Cannot acquire write lock at {lock_path}. "
            "Another writer may be active."
        ) from exc
    return fd


def _release_lock(fd: Any) -> None:
    """Release an advisory file lock."""
    try:
        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        fd.close()


# ---------------------------------------------------------------------------
# Atomic file replace
# ---------------------------------------------------------------------------

def _atomic_replace(src: Path, dst: Path) -> None:
    """Atomically replace dst with src (temp write + rename)."""
    os.replace(str(src), str(dst))


# ---------------------------------------------------------------------------
# DatasetWriter
# ---------------------------------------------------------------------------

class DatasetWriter:
    """Append-only writer for v2 trajectory datasets.

    Thread-safe: acquires an advisory file lock on open. Only one writer
    process may be active at a time. Readers can operate concurrently.

    Usage:
        with DatasetWriter("path/to/dataset_v2") as writer:
            traj_id = writer.add_trajectory(
                tensors={"step_00": t0, "step_04": t1, ..., "final": tf},
                metadata={
                    "prompt": "...",
                    "seed": 12345,
                    ...
                },
            )
    """

    _REQUIRED_METADATA = {
        "prompt", "seed", "cfg", "width", "height",
        "n_steps", "attention_backend", "batch_type",
    }

    def __init__(
        self,
        dataset_dir: str | Path,
        max_blob_bytes: int = 1_000_000_000,
    ) -> None:
        self._root = Path(dataset_dir)
        self._max_blob_bytes = max_blob_bytes

        self._blobs_dir = self._root / "blobs"
        self._index_path = self._root / "index.parquet"
        self._lock_path = self._root / "_write_lock"

        self._lock_fd = None
        self._rows: list[dict] = []          # Accumulated parquet rows
        self._next_traj_id: int = 0
        self._wip_tensors: dict[str, Tensor] = {}  # Key -> Tensor for WIP blob
        self._wip_bytes: int = 0             # Cumulative tensor bytes in WIP
        self._wip_n_traj: int = 0            # Trajectories in WIP blob
        self._sealed_count: int = 0           # Number of sealed blobs written

    def __enter__(self) -> DatasetWriter:
        # Create dirs
        self._root.mkdir(parents=True, exist_ok=True)
        self._blobs_dir.mkdir(exist_ok=True)

        # Acquire lock
        self._lock_fd = _acquire_lock(self._lock_path)

        # Load existing state
        if self._index_path.exists():
            table = pq.read_table(str(self._index_path))
            self._rows = table.to_pylist()
            if self._rows:
                self._next_traj_id = max(r["traj_id"] for r in self._rows) + 1
                # Count existing sealed blobs for sequential naming
                self._sealed_count = sum(
                    1 for f in self._blobs_dir.iterdir()
                    if f.name.startswith("blob_") and f.suffix == ".safetensors"
                )
        else:
            self._rows = []
            self._sealed_count = 0

        return self

    def __exit__(self, *exc) -> None:
        try:
            # Seal the WIP blob if it has data
            if self._wip_tensors:
                self.seal_current_blob()
            # Final index flush
            self._write_index()
        finally:
            if self._lock_fd is not None:
                _release_lock(self._lock_fd)
                self._lock_fd = None

    def add_trajectory(
        self,
        tensors: dict[str, Tensor],
        metadata: dict[str, Any],
    ) -> int:
        """Append one trajectory to the dataset.

        Args:
            tensors: Map of step labels to latent tensors.
                Keys must be "step_XX" or "final".
                Tensors must be (1, C, H, W) or (C, H, W) bfloat16.
            metadata: Trajectory metadata dict. See _REQUIRED_METADATA.

        Returns:
            The assigned traj_id.
        """
        # Validate metadata
        missing = self._REQUIRED_METADATA - set(metadata.keys())
        if missing:
            raise ValueError(f"Missing required metadata keys: {missing}")

        # Validate and prepare tensors
        if not tensors:
            raise ValueError("tensors dict must not be empty")

        # Parse step labels and extract shape info
        step_indices = []
        has_final = False
        prepared: dict[str, Tensor] = {}

        for label, t in tensors.items():
            if label == "final":
                has_final = True
            elif label.startswith("step_"):
                idx = int(label.split("_")[1])
                step_indices.append(idx)
            else:
                raise ValueError(
                    f"Invalid tensor key '{label}': must be 'step_XX' or 'final'"
                )

            # Squeeze batch dim
            if t.dim() == 4 and t.shape[0] == 1:
                t = t.squeeze(0)
            if t.dim() != 3:
                raise ValueError(
                    f"Tensor '{label}' has unexpected shape {t.shape}; "
                    f"expected (C, H, W) or (1, C, H, W)"
                )
            prepared[label] = t.contiguous()

        step_indices.sort()

        # Extract shape from first tensor
        ref_tensor = next(iter(prepared.values()))
        c, h, w = ref_tensor.shape
        dtype_str = str(ref_tensor.dtype).replace("torch.", "")

        # Compute bytes for this trajectory
        traj_bytes = sum(t.nelement() * t.element_size() for t in prepared.values())
        n_tensors = len(prepared)

        # Assign traj_id and key_prefix
        traj_id = self._next_traj_id
        self._next_traj_id += 1
        key_prefix = f"{traj_id:06d}"

        # Check blob rotation: if adding this trajectory would exceed the
        # max blob size, seal the current blob first.
        if self._wip_tensors and (self._wip_bytes + traj_bytes > self._max_blob_bytes):
            self.seal_current_blob()

        # Add tensors to WIP blob dict with prefixed keys
        for label, t in prepared.items():
            blob_key = f"{key_prefix}/{label}"
            self._wip_tensors[blob_key] = t

        self._wip_bytes += traj_bytes
        self._wip_n_traj += 1

        # Build parquet row
        now = datetime.now(timezone.utc)
        row = {
            "traj_id": traj_id,
            "prompt": metadata["prompt"],
            "prompt_idx": metadata.get("prompt_idx", -1),
            "seed": int(metadata["seed"]),
            "cfg": float(metadata["cfg"]),
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "n_steps": int(metadata["n_steps"]),
            "attention_backend": metadata["attention_backend"],
            "batch_type": metadata["batch_type"],
            "denoise": float(metadata["denoise"]) if metadata.get("denoise") is not None else None,
            "image_file": metadata.get("image_file"),
            "is_gold": bool(metadata.get("is_gold", False)),
            "batch_idx": int(metadata.get("batch_idx", 0)),
            "packed": bool(metadata.get("packed", False)),
            "step_indices": step_indices,
            "has_final": has_final,
            "latent_channels": c,
            "latent_height": h,
            "latent_width": w,
            "latent_dtype": dtype_str,
            "blob_file": _WIP_BLOB,
            "key_prefix": key_prefix,
            "n_tensors": n_tensors,
            "bytes_total": traj_bytes,
            "sampling_shift": float(metadata["sampling_shift"]) if metadata.get("sampling_shift") is not None else None,
            "timing_seconds": float(metadata["timing_seconds"]) if metadata.get("timing_seconds") is not None else None,
            "created_at": now,
            "parent_traj_id": int(metadata["parent_traj_id"]) if metadata.get("parent_traj_id") is not None else None,
            "parent_step": metadata.get("parent_step"),
            "parent_denoise": float(metadata["parent_denoise"]) if metadata.get("parent_denoise") is not None else None,
            # Sampling state identity hashes (nullable)
            "model_state_hash": metadata.get("model_state_hash"),
            "base_model_hash": metadata.get("base_model_hash"),
            "adapter_set_hash": metadata.get("adapter_set_hash"),
            "trajectory_hash": metadata.get("trajectory_hash"),
            "active_adapters": metadata.get("active_adapters"),
        }
        self._rows.append(row)
        return traj_id

    def flush(self) -> None:
        """Seal the current in-memory blob and write the parquet index to disk.

        Call this periodically during long generation runs to limit data loss
        on crash to at most the trajectories added since the last flush.
        """
        if self._wip_tensors:
            self.seal_current_blob()
        self._write_index()

    def seal_current_blob(self) -> str | None:
        """Write in-memory tensors as a sealed blob. Returns the filename.

        Tensors are accumulated in memory (no WIP blob on disk) and written
        once here. This avoids the O(N^2) write amplification of rewriting
        a WIP blob on every add_trajectory call, and eliminates the crash
        corruption vector of partial WIP blob writes.
        """
        if not self._wip_tensors:
            return None

        new_name = f"blob_{self._sealed_count:03d}.safetensors"
        new_path = self._blobs_dir / new_name
        temp_path = new_path.with_suffix(".tmp")

        blob_meta = {
            "dataset_version": "2",
            "n_trajectories": str(self._wip_n_traj),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Write to temp file first, then atomic rename
        save_file(self._wip_tensors, str(temp_path), metadata=blob_meta)
        os.replace(str(temp_path), str(new_path))
        self._sealed_count += 1

        # Update blob_file in all pending rows (they were assigned _WIP_BLOB)
        for row in self._rows:
            if row["blob_file"] == _WIP_BLOB:
                row["blob_file"] = new_name

        # Reset in-memory buffer
        self._wip_tensors = {}
        self._wip_bytes = 0
        self._wip_n_traj = 0

        # Write updated index (rows now reference the sealed blob name)
        self._write_index()

        return new_name

    @property
    def n_trajectories(self) -> int:
        return len(self._rows)

    @property
    def next_traj_id(self) -> int:
        return self._next_traj_id

    # _flush_to_disk removed: tensors stay in memory until seal_current_blob().
    # This eliminates O(N^2) write amplification and the WIP blob corruption vector.

    def _write_index(self) -> None:
        """Atomically write the parquet index to disk."""
        if not self._rows:
            return

        table = pa.Table.from_pylist(self._rows, schema=INDEX_SCHEMA)
        temp_path = self._index_path.with_suffix(".parquet.tmp")
        pq.write_table(table, str(temp_path), **_PARQUET_WRITE_KWARGS)
        _atomic_replace(temp_path, self._index_path)


# ---------------------------------------------------------------------------
# Blob handle LRU cache
# ---------------------------------------------------------------------------

class _BlobCache:
    """LRU cache of open safetensors file handles."""

    def __init__(self, blobs_dir: Path, max_open: int = 8):
        self._blobs_dir = blobs_dir
        self._max_open = max_open
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get_tensor(self, blob_file: str, key: str) -> Tensor:
        """Load a tensor from a blob, opening the file if needed."""
        if blob_file not in self._cache:
            if len(self._cache) >= self._max_open:
                # Evict LRU entry
                _, old_handle = self._cache.popitem(last=False)
                # safe_open does not have __exit__ when used outside `with`;
                # just drop the reference (the mmap handle will be GC'd).
                del old_handle
            path = self._blobs_dir / blob_file
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._cache[blob_file] = handle
        else:
            self._cache.move_to_end(blob_file)
        return self._cache[blob_file].get_tensor(key)

    def close(self) -> None:
        """Close all open handles."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# TensorAccessor
# ---------------------------------------------------------------------------

class TensorAccessor:
    """Lazy tensor accessor for one trajectory's latents.

    Does not load any tensor data until explicitly requested.
    Uses safetensors mmap for zero-copy reads where possible.
    """

    def __init__(
        self,
        blob_cache: _BlobCache,
        blob_file: str,
        key_prefix: str,
        step_indices: list[int],
        has_final: bool,
    ):
        self._cache = blob_cache
        self._blob_file = blob_file
        self._key_prefix = key_prefix
        self._step_indices = step_indices
        self._has_final = has_final

    def __getitem__(self, step_label: str) -> Tensor:
        """Load a single tensor by step label.

        Returns tensor with batch dimension restored: (1, C, H, W).
        """
        key = f"{self._key_prefix}/{step_label}"
        t = self._cache.get_tensor(self._blob_file, key)
        return t.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

    def load_all(self) -> dict[str, Tensor]:
        """Load all tensors for this trajectory.

        Returns dict mapping step labels to (1, C, H, W) tensors.
        """
        result = {}
        for step_label in self.available_steps:
            result[step_label] = self[step_label]
        return result

    @property
    def available_steps(self) -> list[str]:
        """List of available step labels (e.g., ['step_00', ..., 'final'])."""
        labels = [f"step_{idx:02d}" for idx in self._step_indices]
        if self._has_final:
            labels.append("final")
        return labels


# ---------------------------------------------------------------------------
# FilteredView
# ---------------------------------------------------------------------------

class FilteredView:
    """A filtered view of the dataset. Lightweight (stores only indices)."""

    def __init__(self, table: pa.Table, reader: DatasetReader):
        self._table = table
        self._reader = reader
        self._traj_ids: list[int] | None = None

    @property
    def traj_ids(self) -> list[int]:
        if self._traj_ids is None:
            self._traj_ids = self._table.column("traj_id").to_pylist()
        return self._traj_ids

    @property
    def count(self) -> int:
        return len(self._table)

    def __iter__(self) -> Iterator[int]:
        return iter(self.traj_ids)

    def __len__(self) -> int:
        return self.count

    def sample(self, n: int, rng: random.Random | None = None) -> list[int]:
        """Sample n traj_ids from the filtered set."""
        ids = self.traj_ids
        if rng is None:
            rng = random.Random()
        return rng.sample(ids, min(n, len(ids)))

    def to_table(self) -> pa.Table:
        """Return the filtered rows as a pyarrow Table."""
        return self._table


# ---------------------------------------------------------------------------
# DatasetReader
# ---------------------------------------------------------------------------

class DatasetReader:
    """Read-only interface to a v2 trajectory dataset.

    Loads the parquet index into memory on construction. Tensor data is
    lazy-loaded from safetensors blobs via mmap on access.
    """

    def __init__(self, dataset_dir: str | Path, max_open_blobs: int = 8) -> None:
        self._root = Path(dataset_dir)
        self._blobs_dir = self._root / "blobs"
        self._index_path = self._root / "index.parquet"
        self._blob_cache = _BlobCache(self._blobs_dir, max_open=max_open_blobs)

        self._table: pa.Table | None = None
        self._row_lookup: dict[int, int] | None = None  # traj_id -> row index

        self.reload()

    def reload(self) -> None:
        """Re-read index.parquet from disk."""
        if not self._index_path.exists():
            self._table = pa.table({}, schema=INDEX_SCHEMA)
            self._row_lookup = {}
            return

        self._table = pq.read_table(str(self._index_path))
        traj_ids = self._table.column("traj_id").to_pylist()
        self._row_lookup = {tid: i for i, tid in enumerate(traj_ids)}

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, traj_id: int) -> bool:
        return traj_id in self._row_lookup

    def __getitem__(self, traj_id: int) -> tuple[dict, TensorAccessor]:
        """Random access by traj_id.

        Returns:
            (metadata_dict, tensor_accessor)
        """
        if traj_id not in self._row_lookup:
            raise KeyError(f"traj_id {traj_id} not in dataset")

        row_idx = self._row_lookup[traj_id]
        row = {
            col: self._table.column(col)[row_idx].as_py()
            for col in self._table.column_names
        }

        accessor = TensorAccessor(
            blob_cache=self._blob_cache,
            blob_file=row["blob_file"],
            key_prefix=row["key_prefix"],
            step_indices=row["step_indices"],
            has_final=row["has_final"],
        )

        return row, accessor

    def filter(self, **kwargs) -> FilteredView:
        """Return a filtered view using equality filters on columns."""
        mask = None
        for col, val in kwargs.items():
            col_mask = pc.equal(self._table.column(col), val)
            if mask is None:
                mask = col_mask
            else:
                mask = pc.and_(mask, col_mask)

        if mask is None:
            filtered = self._table
        else:
            filtered = self._table.filter(mask)

        return FilteredView(filtered, self)

    def filter_expr(self, expr: pc.Expression) -> FilteredView:
        """Return a filtered view using a pyarrow compute expression."""
        filtered = self._table.filter(expr)
        return FilteredView(filtered, self)

    def sample(
        self,
        n: int,
        rng: random.Random | None = None,
        **filter_kwargs,
    ) -> list[int]:
        """Sample n traj_ids uniformly at random, optionally filtered."""
        if filter_kwargs:
            view = self.filter(**filter_kwargs)
            return view.sample(n, rng=rng)

        ids = self._table.column("traj_id").to_pylist()
        if rng is None:
            rng = random.Random()
        return rng.sample(ids, min(n, len(ids)))

    def iter_metadata(
        self, **filter_kwargs
    ) -> Iterator[tuple[int, dict]]:
        """Iterate over (traj_id, metadata_dict) pairs.

        Efficient: reads only parquet data, no tensor I/O.
        """
        if filter_kwargs:
            table = self.filter(**filter_kwargs).to_table()
        else:
            table = self._table

        rows = table.to_pylist()
        for row in rows:
            yield row["traj_id"], row

    def scrimble_split(self) -> tuple[list[int], list[int]]:
        """Return (sdpa_traj_ids, sage_traj_ids) for BTRM head 0 training.

        Only 30-step t2i trajectories.
        """
        mask = pc.and_(
            pc.equal(self._table.column("batch_type"), "t2i"),
            pc.equal(self._table.column("n_steps"), 30),
        )
        filtered = self._table.filter(mask)
        sdpa_ids = []
        sage_ids = []
        for i in range(len(filtered)):
            tid = filtered.column("traj_id")[i].as_py()
            backend = filtered.column("attention_backend")[i].as_py()
            if backend == "sdpa":
                sdpa_ids.append(tid)
            else:
                sage_ids.append(tid)
        return sdpa_ids, sage_ids

    def scrongle_split(self) -> tuple[list[int], list[int]]:
        """Return (full_step_traj_ids, reduced_step_traj_ids) for BTRM head 1.

        Full = 30 steps, reduced = <30 steps. t2i only.
        """
        mask = pc.equal(self._table.column("batch_type"), "t2i")
        filtered = self._table.filter(mask)
        full_ids = []
        reduced_ids = []
        for i in range(len(filtered)):
            tid = filtered.column("traj_id")[i].as_py()
            n_steps = filtered.column("n_steps")[i].as_py()
            if n_steps == 30:
                full_ids.append(tid)
            else:
                reduced_ids.append(tid)
        return full_ids, reduced_ids

    def close(self) -> None:
        """Close all open file handles."""
        self._blob_cache.close()
