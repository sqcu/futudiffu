# Dataset V2 Specification: Parquet Index + Safetensors Blobs

**Status**: Design spec (pre-implementation)
**Date**: 2026-02-14
**Replaces**: v1 filesystem format (one directory per trajectory, .pt files, manifest.json)

---

## 1. Motivation

The v1 format stores each trajectory as a separate directory containing individual
`.pt` files and a `meta.json`. This has several scaling problems:

- **Filesystem overhead**: At 2304 planned trajectories, that is 2304 directories,
  ~18,000 small files. At megasample scale, this becomes a filesystem metadata
  bottleneck (slow `stat()`, slow `opendir()`, NTFS journal pressure on Windows).
- **No efficient filtering**: To find "all t2i trajectories with sdpa attention and
  30 steps," the reader must load and parse every `meta.json` or the full
  `manifest.json` (which grows linearly and is not indexed).
- **No concurrent write safety**: The manifest.json is rewritten in full on every
  generation run. A reader during a write sees a truncated or corrupt file.
- **Pickle in `.pt` files**: `torch.save` uses pickle. Safetensors is the
  zero-deserialization-attack standard.
- **HuggingFace upload friction**: HF datasets expects parquet for metadata and
  safetensors for tensors. The v1 format requires a full conversion pass.

## 2. Design Goals

1. O(1) random access to any trajectory's metadata (parquet row lookup).
2. O(1) random access to any trajectory's tensors (safetensors mmap offset read).
3. Efficient columnar filtering on any metadata field (parquet predicate pushdown).
4. Append-only writes: new trajectories never mutate existing blobs.
5. Concurrent reader safety: readers always see a consistent snapshot.
6. Direct HuggingFace datasets compatibility (parquet + safetensors).
7. No pickle anywhere. No torch.save. No torch.load.

## 3. Directory Layout

```
dataset_v2/
    index.parquet                     # Single parquet file, one row per trajectory
    blobs/
        blob_a1b2c3d4.safetensors    # Multi-trajectory tensor blob (~1 GB target)
        blob_e5f6g7h8.safetensors    # Next blob when first is full
        ...
    _write_lock                       # Advisory lockfile for writer exclusivity
    dataset_card.yaml                 # HuggingFace dataset card metadata
```

### File naming

- **index.parquet**: Always this exact name. One file, not partitioned.
  Partitioned parquet adds complexity for a dataset that fits comfortably in RAM
  as a flat table (2304 rows x ~20 columns = ~200 KB; even 1M rows < 100 MB).
- **blob_{hash8}.safetensors**: The `{hash8}` is the first 8 hex characters of
  the SHA-256 of the blob's content at seal time. This provides deduplication
  detection and corruption checking. During active writes (before sealing), the
  file is named `blob_wip.safetensors`.
- **_write_lock**: Zero-byte file, used as an `fcntl.flock()` / `msvcrt.locking()`
  advisory lock. Only one writer process at a time.

## 4. Parquet Index Schema

One row per trajectory. All columns are required unless marked nullable.

| Column | Arrow Type | Description |
|--------|-----------|-------------|
| `traj_id` | `int64` | Globally unique trajectory ID. Monotonically increasing. Primary key. |
| `prompt` | `utf8` (large_string) | Full prompt text. |
| `prompt_idx` | `int32` | Index into `PROMPT_TEMPLATES` (-1 for i2i or freeform prompts). |
| `seed` | `uint64` | PRNG seed used for noise generation. |
| `cfg` | `float32` | Classifier-free guidance scale. |
| `width` | `int32` | Output image width in pixels. |
| `height` | `int32` | Output image height in pixels. |
| `n_steps` | `int32` | Total diffusion steps in the trajectory. |
| `attention_backend` | `utf8` | `"sdpa"` or `"sage"`. (Renamed from v1 `precision` for clarity.) |
| `batch_type` | `utf8` | `"t2i"` or `"i2i"`. |
| `denoise` | `float32` (nullable) | Denoise strength. NULL for t2i. |
| `image_file` | `utf8` (nullable) | Source image filename for i2i. NULL for t2i. |
| `is_gold` | `bool` | True if this is a gold-standard reference trajectory (sdpa, 30 steps). |
| `batch_idx` | `int32` | Generation schedule batch index (provenance). |
| `packed` | `bool` | True if generated via FlexAttention packed forward. |
| `step_indices` | `list<int32>` | Which diffusion steps have stored latents (e.g., `[0, 4, 9, 14, 19, 24, 29]`). |
| `has_final` | `bool` | Whether `final` latent is stored. Always True for completed trajectories. |
| `latent_channels` | `int32` | Latent channel count (16 for Z-Image). |
| `latent_height` | `int32` | Latent spatial height (pixels / 8, e.g., 104 for 832px). |
| `latent_width` | `int32` | Latent spatial width (pixels / 8, e.g., 160 for 1280px). |
| `latent_dtype` | `utf8` | Tensor dtype string: `"bfloat16"`. |
| `blob_file` | `utf8` | Filename of the safetensors blob (e.g., `"blob_a1b2c3d4.safetensors"`). |
| `key_prefix` | `utf8` | Prefix for this trajectory's keys in the blob (e.g., `"000042"`). |
| `n_tensors` | `int32` | Number of tensors stored for this trajectory (len(step_indices) + has_final). |
| `bytes_total` | `int64` | Total bytes of tensor data for this trajectory. |
| `timing_seconds` | `float32` (nullable) | Wall-clock generation time. NULL if not recorded. |
| `created_at` | `timestamp[us, tz=UTC]` | When this trajectory was generated. |
| `parent_traj_id` | `int64` (nullable) | Parent trajectory ID for i2i2i chains. NULL for non-i2i2i. |
| `parent_step` | `utf8` (nullable) | Parent step label (e.g., `"step_14"`). NULL for non-i2i2i. |
| `parent_denoise` | `float32` (nullable) | Denoise strength used in i2i2i pass. NULL for non-i2i2i. |
| `source_dir` | `utf8` (nullable) | Source directory name (provenance). |
| `run_name` | `utf8` (nullable) | Training run name (provenance). E.g., `"original_v1"`, `"policy_rollout_v1"`, `"2xh100_20260216"`. |
| `source_device` | `utf8` (nullable) | Source GPU device (provenance). E.g., `"local"`, `"gpu0"`, `"gpu1"`. |
| `model_state_hash` | `utf8` (nullable) | SHA-256 hex digest identifying the exact model state (base + adapters). |
| `base_model_hash` | `utf8` (nullable) | SHA-256 hex digest (or placeholder) identifying the base model weights. |
| `adapter_set_hash` | `utf8` (nullable) | SHA-256 hex digest of the active adapter set. `""` = no adapters. NULL = unknown. |
| `trajectory_hash` | `utf8` (nullable) | SHA-256 hex digest of the full trajectory identity (model state + sampling params). |
| `active_adapters` | `utf8` (nullable) | JSON-serialized list of `{"name": str, "strength": float, "param_hash": str}`. `"[]"` = no adapters. NULL = unknown. |

### Sampling state identity hashes

The schema columns `(prompt, seed, cfg, n_steps, width, height, attention_backend)`
identify a sampling configuration but NOT the model state. When the model has
active LoRA adapters, the same sampling configuration produces different outputs.

The hash columns form a hierarchy:

1. **`base_model_hash`**: Identifies the frozen base model weights. Currently a
   placeholder string `"z_image_v1"` (hashing 6 GB of FP8 weights is deferred).
   When a policy adapter is materialized into the base model (base_new =
   base_old + A @ B), the base_model_hash changes.

2. **`adapter_set_hash`**: Identifies the UNORDERED SET of active adapters.
   - Each adapter contributes `(strength, param_hash)` to the hash.
   - Adapters with `strength=0` are excluded (disabled adapters don't affect output).
   - Order-independent: `hash([(1.0, h_a), (0.5, h_b)]) == hash([(0.5, h_b), (1.0, h_a)])`.
   - `""` (empty string) means no active adapters. NULL means unknown.
   - `adapter_param_hash` = SHA-256 of the adapter's parameter tensors (they
     mutate during training, so the hash captures a specific training checkpoint).

3. **`model_state_hash`**: `SHA-256(base_model_hash || adapter_set_hash)`.
   Uniquely identifies the exact model state for generation.

4. **`trajectory_hash`**: `SHA-256(model_state_hash || prompt || seed || cfg || n_steps || width || height)`.
   Uniquely identifies a specific diffusion sampling run.

**Backfill status**: For `original_v1` trajectories, all hashes are computed.
For `policy_rollout_v1` and `2xh100_20260216`, only `base_model_hash` is filled
(the adapter state at generation time was not recorded). See
`scripts_ii/backfill_v2_hashes.py`.

### Column rationale

- **`key_prefix`** is zero-padded 6-digit traj_id as string (e.g., `"000042"`),
  not the traj_id integer. This lets safetensors key lookups be pure string
  operations without int-to-string conversion at read time.
- **`step_indices`** is a list column because different trajectories may store
  different subsets of steps (reduced-step trajectories store fewer checkpoints).
- **`latent_channels/height/width`** are per-row because i2i trajectories can
  have different resolutions (496x544, 512x512, 832x1280, etc.).
- **`blob_file`** is a relative filename (not a full path). The reader
  constructs the full path as `dataset_dir / "blobs" / blob_file`.
- **`bytes_total`** enables the writer to track blob fill level without
  reading back the safetensors file.

### Parquet write settings

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Write settings for optimal read performance
pq.write_table(
    table,
    "index.parquet",
    compression="zstd",           # Best ratio for string-heavy data
    compression_level=3,          # Fast compression, still good ratio
    write_statistics=True,        # Enable min/max stats for predicate pushdown
    use_dictionary=[              # Dictionary-encode low-cardinality columns
        "attention_backend",
        "batch_type",
        "latent_dtype",
        "blob_file",
    ],
    row_group_size=10_000,        # Reasonable for datasets up to ~1M rows
)
```

## 5. Safetensors Blob Format

### Key naming convention

Each tensor in a blob is keyed as `{key_prefix}/{step_label}`:

```
000042/step_00      # Trajectory 42, step index 0
000042/step_04      # Trajectory 42, step index 4
000042/step_09      # Trajectory 42, step index 9
000042/step_14      # ...
000042/step_19
000042/step_24
000042/step_29
000042/final        # Trajectory 42, final latent (post inverse_noise_scaling)
```

- `key_prefix`: 6-digit zero-padded `traj_id` (matches the `key_prefix` column
  in the parquet index).
- `step_label`: Either `step_{idx:02d}` for intermediate steps, or `final` for
  the post-denoising latent.

### Tensor shape

All tensors are stored with the batch dimension squeezed:
- Shape: `(C, H, W)` where C=16, H=latent_height, W=latent_width.
- Dtype: `bfloat16` (stored as the safetensors `BF16` type).

The batch dimension `(1, C, H, W)` from the inference server is squeezed to
`(C, H, W)` at write time. The reader unsqueezes on load. This matches the
existing `pack_trajectories.py` convention and saves no bytes but avoids
ambiguity about batch semantics.

### Blob size management

- **Target size**: ~1 GB per blob (configurable via `max_blob_bytes` parameter).
- **Rotation**: When appending a trajectory would cause the WIP blob to exceed
  `max_blob_bytes`, the WIP blob is sealed (renamed with its content hash) and a
  new WIP blob is started.
- **Size estimation**: The writer tracks cumulative bytes via the parquet
  `bytes_total` column sum for the current WIP blob. A single 1280x832 t2i
  trajectory with 8 checkpoints = 8 * 16 * 104 * 160 * 2 = 4,259,840 bytes
  (~4.07 MB). At ~1 GB per blob, that is ~250 trajectories per blob.
- **Immutability**: Once a blob is sealed (renamed from `blob_wip.safetensors`
  to `blob_{hash8}.safetensors`), it is never modified. Write-once, read-many.

### Why not one safetensors per trajectory?

At 2304 trajectories, one-per-trajectory means 2304 files at ~4 MB each. This
is manageable but does not scale to megasamples (1M files is a filesystem
nightmare). Multi-trajectory blobs:
- Reduce file count by ~250x.
- Enable sequential I/O patterns (one `open()` + multiple `mmap` reads).
- Align with HuggingFace dataset sharding conventions.

### Blob metadata header

Safetensors supports a JSON metadata header. Each blob stores:

```json
{
    "__metadata__": {
        "dataset_version": "2",
        "blob_id": "a1b2c3d4",
        "n_trajectories": "247",
        "created_at": "2026-02-14T15:30:00Z",
        "sealed": "true"
    }
}
```

The WIP blob has `"sealed": "false"`. The sealed blob has `"sealed": "true"`.
These are all strings because safetensors metadata is string-valued only.

## 6. Writer API

### Module: `futudiffu.dataset_v2.writer`

```python
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
                    "prompt_idx": 5,
                    "seed": 12345,
                    "cfg": 4.0,
                    "width": 1280,
                    "height": 832,
                    "n_steps": 30,
                    "attention_backend": "sdpa",
                    "batch_type": "t2i",
                    # ... other fields
                },
            )
            print(f"Wrote trajectory {traj_id}")
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        max_blob_bytes: int = 1_000_000_000,  # ~1 GB
    ) -> None:
        """Open or create a v2 dataset directory.

        Args:
            dataset_dir: Path to the dataset root directory.
            max_blob_bytes: Maximum blob size before rotation.
        """
        ...

    def __enter__(self) -> "DatasetWriter": ...
    def __exit__(self, *exc) -> None: ...

    def add_trajectory(
        self,
        tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
    ) -> int:
        """Append one trajectory to the dataset.

        Args:
            tensors: Map of step labels to latent tensors.
                Keys must be "step_XX" or "final".
                Tensors must be (1, C, H, W) or (C, H, W) bfloat16.
            metadata: Trajectory metadata dict. Required keys:
                prompt, seed, cfg, width, height, n_steps,
                attention_backend, batch_type.
                Optional keys: prompt_idx, denoise, image_file,
                is_gold, batch_idx, packed, timing_seconds.

        Returns:
            The assigned traj_id (int).

        Raises:
            ValueError: If required metadata keys are missing.
            ValueError: If tensor shapes are inconsistent.
            RuntimeError: If write lock cannot be acquired.
        """
        ...

    def flush(self) -> None:
        """Force-write the current parquet index to disk.

        Called automatically on context manager exit and after blob rotation.
        Safe to call at any time for checkpoint consistency.
        """
        ...

    def seal_current_blob(self) -> str | None:
        """Seal the WIP blob (rename with content hash). Returns new filename.

        Called automatically when blob size exceeds max_blob_bytes.
        Can be called manually to force blob finalization (e.g., at end of
        a generation run).

        Returns None if no WIP blob exists.
        """
        ...

    @property
    def n_trajectories(self) -> int:
        """Total trajectory count in the dataset."""
        ...

    @property
    def next_traj_id(self) -> int:
        """Next traj_id that will be assigned."""
        ...
```

### Write protocol (internal)

1. **Lock**: Acquire advisory lock on `_write_lock`. Fail fast if already locked.
2. **Load state**: Read `index.parquet` if it exists to determine `next_traj_id`
   and current WIP blob filename. If no parquet exists, start fresh.
3. **Per trajectory**:
   a. Assign `traj_id = next_traj_id; next_traj_id += 1`.
   b. Compute `key_prefix = f"{traj_id:06d}"`.
   c. Squeeze tensors from (1, C, H, W) to (C, H, W) if needed.
   d. Compute `bytes_total` for this trajectory.
   e. Check if adding this trajectory would exceed `max_blob_bytes` for the
      current WIP blob. If so, seal the current blob and start a new WIP.
   f. **Append tensors to WIP blob**: Safetensors does not support incremental
      append. The writer must accumulate tensors in memory for the current WIP
      blob and rewrite the entire WIP blob on each `add_trajectory` call.
      This is acceptable because:
      - The WIP blob is at most `max_blob_bytes` (~1 GB).
      - Rewrites happen to a single file.
      - Sealed blobs are never rewritten.

      **Optimization**: For large datasets where rewriting the entire WIP blob
      on every trajectory is too slow, the writer can buffer N trajectories in
      memory (e.g., 16) and flush them to the WIP blob in batches. The
      `flush()` method triggers this write.
   g. Append a row to the in-memory parquet table.
   h. Write `index.parquet` to a temp file, then atomic rename. This ensures
      readers always see a complete parquet file.
4. **Seal on close**: When the context manager exits, seal the WIP blob and
   write the final parquet index.

### Atomic parquet updates

```python
# Write to temp, then atomic rename (POSIX) or replace (Windows)
temp_path = index_path.with_suffix(".parquet.tmp")
pq.write_table(table, str(temp_path), ...)
if sys.platform == "win32":
    # Windows: os.replace is atomic on NTFS for same-volume renames
    os.replace(str(temp_path), str(index_path))
else:
    os.rename(str(temp_path), str(index_path))
```

### WIP blob strategy

Because safetensors files cannot be incrementally appended to, the writer uses
one of two strategies (configurable):

**Strategy A: Rewrite-on-each-add** (simple, default for small runs)
- Keep all tensors for the current WIP blob in memory.
- On each `add_trajectory`, add tensors to the in-memory dict and rewrite
  `blob_wip.safetensors` in full.
- On seal, compute hash and rename.

**Strategy B: Buffered batch write** (for high-throughput generation)
- Buffer N trajectories in memory (default: 32).
- On buffer full or explicit `flush()`, write all buffered tensors to the WIP
  blob (rewriting it).
- Between flushes, tensors are only in memory. If the process crashes, unflushed
  trajectories are lost (but the parquet index is only updated on flush, so
  consistency is maintained).

The choice is exposed as a constructor parameter:
```python
DatasetWriter(dataset_dir, flush_every=1)   # Strategy A
DatasetWriter(dataset_dir, flush_every=32)  # Strategy B
```

## 7. Reader API

### Module: `futudiffu.dataset_v2.reader`

```python
class DatasetReader:
    """Read-only interface to a v2 trajectory dataset.

    Loads the parquet index into memory on construction. Tensor data is
    lazy-loaded from safetensors blobs via mmap on access.

    Usage:
        ds = DatasetReader("path/to/dataset_v2")

        # Random access
        meta, tensors = ds[42]  # Returns metadata dict + lazy tensor accessor

        # Filtering
        subset = ds.filter(batch_type="t2i", attention_backend="sdpa", n_steps=30)
        for traj_id in subset.traj_ids:
            meta, tensors = ds[traj_id]

        # Sampling
        batch = ds.sample(n=32, batch_type="t2i")

        # Iteration
        for traj_id, meta in ds.iter_metadata():
            ...
    """

    def __init__(self, dataset_dir: str | Path) -> None:
        """Open a v2 dataset for reading.

        Args:
            dataset_dir: Path to the dataset root directory.

        Loads index.parquet into a pyarrow Table (in memory).
        Opens safetensors blob file handles lazily on first access.
        """
        ...

    def __len__(self) -> int:
        """Number of trajectories in the dataset."""
        ...

    def __getitem__(self, traj_id: int) -> tuple[dict, "TensorAccessor"]:
        """Random access by traj_id.

        Returns:
            (metadata_dict, tensor_accessor) where:
            - metadata_dict has all parquet columns as Python types.
            - tensor_accessor is a lazy-loading object (see below).

        Raises:
            KeyError: If traj_id is not in the dataset.
        """
        ...

    def __contains__(self, traj_id: int) -> bool:
        """Check if a traj_id exists in the dataset."""
        ...

    def filter(self, **kwargs) -> "FilteredView":
        """Return a filtered view of the dataset.

        Keyword arguments are column=value equality filters.
        Supports: batch_type, attention_backend, n_steps, width, height,
                  is_gold, prompt_idx, seed, blob_file.

        For range filters, use filter_expr() with pyarrow compute expressions.

        Returns:
            A FilteredView object with .traj_ids, .count, iteration, etc.

        Example:
            sdpa_30 = ds.filter(batch_type="t2i", attention_backend="sdpa", n_steps=30)
            sage_30 = ds.filter(batch_type="t2i", attention_backend="sage", n_steps=30)
        """
        ...

    def filter_expr(self, expr: pa.compute.Expression) -> "FilteredView":
        """Return a filtered view using a pyarrow compute expression.

        Example:
            import pyarrow.compute as pc
            reduced = ds.filter_expr(
                (pc.field("batch_type") == "t2i") &
                (pc.field("n_steps") < 30) &
                (pc.field("n_steps") >= 8)
            )
        """
        ...

    def sample(
        self,
        n: int,
        rng: random.Random | None = None,
        **filter_kwargs,
    ) -> list[int]:
        """Sample n traj_ids uniformly at random, optionally filtered.

        Args:
            n: Number of trajectories to sample.
            rng: Optional RNG for deterministic sampling.
            **filter_kwargs: Passed to filter() before sampling.

        Returns:
            List of traj_ids.
        """
        ...

    def iter_metadata(
        self, **filter_kwargs
    ) -> Iterator[tuple[int, dict]]:
        """Iterate over (traj_id, metadata_dict) pairs.

        Efficient: reads only parquet data, no tensor I/O.
        """
        ...

    def scrimble_split(self) -> tuple[list[int], list[int]]:
        """Return (sdpa_traj_ids, sage_traj_ids) for BTRM head 0 training.

        Only 30-step t2i trajectories. Equivalent to TrajectoryPool.scrimble_split().
        """
        ...

    def scrongle_split(self) -> tuple[list[int], list[int]]:
        """Return (full_step_traj_ids, reduced_step_traj_ids) for BTRM head 1.

        Full = 30 steps, reduced = <30 steps. t2i only.
        Equivalent to TrajectoryPool.scrongle_split().
        """
        ...

    def close(self) -> None:
        """Close all open file handles."""
        ...

    def reload(self) -> None:
        """Re-read index.parquet from disk. Use to pick up new trajectories
        written by a concurrent writer."""
        ...


class TensorAccessor:
    """Lazy tensor accessor for one trajectory's latents.

    Does not load any tensor data until explicitly requested.
    Uses safetensors mmap for zero-copy reads where possible.

    Usage:
        meta, tensors = ds[42]
        step_00 = tensors["step_00"]          # Returns (1, 16, H, W) bf16 Tensor
        final = tensors["final"]              # Returns (1, 16, H, W) bf16 Tensor
        all_steps = tensors.load_all()        # Returns dict[str, Tensor]
        step_names = tensors.available_steps  # ["step_00", "step_04", ..., "final"]
    """

    def __getitem__(self, step_label: str) -> torch.Tensor:
        """Load a single tensor by step label.

        Returns tensor with batch dimension restored: (1, C, H, W).
        """
        ...

    def load_all(self) -> dict[str, torch.Tensor]:
        """Load all tensors for this trajectory.

        Returns dict mapping step labels to (1, C, H, W) tensors.
        """
        ...

    @property
    def available_steps(self) -> list[str]:
        """List of available step labels (e.g., ["step_00", ..., "final"])."""
        ...


class FilteredView:
    """A filtered view of the dataset. Lightweight (stores only indices)."""

    @property
    def traj_ids(self) -> list[int]:
        """All traj_ids matching the filter."""
        ...

    @property
    def count(self) -> int:
        """Number of matching trajectories."""
        ...

    def __iter__(self) -> Iterator[int]:
        """Iterate over matching traj_ids."""
        ...

    def sample(self, n: int, rng: random.Random | None = None) -> list[int]:
        """Sample n traj_ids from the filtered set."""
        ...

    def to_table(self) -> pa.Table:
        """Return the filtered rows as a pyarrow Table."""
        ...
```

### Compatibility with TrajectoryPool

The v2 `DatasetReader` replaces `TrajectoryPool`. The key interface changes:

| TrajectoryPool (v1) | DatasetReader (v2) | Notes |
|---------------------|-------------------|-------|
| `__init__(dataset_dir)` | `__init__(dataset_dir)` | Same constructor signature. |
| `.examples` -> `list[TrajectoryExample]` | `.iter_metadata()` -> `Iterator[tuple[int, dict]]` | v2 returns raw dicts, not dataclass instances. |
| `.load_checkpoint(example)` -> `Tensor` | `ds[traj_id][1]["step_04"]` -> `Tensor` | v2 uses traj_id indexing + lazy accessor. |
| `.scrimble_split()` | `.scrimble_split()` | Same signature and semantics. |
| `.scrongle_split()` | `.scrongle_split()` | Same signature and semantics. |

The v2 reader does NOT enumerate per-step examples like v1. In v1, each
(trajectory, step) pair is a separate `TrajectoryExample`. In v2, the unit is
the trajectory. Per-step iteration is the caller's responsibility:

```python
# v1 (TrajectoryPool):
for ex in pool.examples:
    latent = pool.load_checkpoint(ex)
    sigma = ex.sigma

# v2 (DatasetReader):
for traj_id, meta in ds.iter_metadata(batch_type="t2i"):
    _, tensors = ds[traj_id]
    for step_idx in meta["step_indices"]:
        step_label = f"step_{step_idx:02d}"
        latent = tensors[step_label]
        sigma = sigmas[step_idx]
```

The `train.py` BTRM training loop must be updated to iterate trajectories rather
than pre-flattened examples. This is a net improvement because it gives the
trainer control over which steps to sample per trajectory per epoch.

## 8. Safetensors Blob I/O Details

### Writing tensors to a blob

```python
from safetensors.torch import save_file

def write_blob(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a safetensors blob.

    All tensor keys follow the {key_prefix}/{step_label} convention.
    All tensors must be contiguous bfloat16.
    """
    # Ensure all tensors are contiguous (safetensors requirement)
    clean = {}
    for k, v in tensors.items():
        if v.dim() == 4 and v.shape[0] == 1:
            v = v.squeeze(0)  # (1, C, H, W) -> (C, H, W)
        clean[k] = v.contiguous()
    save_file(clean, str(path), metadata=metadata)
```

### Reading tensors from a blob

```python
from safetensors import safe_open

def open_blob(path: str | Path) -> safe_open:
    """Open a safetensors blob for lazy tensor reads.

    Returns a context manager / file handle. Use .get_tensor(key) to load
    individual tensors. The file is memory-mapped; only accessed pages are
    read from disk.
    """
    return safe_open(str(path), framework="pt", device="cpu")

# Usage:
with open_blob("blobs/blob_a1b2c3d4.safetensors") as f:
    tensor = f.get_tensor("000042/step_04")  # (16, 104, 160) bf16
    tensor = tensor.unsqueeze(0)             # (1, 16, 104, 160) bf16
```

### File handle caching

The reader maintains an LRU cache of open `safe_open` handles (one per blob
file). The default cache size is 8. Since a ~1 GB blob contains ~250
trajectories, random access across the full dataset with good locality needs
at most `ceil(n_trajectories / 250)` handles open simultaneously.

```python
class _BlobCache:
    """LRU cache of open safetensors file handles."""

    def __init__(self, blobs_dir: Path, max_open: int = 8):
        self._blobs_dir = blobs_dir
        self._max_open = max_open
        self._cache: OrderedDict[str, safe_open] = OrderedDict()

    def get_tensor(self, blob_file: str, key: str) -> torch.Tensor:
        """Load a tensor from a blob, opening the file if needed."""
        if blob_file not in self._cache:
            if len(self._cache) >= self._max_open:
                # Evict LRU
                _, old_handle = self._cache.popitem(last=False)
                old_handle.__exit__(None, None, None)
            path = self._blobs_dir / blob_file
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._cache[blob_file] = handle
        else:
            # Move to end (most recently used)
            self._cache.move_to_end(blob_file)
        return self._cache[blob_file].get_tensor(key)
```

## 9. Concurrency Model

### Writer exclusivity

Only one writer process at a time. Enforced by advisory file lock on
`_write_lock`. The lock is acquired in `__enter__` and released in `__exit__`.

```python
import fcntl  # POSIX
# or
import msvcrt  # Windows

# Platform-adaptive locking:
if sys.platform == "win32":
    lock_fd = open(lock_path, "w")
    msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
else:
    lock_fd = open(lock_path, "w")
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
```

### Reader consistency during writes

Readers see a consistent snapshot because:

1. **Parquet index**: Written atomically via temp file + rename. A reader that
   opens the file before the rename sees the old version. A reader that opens
   after sees the new version. No partial reads.
2. **Sealed blobs**: Immutable. Once a blob is sealed, its content never changes.
3. **WIP blob**: The reader's `safe_open` call may see a partially-written WIP
   blob. However, the reader only accesses trajectories listed in the parquet
   index. Since the parquet index is updated atomically AFTER the WIP blob is
   written, the reader will only request tensors that are fully written.
4. **`reload()`**: Readers can call `reload()` to re-read the parquet index and
   pick up newly committed trajectories without restarting.

### Upload daemon compatibility

The `upload_to_hf.py` watch daemon runs concurrently with the writer. It reads
the parquet index to discover new trajectories and uploads sealed blobs. Since
sealed blobs are immutable, the daemon can safely upload them while the writer
continues appending to a new WIP blob. The daemon should:

1. Read `index.parquet`.
2. Identify sealed blobs (all blobs except `blob_wip.safetensors`).
3. Upload sealed blobs that have not yet been uploaded (track in a local state
   file or by checking the HF repo).
4. Upload `index.parquet` last (so the HF repo always has tensor data before
   the index references it).

## 10. Migration from V1

### Script: `scripts/migrate_v1_to_v2.py`

Converts the v1 filesystem format to v2 parquet+safetensors.

```python
def migrate_v1_to_v2(
    v1_dir: str | Path,
    v2_dir: str | Path,
    max_blob_bytes: int = 1_000_000_000,
    verify: bool = True,
) -> None:
    """Migrate a v1 dataset to v2 format.

    Args:
        v1_dir: Path to the v1 dataset root (contains manifest.json, latents/).
        v2_dir: Path for the new v2 dataset (will be created).
        max_blob_bytes: Blob size limit.
        verify: If True, round-trip verify every tensor.

    The migration:
    1. Reads v1 manifest.json to enumerate all trajectories.
    2. For each trajectory:
       a. Loads meta.json for metadata.
       b. Loads all .pt files via torch.load (weights_only=True).
       c. Writes tensors + metadata via DatasetWriter.add_trajectory().
    3. If verify=True, reads back every tensor from v2 and compares
       against the v1 original (bitwise equality check).
    4. Prints summary: n_trajectories migrated, total bytes, n_blobs created.
    """
    ...
```

### Field mapping (v1 -> v2)

| V1 field (meta.json) | V2 column (parquet) | Transform |
|----------------------|--------------------|-----------|
| `type` | `batch_type` | Rename only. |
| `seed` | `seed` | Direct copy. Cast to uint64. |
| `prompt` | `prompt` | Direct copy. |
| `prompt_idx` | `prompt_idx` | Direct copy. Default -1 if missing. |
| `n_steps` | `n_steps` | Direct copy. |
| `precision` | `attention_backend` | Rename only. |
| `batch_idx` | `batch_idx` | Direct copy. |
| `denoise` | `denoise` | Direct copy. NULL if missing (t2i). |
| `image_file` | `image_file` | Direct copy. NULL if missing (t2i). |
| `output_width` / default 1280 | `width` | Use `output_width` if present, else 1280. |
| `output_height` / default 832 | `height` | Use `output_height` if present, else 832. |
| (not in v1) | `cfg` | Default 4.0 (from `GENERATION_DEFAULTS`). |
| (not in v1) | `is_gold` | Infer: `True` if `attention_backend=="sdpa"` and `n_steps==30`. |
| (not in v1) | `packed` | Use `meta.get("packed", False)`. |
| (not in v1) | `timing_seconds` | NULL (v1 does not record per-trajectory timing). |
| (not in v1) | `created_at` | Use file mtime of `meta.json`. |
| (implicit) | `step_indices` | Derived from which `step_XX.pt` files exist. |
| (implicit) | `has_final` | `True` if `final.pt` exists. |
| (implicit) | `latent_channels` | Read from first tensor shape. |
| (implicit) | `latent_height` | Read from first tensor shape. |
| (implicit) | `latent_width` | Read from first tensor shape. |
| (implicit) | `latent_dtype` | Read from first tensor dtype. Always `"bfloat16"`. |
| (computed) | `key_prefix` | `f"{traj_id:06d}"` |
| (computed) | `blob_file` | Assigned by writer during migration. |
| (computed) | `n_tensors` | Count of step files + final. |
| (computed) | `bytes_total` | Sum of tensor bytes. |

### Migration CLI

```
.venv/Scripts/python.exe scripts/migrate_v1_to_v2.py \
    --v1-dir F:\dox\repos\ai\futudiffu\btrm_dataset \
    --v2-dir F:\dox\repos\ai\futudiffu\btrm_dataset_v2 \
    --verify
```

## 11. HuggingFace Compatibility

### Direct upload structure

The v2 dataset directory is directly uploadable to HuggingFace:

```
repo/
    index.parquet        # HF datasets auto-discovers parquet files
    blobs/
        blob_*.safetensors  # HF natively serves safetensors via API
    dataset_card.yaml    # Rendered as the dataset README on HF
```

### HuggingFace datasets integration

The parquet file is natively loadable by `datasets.load_dataset`:

```python
from datasets import load_dataset

# Load just the index (metadata) -- no tensor download
ds = load_dataset("parquet", data_files="index.parquet")

# Filter efficiently
t2i_sdpa = ds.filter(lambda x: x["batch_type"] == "t2i" and x["attention_backend"] == "sdpa")
```

Tensor access requires downloading the blob files and using `safe_open`. This is
intentional: HF datasets is for metadata browsing and filtering, while tensor
access is via our `DatasetReader` after local download.

### dataset_card.yaml

```yaml
dataset_info:
  description: >
    BTRM trajectory dataset for Z-Image diffusion model training.
    Contains sparse-step latent trajectories across prompts, seeds,
    step counts, and attention backends.
  features:
    traj_id: int64
    prompt: string
    batch_type: string
    attention_backend: string
    n_steps: int32
    seed: uint64
  splits:
    - name: train
      num_examples: 2304
  license: cc-by-nc-4.0
  tags:
    - diffusion
    - trajectories
    - btrm
    - safetensors
```

## 12. Dependencies

### Required (must be in pyproject.toml)

```toml
[project]
dependencies = [
    # ... existing deps ...
    "pyarrow>=14.0",      # Parquet read/write + compute expressions
]
```

**Why pyarrow over polars**: pyarrow is a hard dependency of `datasets` (HF) and
provides native parquet read/write. Adding polars would be a second large
dependency for no gain -- the dataset index is small enough that pyarrow's
in-memory Table is sufficient.

**safetensors**: Already in dependencies.

### No new dependencies beyond pyarrow

- `safetensors`: Already present.
- `torch`: Already present.
- No `polars`, `pandas`, `duckdb`, or other data processing libraries.

## 13. Size Estimates

### Current dataset (50 trajectories)

| Item | Count | Size |
|------|-------|------|
| Trajectories | 50 | |
| Tensors | ~400 | |
| Tensor bytes | ~200 MB | 50 * 8 * 532,480 |
| Blobs (1 GB target) | 1 | All fit in one blob |
| Parquet index | ~10 KB | 50 rows |

### Target dataset (2304 trajectories)

| Item | Count | Size |
|------|-------|------|
| Trajectories | 2,304 | |
| Tensors | ~18,000 | |
| Tensor bytes | ~9.4 GB | 2304 * 8 * 532,480 (assuming uniform 1280x832) |
| Blobs (1 GB target) | ~10 | |
| Parquet index | ~500 KB | 2304 rows |

### Megasample scale (1M trajectories)

| Item | Count | Size |
|------|-------|------|
| Trajectories | 1,000,000 | |
| Tensors | ~8,000,000 | |
| Tensor bytes | ~4 TB | |
| Blobs (1 GB target) | ~4,000 | |
| Parquet index | ~200 MB | 1M rows (fits in RAM) |
| File count | ~4,002 | vs. 9M files in v1 |

## 14. Open Questions

1. **Blob size tuning**: The 1 GB target is a starting point. For HuggingFace
   upload, 5 GB shards are common. For local I/O, 1 GB keeps the WIP rewrite
   cost low. Should this be configurable at the dataset level (stored in
   parquet metadata) or only at the writer level?

2. **Mixed-resolution blobs**: i2i trajectories have varying resolutions
   (496x544, 512x512, 832x1280). All can coexist in the same blob since
   safetensors supports heterogeneous tensor shapes. The parquet index records
   per-trajectory latent dimensions. No action needed, but worth noting that
   blob packing efficiency varies with resolution mix.

3. **Compression**: Safetensors does not support internal compression. BF16
   latent tensors have high entropy and compress poorly with generic algorithms
   (~5-10% reduction). Not worth the CPU cost for training I/O. For HF upload,
   the transfer uses gzip/zstd at the HTTP layer.

4. **Parquet append strategy**: Currently the spec calls for full parquet
   rewrite on each flush. At megasample scale (200 MB parquet), this becomes
   non-trivial. Options:
   - Append-only via multiple parquet files (one per generation run), with a
     `_manifest.json` listing all parquet files. Readers concatenate on load.
   - Single file rewrite is fine up to ~100K trajectories (~10 MB parquet),
     which covers the near-term plan.

   **Decision**: Start with single-file rewrite. Add multi-file append when
   the parquet file exceeds 50 MB (~500K trajectories).

5. **Tensor deduplication**: If the same seed+prompt+steps+attention is
   generated twice (e.g., cross-session reproducibility testing), should the
   writer detect and skip duplicates? For now, no -- duplicates are the caller's
   responsibility. The parquet index can be queried for existing (seed, prompt,
   n_steps, attention_backend) tuples before generation.
