"""Dataset catalog: identity, integrity, and inspection for V2 datasets.

Solves the contamination problem: datasets are plain directories with no
identity, no integrity checking, and no protection against accidental
mutation. A new generation run can append 60 trajectories to an existing
60-trajectory dataset, producing a corrupt 120-entry dataset with
mismatched metadata and no way to detect the damage.

This module provides:
  1. Dataset identity: each dataset gets a unique ID at registration
     based on name, timestamp, and content hash of the parquet index.
  2. Catalog registry: a JSON file tracking all known datasets with
     pre-computed index summaries (unique counts, value ranges,
     distributions) computed at registration time, not at query time.
  3. Integrity verification: hash-based mutation detection on every load.
  4. Named splits: materialized traj_id lists for train/val/test subsets.
  5. Safe loading: CatalogedDataset wraps DatasetReader with integrity
     checks baked into the load path.

The catalog.json is human-readable, git-friendly, and a derived cache
of the parquet index. If it is deleted, datasets can be re-registered
from their on-disk parquet files.

No database. No server. No complex dependencies. JSON + parquet + hashlib.

Import constraints:
  - pathlib, json, hashlib -- stdlib
  - pyarrow -- for parquet I/O (already a project dependency)
  - No torch imports (this is a metadata-only module)
  - No imports from src/futudiffu/ (frozen)
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DatasetNotFoundError(KeyError):
    """Raised when a dataset ID is not in the catalog."""
    pass


class DatasetIntegrityError(ValueError):
    """Raised when a dataset's index hash does not match the catalog."""
    pass


class DatasetAlreadyRegisteredError(ValueError):
    """Raised when re-registering a path whose hash has changed."""
    pass


# ---------------------------------------------------------------------------
# Index hashing
# ---------------------------------------------------------------------------

def _compute_index_hash(index_path: Path) -> str:
    """Compute SHA-256 hash of the raw parquet file bytes.

    Fast, deterministic, catches any mutation including appended rows,
    modified metadata, or rewritten columns.
    """
    h = hashlib.sha256()
    with open(index_path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1 MB chunks
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _short_hash(full_hash: str, length: int = 4) -> str:
    """First N hex chars of a hash."""
    return full_hash[:length]


# ---------------------------------------------------------------------------
# Index summary computation
# ---------------------------------------------------------------------------

# Columns that are always high-cardinality (skip distribution)
_SKIP_DISTRIBUTION_COLS = {
    "prompt", "key_prefix", "blob_file", "created_at",
    "model_state_hash", "base_model_hash", "adapter_set_hash",
    "trajectory_hash", "active_adapters", "parent_step",
}

# Columns that are list-typed (skip range/distribution)
_LIST_COLS = {"step_indices"}

# Threshold for "low cardinality" -- full histogram if unique <= this
_LOW_CARDINALITY_THRESHOLD = 50


def _compute_index_summary(index_path: Path) -> dict:
    """Compute pre-baked summary statistics for all columns in the index.

    Returns a dict with:
      columns: list of column names
      n_rows: total row count
      unique_counts: {col: n_unique}
      value_ranges: {col: [min, max]}  (numeric columns only)
      value_distributions: {col: {val: count}}  (low-cardinality only)
    """
    import pyarrow.parquet as pq

    table = pq.read_table(str(index_path))
    n_rows = len(table)
    columns = table.column_names

    unique_counts: dict[str, int] = {}
    value_ranges: dict[str, list] = {}
    value_distributions: dict[str, dict[str, int]] = {}

    for col_name in columns:
        if col_name in _LIST_COLS:
            continue

        col = table.column(col_name)
        py_values = col.to_pylist()

        # Filter out None values for analysis
        non_null = [v for v in py_values if v is not None]
        if not non_null:
            unique_counts[col_name] = 0
            continue

        unique_vals = set()
        for v in non_null:
            # Make hashable
            if isinstance(v, (list, dict)):
                unique_vals.add(str(v))
            else:
                unique_vals.add(v)

        n_unique = len(unique_vals)
        unique_counts[col_name] = n_unique

        if col_name in _SKIP_DISTRIBUTION_COLS:
            continue

        # Numeric range
        first_val = non_null[0]
        is_numeric = isinstance(first_val, (int, float))
        is_bool = isinstance(first_val, bool)

        if is_numeric and not is_bool:
            try:
                numeric_vals = [float(v) for v in non_null]
                value_ranges[col_name] = [min(numeric_vals), max(numeric_vals)]
            except (TypeError, ValueError):
                pass

        # Distribution for low-cardinality columns
        if n_unique <= _LOW_CARDINALITY_THRESHOLD:
            counter: Counter = Counter()
            for v in non_null:
                if isinstance(v, (list, dict)):
                    counter[str(v)] += 1
                else:
                    counter[str(v)] += 1
            # Sort by value for deterministic output
            value_distributions[col_name] = dict(
                sorted(counter.items(), key=lambda kv: kv[0])
            )

    return {
        "columns": columns,
        "n_rows": n_rows,
        "unique_counts": unique_counts,
        "value_ranges": value_ranges,
        "value_distributions": value_distributions,
    }


# ---------------------------------------------------------------------------
# Blob stats
# ---------------------------------------------------------------------------

def _compute_blob_stats(dataset_dir: Path) -> tuple[int, float]:
    """Count blobs and compute total dataset size in MB."""
    blobs_dir = dataset_dir / "blobs"
    blob_count = 0
    total_bytes = 0

    if blobs_dir.exists():
        for f in blobs_dir.iterdir():
            if f.name.startswith("blob_") and f.suffix == ".safetensors":
                blob_count += 1
                total_bytes += f.stat().st_size

    # Add index size
    index_path = dataset_dir / "index.parquet"
    if index_path.exists():
        total_bytes += index_path.stat().st_size

    # Add sidecar files
    for sidecar in ["step_sigmas.json", "generation_report.json"]:
        sp = dataset_dir / sidecar
        if sp.exists():
            total_bytes += sp.stat().st_size

    total_mb = total_bytes / (1024 * 1024)
    return blob_count, round(total_mb, 1)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

class DatasetCatalog:
    """Registry of V2 datasets with integrity tracking and split support.

    Backed by a JSON file. Thread-safe for reads; single-writer for mutations.
    """

    def __init__(self, catalog_path: str | Path = "catalog.json"):
        self._path = Path(catalog_path)
        self._data: dict = {"datasets": {}}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        """Read catalog from disk."""
        text = self._path.read_text(encoding="utf-8")
        self._data = json.loads(text)
        if "datasets" not in self._data:
            self._data["datasets"] = {}

    def _save(self) -> None:
        """Write catalog to disk atomically."""
        text = json.dumps(self._data, indent=2, default=str, ensure_ascii=False)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(str(tmp), str(self._path))

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register(
        self,
        path: str | Path,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Register a V2 dataset in the catalog.

        Reads the parquet index, computes the content hash, builds the
        index summary, and writes the entry. Returns the dataset_id.

        If the dataset is already registered (same path and hash), returns
        the existing ID without modification.

        If the path is registered but the hash has changed (dataset was
        mutated externally), raises DatasetAlreadyRegisteredError.

        Args:
            path: Path to the dataset directory (must contain index.parquet).
            name: Human-readable name prefix. Defaults to directory name.
            tags: Optional list of string tags for filtering.

        Returns:
            The dataset_id string (e.g. "multi_res_v2_20260219_a3f7").
        """
        dataset_dir = Path(path).resolve()
        index_path = dataset_dir / "index.parquet"

        if not index_path.exists():
            raise FileNotFoundError(
                f"No index.parquet found at {index_path}. "
                f"Is this a V2 dataset directory?"
            )

        # Compute hash
        full_hash = _compute_index_hash(index_path)
        short = _short_hash(full_hash)

        # Check if already registered
        rel_path = str(dataset_dir)
        for existing_id, entry in self._data["datasets"].items():
            existing_resolved = str(Path(entry["path"]).resolve())
            if existing_resolved == rel_path or entry["path"] == str(path):
                if entry["index_hash"] == full_hash:
                    return existing_id
                else:
                    raise DatasetAlreadyRegisteredError(
                        f"Dataset at {path} is already registered as "
                        f"'{existing_id}' but its index hash has changed.\n"
                        f"  Catalog hash: {entry['index_hash'][:16]}...\n"
                        f"  Current hash: {full_hash[:16]}...\n"
                        f"This means the dataset was mutated after registration.\n"
                        f"To fix: unregister the old entry, then re-register."
                    )

        # Compute summary
        summary = _compute_index_summary(index_path)
        blob_count, total_mb = _compute_blob_stats(dataset_dir)

        # Build ID
        if name is None:
            name = dataset_dir.name
        # Sanitize name: replace spaces and special chars
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        n_traj = summary["n_rows"]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        dataset_id = f"{safe_name}_{n_traj}traj_{timestamp}_{short}"

        # Deduplicate ID if needed
        base_id = dataset_id
        counter = 2
        while dataset_id in self._data["datasets"]:
            dataset_id = f"{base_id}_{counter}"
            counter += 1

        # Build entry
        entry = {
            "path": str(path),
            "created": datetime.now(timezone.utc).isoformat(),
            "n_trajectories": n_traj,
            "index_hash": full_hash,
            "format_version": "v2",
            "index_summary": summary,
            "blob_count": blob_count,
            "total_size_mb": total_mb,
            "tags": tags or [],
            "splits": {},
        }

        self._data["datasets"][dataset_id] = entry
        self._save()
        return dataset_id

    # -------------------------------------------------------------------
    # Unregistration
    # -------------------------------------------------------------------

    def unregister(self, dataset_id: str) -> None:
        """Remove a dataset from the catalog. Does NOT delete files."""
        if dataset_id not in self._data["datasets"]:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' not found in catalog."
            )
        del self._data["datasets"][dataset_id]
        self._save()

    # -------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------

    def get(self, dataset_id: str) -> dict:
        """Get the catalog entry for a dataset. Raises DatasetNotFoundError."""
        if dataset_id not in self._data["datasets"]:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_id}' not found in catalog.\n"
                f"Registered datasets: {list(self._data['datasets'].keys())}"
            )
        return self._data["datasets"][dataset_id]

    def list_datasets(self) -> list[str]:
        """Return all registered dataset IDs."""
        return list(self._data["datasets"].keys())

    def find_by_tag(self, tag: str) -> list[str]:
        """Return dataset IDs that have the given tag."""
        return [
            did for did, entry in self._data["datasets"].items()
            if tag in entry.get("tags", [])
        ]

    def find_by_path(self, path: str | Path) -> str | None:
        """Find a dataset ID by its path. Returns None if not found."""
        target = str(Path(path).resolve())
        for did, entry in self._data["datasets"].items():
            if str(Path(entry["path"]).resolve()) == target:
                return did
        # Also try string match
        target_str = str(path)
        for did, entry in self._data["datasets"].items():
            if entry["path"] == target_str:
                return did
        return None

    # -------------------------------------------------------------------
    # Integrity verification
    # -------------------------------------------------------------------

    def verify(self, dataset_id: str) -> tuple[bool, str]:
        """Verify a dataset's index hash matches the catalog.

        Returns:
            (is_intact, message)
        """
        entry = self.get(dataset_id)
        dataset_dir = Path(entry["path"])
        index_path = dataset_dir / "index.parquet"

        if not index_path.exists():
            return False, f"index.parquet not found at {index_path}"

        current_hash = _compute_index_hash(index_path)
        if current_hash == entry["index_hash"]:
            return True, "INTACT"
        else:
            return False, (
                f"HASH MISMATCH: catalog={entry['index_hash'][:16]}... "
                f"current={current_hash[:16]}..."
            )

    def verify_all(self) -> dict[str, tuple[bool, str]]:
        """Verify all registered datasets. Returns {dataset_id: (ok, msg)}."""
        results = {}
        for did in self.list_datasets():
            results[did] = self.verify(did)
        return results

    # -------------------------------------------------------------------
    # Splits
    # -------------------------------------------------------------------

    def define_split(
        self,
        dataset_id: str,
        split_name: str,
        traj_ids: list[int] | None = None,
        predicate: Callable[[dict], bool] | None = None,
    ) -> list[int]:
        """Define a named split as a materialized list of traj_ids.

        Either provide traj_ids directly, or a predicate function that
        receives a row dict and returns True for included trajectories.
        The predicate is evaluated against the parquet index at definition
        time and the resulting traj_ids are stored in the catalog.

        Args:
            dataset_id: The dataset to split.
            split_name: Name for this split (e.g., "train", "val").
            traj_ids: Explicit list of trajectory IDs.
            predicate: Function taking a row dict, returning bool.

        Returns:
            The materialized list of traj_ids in this split.
        """
        entry = self.get(dataset_id)

        if traj_ids is None and predicate is None:
            raise ValueError("Must provide either traj_ids or predicate")

        if traj_ids is not None:
            materialized = sorted(traj_ids)
        else:
            # Evaluate predicate against the parquet index
            import pyarrow.parquet as pq

            dataset_dir = Path(entry["path"])
            index_path = dataset_dir / "index.parquet"
            table = pq.read_table(str(index_path))
            rows = table.to_pylist()

            materialized = sorted(
                row["traj_id"] for row in rows if predicate(row)
            )

        entry.setdefault("splits", {})[split_name] = materialized
        self._save()
        return materialized

    def get_split(self, dataset_id: str, split_name: str) -> list[int]:
        """Get the traj_ids for a named split."""
        entry = self.get(dataset_id)
        splits = entry.get("splits", {})
        if split_name not in splits:
            available = list(splits.keys()) if splits else ["none"]
            raise KeyError(
                f"Split '{split_name}' not found for dataset '{dataset_id}'. "
                f"Available: {available}"
            )
        return splits[split_name]

    def remove_split(self, dataset_id: str, split_name: str) -> None:
        """Remove a named split."""
        entry = self.get(dataset_id)
        splits = entry.get("splits", {})
        if split_name in splits:
            del splits[split_name]
            self._save()

    # -------------------------------------------------------------------
    # Raw access
    # -------------------------------------------------------------------

    @property
    def raw(self) -> dict:
        """Direct access to the catalog data dict (for serialization)."""
        return self._data


# ---------------------------------------------------------------------------
# CatalogedDataset: safe loading with integrity checks
# ---------------------------------------------------------------------------

class CatalogedDataset:
    """A dataset loaded through the catalog with integrity validation.

    Wraps the raw parquet table (no torch dependency) and provides
    metadata-level access. For tensor loading, use .to_reader() to get
    a DatasetReader (which requires torch/safetensors).
    """

    def __init__(
        self,
        dataset_id: str,
        entry: dict,
        table: Any,  # pa.Table
        split_traj_ids: list[int] | None = None,
    ):
        self.dataset_id = dataset_id
        self.entry = entry
        self._table = table
        self._split_traj_ids = split_traj_ids

        # If split, filter table
        if split_traj_ids is not None:
            import pyarrow.compute as pc
            id_set = set(split_traj_ids)
            mask = [
                self._table.column("traj_id")[i].as_py() in id_set
                for i in range(len(self._table))
            ]
            import pyarrow as pa
            self._table = self._table.filter(pa.array(mask))

    @classmethod
    def load(
        cls,
        dataset_id: str,
        split: str | None = None,
        catalog_path: str | Path = "catalog.json",
    ) -> CatalogedDataset:
        """Load a dataset by ID with integrity validation.

        1. Looks up the dataset in the catalog
        2. Verifies the index hash matches (catches mutation/contamination)
        3. Loads the parquet index
        4. If split is specified, filters to that split's traj_ids
        5. Returns a CatalogedDataset ready for use

        Raises:
            DatasetIntegrityError: If the index hash doesn't match
            DatasetNotFoundError: If the dataset ID isn't in the catalog
        """
        catalog = DatasetCatalog(catalog_path)
        entry = catalog.get(dataset_id)

        # Verify integrity
        dataset_dir = Path(entry["path"])
        index_path = dataset_dir / "index.parquet"

        if not index_path.exists():
            raise FileNotFoundError(
                f"index.parquet not found at {index_path}"
            )

        current_hash = _compute_index_hash(index_path)
        if current_hash != entry["index_hash"]:
            raise DatasetIntegrityError(
                f"Dataset '{dataset_id}' has been modified since registration!\n"
                f"  Catalog hash: {entry['index_hash'][:16]}...\n"
                f"  Current hash: {current_hash[:16]}...\n"
                f"The index file at {index_path} does not match.\n"
                f"This could indicate accidental contamination (e.g., a "
                f"generation script appending to an existing dataset)."
            )

        # Load parquet
        import pyarrow.parquet as pq
        table = pq.read_table(str(index_path))

        # Get split traj_ids if requested
        split_traj_ids = None
        if split is not None:
            split_traj_ids = catalog.get_split(dataset_id, split)

        return cls(dataset_id, entry, table, split_traj_ids)

    @property
    def n_trajectories(self) -> int:
        return len(self._table)

    @property
    def traj_ids(self) -> list[int]:
        return self._table.column("traj_id").to_pylist()

    @property
    def table(self) -> Any:
        """The underlying pyarrow Table (possibly filtered by split)."""
        return self._table

    @property
    def path(self) -> Path:
        return Path(self.entry["path"])

    def iter_metadata(self) -> list[dict]:
        """Return all rows as dicts (metadata only, no tensors)."""
        return self._table.to_pylist()

    def to_reader(self) -> Any:
        """Create a DatasetReader for tensor access.

        This imports from futudiffu.dataset_v2 which requires torch.
        The reader is NOT filtered by split -- use self.traj_ids to
        know which trajectories belong to this split.
        """
        # Late import to avoid torch dependency at module level
        from futudiffu.dataset_v2 import DatasetReader
        return DatasetReader(str(self.path))


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def register_dataset(
    path: str | Path,
    name: str | None = None,
    tags: list[str] | None = None,
    catalog_path: str | Path = "catalog.json",
) -> str:
    """Register a V2 dataset. See DatasetCatalog.register()."""
    catalog = DatasetCatalog(catalog_path)
    return catalog.register(path, name=name, tags=tags)


def load_dataset(
    dataset_id: str,
    split: str | None = None,
    catalog_path: str | Path = "catalog.json",
) -> CatalogedDataset:
    """Load a dataset by ID. See CatalogedDataset.load()."""
    return CatalogedDataset.load(dataset_id, split=split, catalog_path=catalog_path)
