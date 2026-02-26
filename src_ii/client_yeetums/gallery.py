"""In-memory gallery with JSONL sidecar persistence.

Stores generated images as PNG files in a gallery directory. Metadata is
appended to a JSONL sidecar file and kept in memory for fast listing.

No torch, no safetensors.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class Gallery:
    """Thread-safe gallery of generated images.

    Each entry has:
      - A unique ID (UUID4 hex)
      - A PNG file on disk
      - A metadata dict (prompt, seed, resolution, timing, etc.)

    The gallery directory structure:
      gallery_dir/
        gallery.jsonl      (append-only metadata log)
        {id}.png           (generated images)
    """

    def __init__(self, gallery_dir: str | Path = "yeetums_gallery"):
        self._dir = Path(gallery_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sidecar = self._dir / "gallery.jsonl"
        self._entries: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing entries from the JSONL sidecar."""
        if not self._sidecar.exists():
            return
        for line in self._sidecar.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                self._entries.append(entry)
                self._by_id[entry["id"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue

    def add(
        self,
        png_bytes: bytes,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Add a generated image to the gallery.

        Args:
            png_bytes: Raw PNG file bytes.
            metadata: Generation metadata (prompt, seed, etc.).

        Returns:
            Complete gallery entry dict with id, image_url, timestamp.
        """
        entry_id = uuid.uuid4().hex[:12]
        png_path = self._dir / f"{entry_id}.png"
        png_path.write_bytes(png_bytes)

        entry = {
            "id": entry_id,
            "timestamp": time.time(),
            "image_url": f"/api/gallery/{entry_id}/image",
            **metadata,
        }

        self._entries.append(entry)
        self._by_id[entry_id] = entry

        with open(self._sidecar, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def list_entries(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List gallery entries, newest first.

        Args:
            limit: Max entries to return.
            offset: Skip this many entries from the end.

        Returns:
            List of entry dicts.
        """
        reversed_entries = list(reversed(self._entries))
        return reversed_entries[offset:offset + limit]

    def get(self, entry_id: str) -> dict[str, Any] | None:
        """Get a gallery entry by ID."""
        return self._by_id.get(entry_id)

    def get_image_path(self, entry_id: str) -> Path | None:
        """Get the path to a gallery image."""
        path = self._dir / f"{entry_id}.png"
        return path if path.exists() else None

    @property
    def total(self) -> int:
        """Total number of entries."""
        return len(self._entries)
