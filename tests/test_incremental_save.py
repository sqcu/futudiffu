"""Tests for src_ii/incremental_save.py.

Tests TrainingCurveWriter (JSONL append), PeriodicSaver (interval gating),
atomic_json_save, and load_training_curve_jsonl (malformed line tolerance).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src_ii.incremental_save import (
    TrainingCurveWriter,
    load_training_curve_jsonl,
    PeriodicSaver,
    atomic_json_save,
)


class TestTrainingCurveWriter:
    """Tests for TrainingCurveWriter JSONL append semantics."""

    def test_round_trip(self, tmp_path):
        """Write 5 steps, load back, verify identical."""
        path = tmp_path / "curve.jsonl"
        entries = [{"step": i, "loss": 0.5 - i * 0.1} for i in range(5)]

        with TrainingCurveWriter(path) as writer:
            for e in entries:
                writer.write_step(e)
            assert writer.n_written == 5

        loaded = load_training_curve_jsonl(path)
        assert len(loaded) == 5
        for orig, back in zip(entries, loaded):
            assert orig["step"] == back["step"]
            assert abs(orig["loss"] - back["loss"]) < 1e-10

    def test_append_mode(self, tmp_path):
        """Append to existing file, verify total count."""
        path = tmp_path / "curve.jsonl"

        # Write 3 entries
        with TrainingCurveWriter(path) as w:
            for i in range(3):
                w.write_step({"step": i})

        # Append 3 more
        with TrainingCurveWriter(path, append=True) as w:
            for i in range(3, 6):
                w.write_step({"step": i})

        loaded = load_training_curve_jsonl(path)
        assert len(loaded) == 6
        assert [e["step"] for e in loaded] == list(range(6))

    def test_flush_per_write(self, tmp_path):
        """Each write is immediately readable without closing."""
        path = tmp_path / "curve.jsonl"
        writer = TrainingCurveWriter(path)
        writer.write_step({"step": 0, "loss": 0.7})

        # Read without closing writer
        loaded = load_training_curve_jsonl(path)
        assert len(loaded) == 1
        assert loaded[0]["step"] == 0

        writer.close()

    def test_default_serializer(self, tmp_path):
        """Non-JSON-native types use default=str."""
        path = tmp_path / "curve.jsonl"
        with TrainingCurveWriter(path) as w:
            w.write_step({"step": 0, "path": Path("/foo/bar")})

        loaded = load_training_curve_jsonl(path)
        assert loaded[0]["path"] == str(Path("/foo/bar"))


class TestLoadTrainingCurveJsonl:
    """Tests for JSONL loader with malformed line tolerance."""

    def test_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        assert load_training_curve_jsonl(path) == []

    def test_nonexistent_file(self, tmp_path):
        """Missing file returns empty list."""
        assert load_training_curve_jsonl(tmp_path / "nope.jsonl") == []

    def test_malformed_trailing_line(self, tmp_path):
        """Malformed last line (crash mid-write) is skipped."""
        path = tmp_path / "curve.jsonl"
        # Write 3 good lines + 1 truncated
        lines = [
            json.dumps({"step": 0, "loss": 0.7}),
            json.dumps({"step": 1, "loss": 0.6}),
            json.dumps({"step": 2, "loss": 0.5}),
            '{"step": 3, "loss": 0.4',  # truncated
        ]
        path.write_text("\n".join(lines) + "\n")

        loaded = load_training_curve_jsonl(path)
        assert len(loaded) == 3
        assert loaded[-1]["step"] == 2

    def test_blank_lines_skipped(self, tmp_path):
        """Blank lines between entries are harmless."""
        path = tmp_path / "curve.jsonl"
        content = '{"step": 0}\n\n\n{"step": 1}\n\n'
        path.write_text(content)

        loaded = load_training_curve_jsonl(path)
        assert len(loaded) == 2


class TestPeriodicSaver:
    """Tests for PeriodicSaver interval gating."""

    def test_saves_at_interval(self):
        """Saves at steps divisible by interval."""
        saved = []
        saver = PeriodicSaver(save_fn=lambda step: saved.append(step), interval=3)

        for step in range(10):
            saver.maybe_save(step)

        assert saved == [0, 3, 6, 9]

    def test_flush_dedup(self):
        """Flush does not double-save if already saved at that step."""
        saved = []
        saver = PeriodicSaver(save_fn=lambda step: saved.append(step), interval=5)

        saver.maybe_save(0)   # saves
        saver.flush(0)        # should NOT save again
        assert saved == [0]

    def test_flush_forces_save(self):
        """Flush saves at non-interval steps."""
        saved = []
        saver = PeriodicSaver(save_fn=lambda step: saved.append(step), interval=100)

        saver.maybe_save(7)   # does not save (7 % 100 != 0)
        saver.flush(7)        # forces save
        assert saved == [7]

    def test_maybe_save_returns_bool(self):
        """maybe_save returns True when it saves, False otherwise."""
        saver = PeriodicSaver(save_fn=lambda step: None, interval=5)
        assert saver.maybe_save(0) is True
        assert saver.maybe_save(1) is False
        assert saver.maybe_save(5) is True


class TestAtomicJsonSave:
    """Tests for atomic JSON save."""

    def test_round_trip(self, tmp_path):
        """Write and load back."""
        path = tmp_path / "data.json"
        data = {"key": "value", "nested": [1, 2, 3]}
        atomic_json_save(data, path)

        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        """Creates intermediate directories."""
        path = tmp_path / "a" / "b" / "c" / "data.json"
        atomic_json_save({"x": 1}, path)
        assert path.exists()

    def test_overwrites_existing(self, tmp_path):
        """Atomically replaces existing file."""
        path = tmp_path / "data.json"
        atomic_json_save({"version": 1}, path)
        atomic_json_save({"version": 2}, path)

        loaded = json.loads(path.read_text())
        assert loaded["version"] == 2

    def test_no_temp_file_left(self, tmp_path):
        """No .tmp files remain after write."""
        path = tmp_path / "data.json"
        atomic_json_save({"x": 1}, path)

        temps = list(tmp_path.glob("*.tmp"))
        assert len(temps) == 0
