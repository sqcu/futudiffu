"""Upload dataset files to a HuggingFace dataset repository.

Works with both packed (safetensors + manifest.jsonl) and unpacked
(traj_NNNNNN/ directories with .pt + meta.json) formats.

Two modes:
  --once   Upload everything and exit.
  --watch  Poll for new/modified files and upload incrementally.

Auth: HF_TOKEN env var or `huggingface-cli login` default.

Examples:
    # One-shot upload of packed dataset
    python upload_to_hf.py --repo-id user/futudiffu-btrm --source-dir packed_dataset --once

    # Watch mode alongside trajectory generation
    python upload_to_hf.py --repo-id user/futudiffu-btrm --source-dir packed_dataset --watch

    # Upload unpacked trajectories (per-directory .pt files)
    python upload_to_hf.py --repo-id user/futudiffu-btrm --source-dir btrm_dataset --once

    # Upload adapters/checkpoints from a training run
    python upload_to_hf.py --repo-id user/futudiffu-run01 --source-dir run_output --watch --interval 60
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, CommitOperationAdd, RepoUrl
from huggingface_hub.errors import RepositoryNotFoundError


# File extensions we care about. Everything else is ignored.
UPLOAD_EXTENSIONS = {".safetensors", ".json", ".jsonl", ".pt", ".parquet"}

MANIFEST_NAME = ".uploaded_manifest.json"


def load_manifest(source_dir: Path) -> dict:
    """Load the upload tracking manifest.

    Returns dict mapping relative_path -> {"size": int, "mtime": float, "committed": bool}.
    """
    manifest_path = source_dir / MANIFEST_NAME
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def save_manifest(source_dir: Path, manifest: dict):
    """Write the upload tracking manifest atomically."""
    manifest_path = source_dir / MANIFEST_NAME
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(manifest_path)


def discover_files(source_dir: Path) -> list[Path]:
    """Walk source_dir and return all uploadable files, sorted by path."""
    files = []
    for root, _dirs, filenames in os.walk(source_dir):
        root_path = Path(root)
        for name in filenames:
            if name == MANIFEST_NAME:
                continue
            fp = root_path / name
            if fp.suffix in UPLOAD_EXTENSIONS:
                files.append(fp)
    files.sort()
    return files


def find_new_or_modified(
    source_dir: Path, manifest: dict, stability_wait: float = 0.0,
) -> list[Path]:
    """Return files that are new or modified since last upload.

    Args:
        stability_wait: If > 0, re-stat each candidate after this many seconds
            and only include it if size/mtime haven't changed. Prevents
            uploading partially-written files in watch mode.
    """
    all_files = discover_files(source_dir)
    candidates = []
    for fp in all_files:
        rel = str(fp.relative_to(source_dir))
        stat = fp.stat()
        entry = manifest.get(rel)
        if entry is None:
            candidates.append((fp, stat.st_size, stat.st_mtime))
        elif entry["size"] != stat.st_size or entry["mtime"] < stat.st_mtime:
            candidates.append((fp, stat.st_size, stat.st_mtime))

    if not candidates or stability_wait <= 0:
        return [fp for fp, _, _ in candidates]

    # Wait and re-check: skip files still being written
    time.sleep(stability_wait)
    stable = []
    for fp, prev_size, prev_mtime in candidates:
        try:
            stat2 = fp.stat()
        except OSError:
            continue  # file disappeared
        if stat2.st_size == prev_size and stat2.st_mtime == prev_mtime:
            stable.append(fp)
    return stable


def ensure_repo(api: HfApi, repo_id: str) -> str:
    """Create the dataset repo if it doesn't exist. Returns the repo_id."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repo exists: {repo_id}")
    except RepositoryNotFoundError:
        print(f"Creating dataset repo: {repo_id}")
        url: RepoUrl = api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)
        print(f"Created: {url}")
    return repo_id


def upload_files(api: HfApi, repo_id: str, source_dir: Path, files: list[Path], manifest: dict):
    """Upload a batch of files in a single commit and update the manifest."""
    if not files:
        return 0

    # Build commit operations
    operations = []
    file_info = []  # (fp, rel, stat) for manifest update after commit
    total = len(files)

    for i, fp in enumerate(files):
        rel = str(fp.relative_to(source_dir))
        path_in_repo = rel.replace("\\", "/")
        size_mb = fp.stat().st_size / (1024 * 1024)
        print(f"  [{i+1}/{total}] {path_in_repo} ({size_mb:.2f} MB)")
        operations.append(CommitOperationAdd(
            path_in_repo=path_in_repo,
            path_or_fileobj=str(fp),
        ))
        file_info.append((fp, rel))

    print(f"  Committing {total} file(s) to {repo_id}...", end=" ", flush=True)
    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Upload {total} file(s)",
        )
        print("done")
        # Record all uploaded files in manifest
        for fp, rel in file_info:
            stat = fp.stat()
            manifest[rel] = {
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "committed": True,
            }
        save_manifest(source_dir, manifest)
        return total
    except Exception as e:
        print(f"FAILED: {e}")
        # Don't update manifest -- entire batch will retry next cycle
        return 0


def run_once(api: HfApi, repo_id: str, source_dir: Path):
    """Upload all new/modified files and exit."""
    manifest = load_manifest(source_dir)
    pending = find_new_or_modified(source_dir, manifest)

    if not pending:
        print("Nothing to upload. All files are up to date.")
        return

    print(f"Found {len(pending)} file(s) to upload.")
    n = upload_files(api, repo_id, source_dir, pending, manifest)
    print(f"Uploaded {n}/{len(pending)} file(s).")


def run_watch(api: HfApi, repo_id: str, source_dir: Path, interval: int):
    """Poll for new files and upload incrementally."""
    print(f"Watching {source_dir} every {interval}s. Ctrl+C to stop.")
    manifest = load_manifest(source_dir)

    cycle = 0
    while True:
        cycle += 1
        # 2s stability wait: re-stat after delay to skip files still being written
        pending = find_new_or_modified(source_dir, manifest, stability_wait=2.0)

        if pending:
            print(f"\n[cycle {cycle}] {len(pending)} new/modified file(s)")
            n = upload_files(api, repo_id, source_dir, pending, manifest)
            print(f"[cycle {cycle}] Uploaded {n}/{len(pending)}.")
        else:
            ts = time.strftime("%H:%M:%S")
            print(f"\r[{ts}] cycle {cycle}: no new files, sleeping {interval}s", end="", flush=True)

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Upload dataset files to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo (e.g. user/futudiffu-btrm)")
    parser.add_argument("--source-dir", required=True, help="Local directory to upload from")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds for --watch mode")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="Upload everything and exit")
    mode.add_argument("--watch", action="store_true", help="Poll for new files and upload incrementally")

    args = parser.parse_args()
    source_dir = Path(args.source_dir).resolve()

    if not source_dir.is_dir():
        print(f"ERROR: {source_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Auth: HF_TOKEN env > .supersekrit file > huggingface-cli login default
    token = os.environ.get("HF_TOKEN")
    if not token:
        supersekrit = Path(__file__).resolve().parent.parent / ".supersekrit"
        if supersekrit.exists():
            for line in supersekrit.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    token = line
                    break
    api = HfApi(token=token)

    # Verify auth works
    try:
        user = api.whoami()
        print(f"Authenticated as: {user['name']}")
    except Exception as e:
        print(f"ERROR: HuggingFace auth failed: {e}", file=sys.stderr)
        print("Set HF_TOKEN env var or run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    repo_id = ensure_repo(api, args.repo_id)

    if args.once:
        run_once(api, repo_id, source_dir)
    elif args.watch:
        run_watch(api, repo_id, source_dir, args.interval)


if __name__ == "__main__":
    main()
