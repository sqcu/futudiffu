"""Bin assignment for packed FlexAttention inference launches.

Pure Python. No torch. No src_ii imports.
"""

from __future__ import annotations


def _pad32(n: int) -> int:
    return n + ((-n) % 32)


def compute_entry_seq_len(
    latent_h: int,
    latent_w: int,
    cap_len: int,
    patch_size: int = 2,
) -> int:
    """Compute tokens for one entry: latent patches + padded text.

    latent_h, latent_w: spatial dims after VAE encode (pixels / 8).
    cap_len: raw text token count before padding.
    """
    img_tokens = (latent_h // patch_size) * (latent_w // patch_size)
    return _pad32(img_tokens) + _pad32(cap_len)


def pack_for_inference(
    entry_seq_lens: list[int],
    max_total_len: int = 4224,
    target_bins: int | None = None,
) -> list[list[int]]:
    """Assign entries to bins for packed FlexAttention launches.

    Returns list of bins; each bin is a list of entry indices.
    Every entry appears in exactly one bin. Bin sum <= max_total_len.
    """
    n = len(entry_seq_lens)
    if n == 0:
        return []

    # First-Fit Decreasing: sort by descending seq_len.
    order = sorted(range(n), key=lambda i: entry_seq_lens[i], reverse=True)
    bins: list[list[int]] = []
    totals: list[int] = []

    for idx in order:
        length = entry_seq_lens[idx]
        placed = False
        for b in range(len(bins)):
            if totals[b] + length <= max_total_len:
                bins[b].append(idx)
                totals[b] += length
                placed = True
                break
        if not placed:
            bins.append([idx])
            totals.append(length)

    # Split bins to approach target_bins (best-effort; singletons can't split).
    if target_bins is not None:
        while len(bins) < target_bins:
            # Largest multi-entry bin.
            best = next(
                (i for i in sorted(range(len(bins)),
                                   key=lambda i: totals[i], reverse=True)
                 if len(bins[i]) > 1),
                None,
            )
            if best is None:
                break
            b = bins.pop(best)
            totals.pop(best)
            mid = len(b) // 2
            for half in (b[:mid], b[mid:]):
                bins.append(half)
                totals.append(sum(entry_seq_lens[j] for j in half))

    return bins
