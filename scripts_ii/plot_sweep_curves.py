r"""Analyze r_theta LR sweep training curves and produce comparison plots.

Reads training_curve.json from N LR probes (auto-discovered from sweep dir),
computes EMA-smoothed loss, d_loss/d_step, and generates:
  1. Down-and-to-the-left phase-space plot (loss level vs descent rate)
  2. Loss vs step with EMA overlays
  3. Grad norm vs step
  4. Step time vs step
  5. Pre-clip grad norm vs step (v2 data only)
  6. BT loss vs step (v2 data only)
  7. Accuracy vs step

All rendering uses PIL (Pillow). No matplotlib.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\plot_sweep_curves.py
  # or with explicit sweep dir:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts_ii\plot_sweep_curves.py ^
      --sweep-dir F:\dox\repos\ai\futudiffu\rtheta_sweep_output_v2

Output:
  <sweep_dir>/plots/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# src_ii imports
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src_ii.stats import finite_differences


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_probes(sweep_dir: Path) -> list[str]:
    """Auto-discover probe directories (lr_*) sorted by LR value descending."""
    probes = []
    for p in sorted(sweep_dir.iterdir()):
        if p.is_dir() and p.name.startswith("lr_") and (p / "training_curve.json").exists():
            probes.append(p.name)

    # Sort by actual LR value (descending: highest LR first)
    def lr_sort_key(name: str) -> float:
        # "lr_1e-02" -> 0.01, "lr_3e-03" -> 0.003
        lr_str = name.replace("lr_", "").replace("-", "e-")
        # Handle the format: "1e-02" is already valid, "3e-03" is already valid
        # But our format is "1e-02" which means "1e-02", need to parse carefully
        parts = name.replace("lr_", "")  # "1e-02" or "3e-03"
        # Convert "1e-02" -> 1e-02, "3e-03" -> 3e-03
        coeff, exp = parts.split("e-")
        return -float(f"{coeff}e-{exp}")

    probes.sort(key=lr_sort_key)
    return probes


def load_curve(sweep_dir: Path, name: str) -> list[dict]:
    path = sweep_dir / name / "training_curve.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# EMA + finite difference
# ---------------------------------------------------------------------------

def ema(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average. alpha is the weight on the new sample."""
    out = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# PIL chart renderer
# ---------------------------------------------------------------------------

class PILChart:
    """Minimal line/scatter chart renderer using PIL.

    Coordinate system:
        - Data space: (x_min..x_max, y_min..y_max)
        - Plot area: a rectangle within the image, leaving margins for
          axis labels, title, and legend.
    """

    def __init__(
        self,
        width: int = 900,
        height: int = 600,
        bg_color: str = "#ffffff",
        margin_left: int = 90,
        margin_right: int = 30,
        margin_top: int = 50,
        margin_bottom: int = 60,
    ):
        self.img_w = width
        self.img_h = height
        self.bg_color = bg_color
        self.ml = margin_left
        self.mr = margin_right
        self.mt = margin_top
        self.mb = margin_bottom

        self.plot_w = width - margin_left - margin_right
        self.plot_h = height - margin_top - margin_bottom

        self.img = Image.new("RGB", (width, height), bg_color)
        self.draw = ImageDraw.Draw(self.img)

        self.series: list[dict] = []
        self.title: str = ""
        self.x_label: str = ""
        self.y_label: str = ""

        # Try to load a monospace font; fall back to default
        self.font = ImageFont.load_default()
        self.font_small = self.font

    def set_title(self, title: str):
        self.title = title

    def set_labels(self, x_label: str, y_label: str):
        self.x_label = x_label
        self.y_label = y_label

    def add_line(
        self,
        xs: list[float],
        ys: list[float],
        color: str = "#000000",
        label: str = "",
        line_width: int = 2,
        style: str = "solid",  # "solid", "dashed"
    ):
        self.series.append({
            "type": "line",
            "xs": xs,
            "ys": ys,
            "color": color,
            "label": label,
            "line_width": line_width,
            "style": style,
        })

    def add_scatter(
        self,
        xs: list[float],
        ys: list[float],
        color: str = "#000000",
        label: str = "",
        size: int = 3,
    ):
        self.series.append({
            "type": "scatter",
            "xs": xs,
            "ys": ys,
            "color": color,
            "label": label,
            "size": size,
        })

    def _compute_bounds(self):
        all_xs, all_ys = [], []
        for s in self.series:
            all_xs.extend(s["xs"])
            all_ys.extend(s["ys"])

        if not all_xs or not all_ys:
            return 0, 1, 0, 1

        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)

        # Add 5% padding
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        x_min -= 0.05 * x_range
        x_max += 0.05 * x_range
        y_min -= 0.05 * y_range
        y_max += 0.05 * y_range

        return x_min, x_max, y_min, y_max

    def _data_to_pixel(self, x: float, y: float, x_min, x_max, y_min, y_max):
        """Map data coordinates to pixel coordinates."""
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        px = self.ml + (x - x_min) / x_range * self.plot_w
        py = self.mt + (1.0 - (y - y_min) / y_range) * self.plot_h
        return int(px), int(py)

    def _draw_axes(self, x_min, x_max, y_min, y_max):
        d = self.draw
        # Plot area border
        d.rectangle(
            [self.ml, self.mt, self.ml + self.plot_w, self.mt + self.plot_h],
            outline="#cccccc",
            width=1,
        )

        # Grid lines and tick labels
        n_x_ticks = 6
        n_y_ticks = 6

        for i in range(n_x_ticks + 1):
            frac = i / n_x_ticks
            val = x_min + frac * (x_max - x_min)
            px = self.ml + int(frac * self.plot_w)
            # Grid line
            d.line([(px, self.mt), (px, self.mt + self.plot_h)], fill="#eeeeee", width=1)
            # Tick label
            label = _format_tick(val)
            d.text((px, self.mt + self.plot_h + 5), label, fill="#333333", font=self.font, anchor="mt")

        for i in range(n_y_ticks + 1):
            frac = i / n_y_ticks
            val = y_min + frac * (y_max - y_min)
            py = self.mt + int((1.0 - frac) * self.plot_h)
            # Grid line
            d.line([(self.ml, py), (self.ml + self.plot_w, py)], fill="#eeeeee", width=1)
            # Tick label
            label = _format_tick(val)
            d.text((self.ml - 5, py), label, fill="#333333", font=self.font, anchor="rm")

        # Axis labels
        if self.x_label:
            d.text(
                (self.ml + self.plot_w // 2, self.img_h - 10),
                self.x_label,
                fill="#000000",
                font=self.font,
                anchor="mb",
            )
        if self.y_label:
            # Rotate would be ideal but PIL doesn't easily support rotated text
            # with the default font. Just place it at top-left of y axis.
            d.text(
                (5, self.mt + self.plot_h // 2),
                self.y_label,
                fill="#000000",
                font=self.font,
                anchor="lm",
            )

    def _draw_title(self):
        if self.title:
            self.draw.text(
                (self.ml + self.plot_w // 2, 10),
                self.title,
                fill="#000000",
                font=self.font,
                anchor="mt",
            )

    def _draw_legend(self):
        labeled = [s for s in self.series if s.get("label")]
        if not labeled:
            return
        d = self.draw
        x_start = self.ml + self.plot_w - 200
        y_start = self.mt + 10
        line_height = 18

        # Background
        n = len(labeled)
        d.rectangle(
            [x_start - 5, y_start - 5, x_start + 195, y_start + n * line_height + 5],
            fill="#ffffff",
            outline="#cccccc",
        )

        for i, s in enumerate(labeled):
            y = y_start + i * line_height
            color = s["color"]
            # Color swatch
            if s["type"] == "line":
                style = s.get("style", "solid")
                if style == "dashed":
                    for dx in range(0, 25, 6):
                        d.line([(x_start, y + 7), (x_start + min(dx + 3, 25), y + 7)],
                               fill=color, width=2)
                else:
                    d.line([(x_start, y + 7), (x_start + 25, y + 7)], fill=color, width=2)
            else:
                d.ellipse(
                    [x_start + 8, y + 3, x_start + 17, y + 12],
                    fill=color,
                )
            d.text((x_start + 30, y + 2), s["label"], fill="#333333", font=self.font)

    def _draw_series(self, x_min, x_max, y_min, y_max):
        d = self.draw
        for s in self.series:
            xs, ys = s["xs"], s["ys"]
            color = s["color"]

            if s["type"] == "line":
                lw = s.get("line_width", 2)
                style = s.get("style", "solid")
                points = []
                for x, y in zip(xs, ys):
                    px, py = self._data_to_pixel(x, y, x_min, x_max, y_min, y_max)
                    points.append((px, py))
                if len(points) < 2:
                    continue
                if style == "dashed":
                    for i in range(len(points) - 1):
                        # Draw every other segment
                        if i % 2 == 0:
                            d.line([points[i], points[i + 1]], fill=color, width=lw)
                else:
                    d.line(points, fill=color, width=lw)
            elif s["type"] == "scatter":
                sz = s.get("size", 3)
                for x, y in zip(xs, ys):
                    px, py = self._data_to_pixel(x, y, x_min, x_max, y_min, y_max)
                    d.ellipse(
                        [px - sz, py - sz, px + sz, py + sz],
                        fill=color,
                    )

    def render(self) -> Image.Image:
        x_min, x_max, y_min, y_max = self._compute_bounds()
        self._draw_axes(x_min, x_max, y_min, y_max)
        self._draw_series(x_min, x_max, y_min, y_max)
        self._draw_title()
        self._draw_legend()
        return self.img

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.render().save(str(path))
        print(f"  Saved: {path}")


def _format_tick(val: float) -> str:
    """Format a tick value for display."""
    if abs(val) < 0.001 and val != 0:
        return f"{val:.1e}"
    if abs(val) >= 1000:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.1f}"
    if abs(val) >= 1:
        return f"{val:.2f}"
    return f"{val:.3f}"


# ---------------------------------------------------------------------------
# Color palette for N probes
# ---------------------------------------------------------------------------

# Distinct colors for up to 7 probes, ordered from warm to cool
PALETTE = [
    ("#cc3333", "#ee8888", "#880000"),  # red
    ("#cc7733", "#eebb88", "#884400"),  # orange
    ("#33aa33", "#88dd88", "#006600"),  # green
    ("#3333cc", "#8888ee", "#000088"),  # blue
    ("#9933cc", "#cc88ee", "#550088"),  # purple
    ("#33aaaa", "#88dddd", "#006666"),  # teal
    ("#aa3377", "#dd88aa", "#660044"),  # magenta
]


def get_colors(idx: int) -> tuple[str, str, str]:
    """Return (solid, light, dark) colors for probe index."""
    return PALETTE[idx % len(PALETTE)]


def probe_label(name: str) -> str:
    """Convert dir name like 'lr_1e-02' to display label like 'lr=1e-02'."""
    return name.replace("lr_", "lr=")


# ---------------------------------------------------------------------------
# Plot generators (N-probe versions)
# ---------------------------------------------------------------------------

def plot_loss_vs_step(
    probes: dict[str, list[dict]],
    output_dir: Path,
):
    """Loss vs step with raw + EMA overlays for all probes."""
    chart = PILChart(width=1000, height=650)
    chart.set_title("Loss vs Step (r_theta LR sweep)")
    chart.set_labels("Step", "Loss")

    for i, (name, data) in enumerate(probes.items()):
        solid, light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        loss = [d["loss"] for d in data]
        ema_01 = ema(loss, 0.1)
        ema_03 = ema(loss, 0.3)

        # Raw (thin, light)
        chart.add_line(steps, loss, color=light, label="", line_width=1)
        # EMA alpha=0.1 (solid)
        chart.add_line(steps, ema_01, color=solid, label=f"{probe_label(name)} EMA(0.1)", line_width=2)
        # EMA alpha=0.3 (dashed)
        chart.add_line(steps, ema_03, color=solid, label=f"{probe_label(name)} EMA(0.3)", line_width=2, style="dashed")

    chart.save(output_dir / "loss_vs_step.png")


def plot_grad_norm_vs_step(
    probes: dict[str, list[dict]],
    output_dir: Path,
):
    """Grad norm vs step for all probes."""
    chart = PILChart(width=1000, height=650)
    chart.set_title("Grad Norm vs Step (r_theta LR sweep)")
    chart.set_labels("Step", "Grad Norm")

    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        gn = [d["grad_norm"] for d in data]
        chart.add_line(steps, gn, color=solid, label=probe_label(name), line_width=2)

    chart.save(output_dir / "grad_norm_vs_step.png")


def plot_pre_clip_grad_norm_vs_step(
    probes: dict[str, list[dict]],
    output_dir: Path,
):
    """Pre-clip grad norm vs step (v2 data only)."""
    # Check if data has pre_clip_grad_norm
    first_data = next(iter(probes.values()))
    if "pre_clip_grad_norm" not in first_data[0]:
        print("  Skipping pre-clip grad norm plot (field not present)")
        return

    chart = PILChart(width=1000, height=650)
    chart.set_title("Pre-Clip Grad Norm vs Step (r_theta LR sweep)")
    chart.set_labels("Step", "Pre-Clip Grad Norm")

    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        gn = [d["pre_clip_grad_norm"] for d in data]
        chart.add_line(steps, gn, color=solid, label=probe_label(name), line_width=2)

    chart.save(output_dir / "pre_clip_grad_norm_vs_step.png")


def plot_time_vs_step(
    probes: dict[str, list[dict]],
    output_dir: Path,
):
    """Step time vs step for all probes."""
    chart = PILChart(width=1000, height=650)
    chart.set_title("Step Time vs Step (r_theta LR sweep)")
    chart.set_labels("Step", "Time (s)")

    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        time = [d["time_s"] for d in data]
        chart.add_line(steps, time, color=solid, label=probe_label(name), line_width=2)

    chart.save(output_dir / "time_vs_step.png")


def plot_bt_loss_vs_step(
    probes: dict[str, list[dict]],
    output_dir: Path,
):
    """Per-term normalized BT loss vs step (v2 data only)."""
    first_data = next(iter(probes.values()))
    if "loss" not in first_data[0] and "bt_loss" not in first_data[0]:
        print("  Skipping BT loss plot (field not present)")
        return

    chart = PILChart(width=1000, height=650)
    chart.set_title("BT Loss (per-term avg) vs Step (r_theta LR sweep)")
    chart.set_labels("Step", "Loss (per-term avg)")

    for i, (name, data) in enumerate(probes.items()):
        solid, light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        bt = [d.get("loss", d.get("bt_loss", 0.0)) for d in data]
        ema_bt = ema(bt, 0.1)
        chart.add_line(steps, bt, color=light, label="", line_width=1)
        chart.add_line(steps, ema_bt, color=solid, label=f"{probe_label(name)} EMA(0.1)", line_width=2)

    chart.save(output_dir / "bt_loss_vs_step.png")


def plot_accuracy_vs_step(
    probes: dict[str, list[dict]],
    output_dir: Path,
):
    """Accuracy (pinkify and thisnotthat) vs step."""
    # Pinkify
    chart_p = PILChart(width=1000, height=650)
    chart_p.set_title("Accuracy (pinkify) vs Step (r_theta LR sweep)")
    chart_p.set_labels("Step", "Accuracy (pinkify)")

    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        acc = [d["accuracy_pinkify"] for d in data]
        ema_acc = ema(acc, 0.1)
        chart_p.add_line(steps, ema_acc, color=solid, label=probe_label(name), line_width=2)

    chart_p.save(output_dir / "accuracy_pinkify_vs_step.png")

    # Thisnotthat
    chart_t = PILChart(width=1000, height=650)
    chart_t.set_title("Accuracy (thisnotthat) vs Step (r_theta LR sweep)")
    chart_t.set_labels("Step", "Accuracy (thisnotthat)")

    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        acc = [d["accuracy_thisnotthat"] for d in data]
        ema_acc = ema(acc, 0.1)
        chart_t.add_line(steps, ema_acc, color=solid, label=probe_label(name), line_width=2)

    chart_t.save(output_dir / "accuracy_thisnotthat_vs_step.png")


def plot_down_and_left(
    probes: dict[str, list[dict]],
    output_dir: Path,
    alpha: float = 0.1,
):
    """Phase-space plot: X = EMA loss level, Y = d(EMA_loss)/d_step.

    The "winner" is whichever curve reaches further down (lower loss) and
    to the left (more negative derivative = descending faster).
    """
    suffix = f"alpha{alpha:.1f}".replace(".", "")
    chart = PILChart(width=1000, height=700)
    chart.set_title(f"Down-and-to-the-Left: Loss Level vs Descent Rate (EMA alpha={alpha})")
    chart.set_labels("EMA Loss (lower = better)", "d(loss)/d(step) (more negative = faster descent)")

    for i, (name, data) in enumerate(probes.items()):
        solid, light, dark = get_colors(i)
        loss = [d["loss"] for d in data]
        ema_loss = ema(loss, alpha)
        dloss = finite_differences(ema_loss)
        x_vals = ema_loss[:-1]

        # Plot as scatter with connecting lines
        chart.add_line(x_vals, dloss, color=light, label="", line_width=1)
        chart.add_scatter(x_vals, dloss, color=solid, label=probe_label(name), size=3)
        # Mark the final point with a bigger dot
        chart.add_scatter([x_vals[-1]], [dloss[-1]], color=dark, label=f"{probe_label(name)} final", size=6)

    chart.save(output_dir / f"down_and_left_{suffix}.png")


def compute_summary(
    probes: dict[str, list[dict]],
) -> dict:
    """Compute summary statistics for the sweep."""
    results = {}

    # Per-alpha EMA analysis
    for alpha in [0.1, 0.3]:
        key = f"alpha_{alpha}"
        results[key] = {}

        best_final_loss = float("inf")
        best_final_name = ""
        best_mean_dloss = float("inf")
        best_dloss_name = ""

        for name, data in probes.items():
            loss = [d["loss"] for d in data]
            ema_loss = ema(loss, alpha)
            dloss = finite_differences(ema_loss)

            probe_key = probe_label(name)
            results[key][probe_key] = {
                "ema_final_loss": ema_loss[-1],
                "mean_d_loss": sum(dloss) / len(dloss),
                "final_d_loss": dloss[-1],
                "ema_min_loss": min(ema_loss),
                "ema_min_loss_step": ema_loss.index(min(ema_loss)),
            }

            if ema_loss[-1] < best_final_loss:
                best_final_loss = ema_loss[-1]
                best_final_name = probe_key
            if sum(dloss) / len(dloss) < best_mean_dloss:
                best_mean_dloss = sum(dloss) / len(dloss)
                best_dloss_name = probe_key

        if best_final_name == best_dloss_name:
            results[key]["down_and_left_winner"] = f"{best_final_name} (unambiguous: lowest loss, fastest descent)"
        else:
            results[key]["down_and_left_winner"] = (
                f"split: {best_final_name} has lowest final EMA loss ({best_final_loss:.4f}), "
                f"{best_dloss_name} has fastest mean descent ({best_mean_dloss:.6f})"
            )

    # Grad norm stats
    results["grad_norm"] = {}
    for name, data in probes.items():
        gn = [d["grad_norm"] for d in data]
        results["grad_norm"][probe_label(name)] = {
            "max": max(gn),
            "max_step": gn.index(max(gn)),
            "mean": sum(gn) / len(gn),
            "final": gn[-1],
        }

    # Pre-clip grad norm (v2 only)
    first_data = next(iter(probes.values()))
    if "pre_clip_grad_norm" in first_data[0]:
        results["pre_clip_grad_norm"] = {}
        for name, data in probes.items():
            gn = [d["pre_clip_grad_norm"] for d in data]
            results["pre_clip_grad_norm"][probe_label(name)] = {
                "max": max(gn),
                "max_step": gn.index(max(gn)),
                "mean": sum(gn) / len(gn),
                "final": gn[-1],
            }

    # Step time stats
    results["step_time"] = {}
    for name, data in probes.items():
        time = [d["time_s"] for d in data]
        entry = {
            "mean": sum(time) / len(time),
            "max": max(time),
            "max_step": time.index(max(time)),
            "total": sum(time),
        }
        # Compute slowdown ratio if we have enough steps
        if len(time) > 20:
            baseline_mean = sum(time[1:20]) / 19  # skip step 0 (compile warmup)
            if baseline_mean > 0:
                entry["slowdown_ratio_at_max"] = max(time) / baseline_mean
        results["step_time"][probe_label(name)] = entry

    # Accuracy stats
    results["accuracy"] = {}
    for name, data in probes.items():
        acc_p = [d["accuracy_pinkify"] for d in data]
        acc_t = [d["accuracy_thisnotthat"] for d in data]
        n_last = min(20, len(acc_p))
        results["accuracy"][probe_label(name)] = {
            "pinkify_mean": sum(acc_p) / len(acc_p),
            "thisnotthat_mean": sum(acc_t) / len(acc_t),
            f"pinkify_last{n_last}_mean": sum(acc_p[-n_last:]) / n_last,
            f"thisnotthat_last{n_last}_mean": sum(acc_t[-n_last:]) / n_last,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _generate_all_plots(
    probes: dict[str, list[dict]],
    output_dir: Path,
    probe_names: list[str],
    suffix: str = "",
    label_extra: str = "",
):
    """Generate the full suite of plots for a probe set.

    Args:
        suffix: appended to each filename (e.g. "_zoomed")
        label_extra: appended to each title (e.g. " [excl lr=1e-02]")
    """
    # 1. Loss vs step
    print(f"\nPlotting loss vs step{suffix}...")
    chart = PILChart(width=1000, height=650)
    chart.set_title(f"Loss vs Step (r_theta LR sweep){label_extra}")
    chart.set_labels("Step", "Loss")
    for i, (name, data) in enumerate(probes.items()):
        solid, light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        loss = [d["loss"] for d in data]
        ema_01 = ema(loss, 0.1)
        ema_03 = ema(loss, 0.3)
        chart.add_line(steps, loss, color=light, label="", line_width=1)
        chart.add_line(steps, ema_01, color=solid, label=f"{probe_label(name)} EMA(0.1)", line_width=2)
        chart.add_line(steps, ema_03, color=solid, label=f"{probe_label(name)} EMA(0.3)", line_width=2, style="dashed")
    chart.save(output_dir / f"loss_vs_step{suffix}.png")

    # 1b. Log-scale loss vs step
    print(f"Plotting log-scale loss vs step{suffix}...")
    chart_log = PILChart(width=1000, height=650)
    chart_log.set_title(f"log10(Loss) vs Step (r_theta LR sweep){label_extra}")
    chart_log.set_labels("Step", "log10(Loss)")
    for i, (name, data) in enumerate(probes.items()):
        solid, light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        loss = [d["loss"] for d in data]
        log_loss = [math.log10(max(v, 1e-6)) for v in loss]
        log_ema_01 = [math.log10(max(v, 1e-6)) for v in ema(loss, 0.1)]
        chart_log.add_line(steps, log_loss, color=light, label="", line_width=1)
        chart_log.add_line(steps, log_ema_01, color=solid, label=f"{probe_label(name)} EMA(0.1)", line_width=2)
    chart_log.save(output_dir / f"loss_vs_step_logscale{suffix}.png")

    # 2. Grad norm vs step
    print(f"Plotting grad norm vs step{suffix}...")
    chart = PILChart(width=1000, height=650)
    chart.set_title(f"Grad Norm vs Step (r_theta LR sweep){label_extra}")
    chart.set_labels("Step", "Grad Norm")
    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        gn = [d["grad_norm"] for d in data]
        chart.add_line(steps, gn, color=solid, label=probe_label(name), line_width=2)
    chart.save(output_dir / f"grad_norm_vs_step{suffix}.png")

    # 3. Pre-clip grad norm vs step (v2 only)
    first_data = next(iter(probes.values()))
    if "pre_clip_grad_norm" in first_data[0]:
        print(f"Plotting pre-clip grad norm vs step{suffix}...")
        chart = PILChart(width=1000, height=650)
        chart.set_title(f"Pre-Clip Grad Norm vs Step (r_theta LR sweep){label_extra}")
        chart.set_labels("Step", "Pre-Clip Grad Norm")
        for i, (name, data) in enumerate(probes.items()):
            solid, _light, _dark = get_colors(i)
            steps = [d["step"] for d in data]
            gn = [d["pre_clip_grad_norm"] for d in data]
            chart.add_line(steps, gn, color=solid, label=probe_label(name), line_width=2)
        chart.save(output_dir / f"pre_clip_grad_norm_vs_step{suffix}.png")

        # 3b. Log-scale pre-clip grad norm
        print(f"Plotting log-scale pre-clip grad norm vs step{suffix}...")
        chart_log = PILChart(width=1000, height=650)
        chart_log.set_title(f"log10(Pre-Clip Grad Norm) vs Step{label_extra}")
        chart_log.set_labels("Step", "log10(Pre-Clip Grad Norm)")
        for i, (name, data) in enumerate(probes.items()):
            solid, _light, _dark = get_colors(i)
            steps = [d["step"] for d in data]
            gn = [math.log10(max(d["pre_clip_grad_norm"], 1e-10)) for d in data]
            chart_log.add_line(steps, gn, color=solid, label=probe_label(name), line_width=2)
        chart_log.save(output_dir / f"pre_clip_grad_norm_vs_step_logscale{suffix}.png")

    # 4. Step time vs step
    print(f"Plotting step time vs step{suffix}...")
    chart = PILChart(width=1000, height=650)
    chart.set_title(f"Step Time vs Step (r_theta LR sweep){label_extra}")
    chart.set_labels("Step", "Time (s)")
    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        time = [d["time_s"] for d in data]
        chart.add_line(steps, time, color=solid, label=probe_label(name), line_width=2)
    chart.save(output_dir / f"time_vs_step{suffix}.png")

    # 5. BT loss (per-term normalized) vs step (v2 only)
    if "loss" in first_data[0] or "bt_loss" in first_data[0]:
        print(f"Plotting BT loss vs step{suffix}...")
        chart = PILChart(width=1000, height=650)
        chart.set_title(f"BT Loss (per-term avg) vs Step (r_theta LR sweep){label_extra}")
        chart.set_labels("Step", "Loss (per-term avg)")
        for i, (name, data) in enumerate(probes.items()):
            solid, light, _dark = get_colors(i)
            steps = [d["step"] for d in data]
            bt = [d.get("loss", d.get("bt_loss", 0.0)) for d in data]
            ema_bt = ema(bt, 0.1)
            chart.add_line(steps, bt, color=light, label="", line_width=1)
            chart.add_line(steps, ema_bt, color=solid, label=f"{probe_label(name)} EMA(0.1)", line_width=2)
        chart.save(output_dir / f"bt_loss_vs_step{suffix}.png")

    # 6. Accuracy vs step
    print(f"Plotting accuracy vs step{suffix}...")
    chart_p = PILChart(width=1000, height=650)
    chart_p.set_title(f"Accuracy (pinkify) vs Step{label_extra}")
    chart_p.set_labels("Step", "Accuracy (pinkify)")
    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        acc = [d["accuracy_pinkify"] for d in data]
        ema_acc = ema(acc, 0.1)
        chart_p.add_line(steps, ema_acc, color=solid, label=probe_label(name), line_width=2)
    chart_p.save(output_dir / f"accuracy_pinkify_vs_step{suffix}.png")

    chart_t = PILChart(width=1000, height=650)
    chart_t.set_title(f"Accuracy (thisnotthat) vs Step{label_extra}")
    chart_t.set_labels("Step", "Accuracy (thisnotthat)")
    for i, (name, data) in enumerate(probes.items()):
        solid, _light, _dark = get_colors(i)
        steps = [d["step"] for d in data]
        acc = [d["accuracy_thisnotthat"] for d in data]
        ema_acc = ema(acc, 0.1)
        chart_t.add_line(steps, ema_acc, color=solid, label=probe_label(name), line_width=2)
    chart_t.save(output_dir / f"accuracy_thisnotthat_vs_step{suffix}.png")

    # 7. Down-and-to-the-left plots
    for alpha in [0.1, 0.3]:
        alpha_str = f"alpha{alpha:.1f}".replace(".", "")
        print(f"Plotting down-and-to-the-left (alpha={alpha}){suffix}...")
        chart = PILChart(width=1000, height=700)
        chart.set_title(f"Down-and-to-the-Left: Loss vs Descent Rate (EMA alpha={alpha}){label_extra}")
        chart.set_labels("EMA Loss (lower = better)", "d(loss)/d(step) (more negative = faster)")
        for i, (name, data) in enumerate(probes.items()):
            solid, light, dark = get_colors(i)
            loss = [d["loss"] for d in data]
            ema_loss = ema(loss, alpha)
            dloss = finite_differences(ema_loss)
            x_vals = ema_loss[:-1]
            chart.add_line(x_vals, dloss, color=light, label="", line_width=1)
            chart.add_scatter(x_vals, dloss, color=solid, label=probe_label(name), size=3)
            chart.add_scatter([x_vals[-1]], [dloss[-1]], color=dark, label=f"{probe_label(name)} final", size=6)
        chart.save(output_dir / f"down_and_left_{alpha_str}{suffix}.png")


def main():
    parser = argparse.ArgumentParser(description="Plot r_theta LR sweep training curves")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=REPO_ROOT / "rtheta_sweep_output",
        help="Directory containing lr_* probe subdirectories (default: rtheta_sweep_output)",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    output_dir = sweep_dir / "plots"

    print(f"Sweep directory: {sweep_dir}")
    probe_names = discover_probes(sweep_dir)
    if not probe_names:
        print(f"ERROR: No probe directories (lr_*) found in {sweep_dir}")
        sys.exit(1)

    print(f"Discovered {len(probe_names)} probes: {', '.join(probe_names)}")

    # Load all curves (ordered dict preserving discovery order = descending LR)
    probes: dict[str, list[dict]] = {}
    for name in probe_names:
        data = load_curve(sweep_dir, name)
        probes[name] = data
        print(f"  {name}: {len(data)} steps")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: All probes (original linear-scale plots preserved) ---
    print("\n" + "=" * 60)
    print("PASS 1: All probes (original)")
    print("=" * 60)
    _generate_all_plots(probes, output_dir, probe_names)

    # --- Pass 2: Zoomed (excluding highest-LR outlier) ---
    if len(probe_names) > 2:
        # Exclude the first probe (highest LR, most likely to be the outlier)
        excl_name = probe_names[0]
        zoomed_probes = {k: v for k, v in probes.items() if k != excl_name}
        zoomed_names = [n for n in probe_names if n != excl_name]
        print("\n" + "=" * 60)
        print(f"PASS 2: Zoomed (excluding {excl_name})")
        print("=" * 60)
        _generate_all_plots(
            zoomed_probes, output_dir, zoomed_names,
            suffix="_zoomed",
            label_extra=f" [excl {probe_label(excl_name)}]",
        )

    # --- Summary JSON (uses all probes) ---
    print("\nComputing summary statistics...")
    summary = compute_summary(probes)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Print key results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for alpha_key in ["alpha_0.1", "alpha_0.3"]:
        s = summary[alpha_key]
        print(f"\n  {alpha_key}:")
        for pname in probe_names:
            plabel = probe_label(pname)
            if plabel in s:
                print(f"    {plabel}: final EMA loss = {s[plabel]['ema_final_loss']:.4f}, "
                      f"mean d_loss = {s[plabel]['mean_d_loss']:.6f}")
        if "down_and_left_winner" in s:
            print(f"    Winner: {s['down_and_left_winner']}")

    print(f"\n  Grad norms:")
    for pname in probe_names:
        plabel = probe_label(pname)
        gn = summary["grad_norm"][plabel]
        print(f"    {plabel}: max={gn['max']:.1f} (step {gn['max_step']}), mean={gn['mean']:.4f}")

    if "pre_clip_grad_norm" in summary:
        print(f"\n  Pre-clip grad norms:")
        for pname in probe_names:
            plabel = probe_label(pname)
            gn = summary["pre_clip_grad_norm"][plabel]
            print(f"    {plabel}: max={gn['max']:.1f} (step {gn['max_step']}), mean={gn['mean']:.1f}")

    print(f"\n  Step time:")
    for pname in probe_names:
        plabel = probe_label(pname)
        st = summary["step_time"][plabel]
        extra = ""
        if "slowdown_ratio_at_max" in st:
            extra = f", slowdown at peak={st['slowdown_ratio_at_max']:.1f}x"
        print(f"    {plabel}: mean={st['mean']:.1f}s, total={st['total']:.0f}s{extra}")

    print(f"\n  Accuracy (last 20 steps):")
    for pname in probe_names:
        plabel = probe_label(pname)
        acc = summary["accuracy"][plabel]
        # Find the last-N keys
        pink_key = [k for k in acc if k.startswith("pinkify_last")][0]
        tnt_key = [k for k in acc if k.startswith("thisnotthat_last")][0]
        print(f"    {plabel}: pinkify={acc[pink_key]:.0%}, thisnotthat={acc[tnt_key]:.0%}")

    print("\nDone.")


if __name__ == "__main__":
    main()
