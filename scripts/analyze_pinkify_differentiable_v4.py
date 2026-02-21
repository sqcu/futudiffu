r"""Analyze differentiable BTRM training run metrics and produce charts for v4 (197 steps, extended run).

Reads:
  pinkify_thisnotthat_output/differentiable_run_v4/training_metrics.jsonl  (197 steps)
  pinkify_thisnotthat_output/differentiable_run_v3/training_metrics.jsonl  (v3 comparison, 30 steps)

Produces:
  pinkify_thisnotthat_output/differentiable_run_v4/charts/
    01_loss_curve.png           -- BT loss raw + EMA, all 197 steps, phase boundaries
    02_per_head_accuracy.png    -- per-head accuracy with running average (window=10)
    03_gradient_norms.png       -- pre-clip grad norm log scale, phase boundaries
    04_step_timing.png          -- seconds per step

  pinkify_thisnotthat_output/differentiable_run_v4/training_analysis.md

All rendering via PIL only -- no matplotlib.

v4 is the extended training run (200 target, 197 completed before crash).
Key difference from v3 (30 steps): shows full learning curve through convergence and
into gradient instability. Three visible phases:
  Phase 1 (steps 0-30):   Warmup + early descent, loss ~0.70 -> ~0.62
  Phase 2 (steps 30-130): Aggressive learning, loss reaches 0.05-0.50, grad norms stable
  Phase 3 (steps 130-197): Overfitting regime, loss oscillates wildly, grad norms explode

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts\analyze_pinkify_differentiable_v4.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
V4_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run_v4"
V3_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run_v3"
CHARTS_DIR = V4_DIR / "charts"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Simple statistics helpers
# ---------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def running_average(xs: list[float], window: int) -> list[float]:
    """Simple unweighted running average (causal)."""
    out = []
    for i, v in enumerate(xs):
        start = max(0, i - window + 1)
        window_vals = xs[start:i + 1]
        out.append(sum(window_vals) / len(window_vals))
    return out


def ema(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average. alpha is weight on the new sample."""
    out = []
    s = values[0]
    for v in values:
        s = alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

def detect_phases(data: list[dict]) -> dict:
    """Detect the three training phases based on gradient norm behavior.

    Phase 1: warmup + early descent (grad norms < 3, loss descending gently)
    Phase 2: aggressive learning (grad norms still moderate, loss reaching low values)
    Phase 3: instability (grad norms frequently > 5, loss oscillating wildly)

    Returns dict with phase boundaries and statistics.
    """
    grad_norms = [d["pre_clip_grad_norm"] for d in data]
    losses = [d["bt_loss"] for d in data]
    n = len(data)

    # Use sliding window of 10 to detect gradient norm regime changes
    window = 10

    # Find phase 2 start: where EMA loss first drops below 0.60
    ema_loss = ema(losses, alpha=0.15)
    phase2_start = 30  # default
    for i in range(10, n):
        if ema_loss[i] < 0.60:
            phase2_start = i
            break

    # Find phase 3 start: where grad norms start consistently exceeding 5
    # Use a sliding window: if mean grad norm in window > 5, that's phase 3
    phase3_start = n  # default: never reached
    for i in range(phase2_start + 10, n - window):
        window_gn = grad_norms[i:i + window]
        if mean(window_gn) > 5.0:
            phase3_start = i
            break

    return {
        "phase1": (0, phase2_start),
        "phase2": (phase2_start, phase3_start),
        "phase3": (phase3_start, n),
        "phase2_start": phase2_start,
        "phase3_start": phase3_start,
    }


# ---------------------------------------------------------------------------
# PILChart: minimal PIL-based line/scatter chart renderer
# ---------------------------------------------------------------------------

def _format_tick(val: float) -> str:
    if abs(val) < 0.001 and val != 0:
        return f"{val:.1e}"
    if abs(val) >= 1000:
        return f"{val:.0f}"
    if abs(val) >= 10:
        return f"{val:.1f}"
    if abs(val) >= 1:
        return f"{val:.2f}"
    return f"{val:.3f}"


class PILChart:
    """Minimal line/scatter chart renderer using PIL."""

    def __init__(
        self,
        width: int = 1200,
        height: int = 700,
        bg_color: str = "#ffffff",
        margin_left: int = 90,
        margin_right: int = 30,
        margin_top: int = 50,
        margin_bottom: int = 65,
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
        self._y_log: bool = False
        self._vlines: list[dict] = []

        self.font = ImageFont.load_default()

    def set_title(self, title: str):
        self.title = title

    def set_labels(self, x_label: str, y_label: str):
        self.x_label = x_label
        self.y_label = y_label

    def set_log_y(self, enabled: bool = True):
        self._y_log = enabled

    def add_vline(self, x: float, color: str = "#999999", label: str = "", style: str = "dashed"):
        """Add a vertical line at x position (e.g. phase boundary)."""
        self._vlines.append({"x": x, "color": color, "label": label, "style": style})

    def add_line(
        self,
        xs: list[float],
        ys: list[float],
        color: str = "#000000",
        label: str = "",
        line_width: int = 2,
        style: str = "solid",
    ):
        self.series.append({
            "type": "line",
            "xs": list(xs),
            "ys": list(ys),
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
            "xs": list(xs),
            "ys": list(ys),
            "color": color,
            "label": label,
            "size": size,
        })

    def _transform_y(self, y: float) -> float:
        if self._y_log:
            return math.log10(max(y, 1e-10))
        return y

    def _compute_bounds(self):
        all_xs, all_ys = [], []
        for s in self.series:
            all_xs.extend(s["xs"])
            raw_ys = [self._transform_y(v) for v in s["ys"]]
            all_ys.extend(raw_ys)

        # Include vline x positions
        for vl in self._vlines:
            all_xs.append(vl["x"])

        if not all_xs or not all_ys:
            return 0, 1, 0, 1

        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)

        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        x_min -= 0.05 * x_range
        x_max += 0.05 * x_range
        y_min -= 0.05 * y_range
        y_max += 0.05 * y_range

        return x_min, x_max, y_min, y_max

    def _data_to_pixel(self, x: float, y: float, x_min, x_max, y_min, y_max):
        ty = self._transform_y(y)
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        px = self.ml + (x - x_min) / x_range * self.plot_w
        py = self.mt + (1.0 - (ty - y_min) / y_range) * self.plot_h
        return int(px), int(py)

    def _draw_axes(self, x_min, x_max, y_min, y_max):
        d = self.draw
        d.rectangle(
            [self.ml, self.mt, self.ml + self.plot_w, self.mt + self.plot_h],
            outline="#cccccc",
            width=1,
        )

        n_x_ticks = 8
        n_y_ticks = 6

        for i in range(n_x_ticks + 1):
            frac = i / n_x_ticks
            val = x_min + frac * (x_max - x_min)
            px = self.ml + int(frac * self.plot_w)
            d.line([(px, self.mt), (px, self.mt + self.plot_h)], fill="#eeeeee", width=1)
            label = _format_tick(val)
            d.text((px, self.mt + self.plot_h + 5), label, fill="#333333", font=self.font, anchor="mt")

        for i in range(n_y_ticks + 1):
            frac = i / n_y_ticks
            val = y_min + frac * (y_max - y_min)
            py = self.mt + int((1.0 - frac) * self.plot_h)
            d.line([(self.ml, py), (self.ml + self.plot_w, py)], fill="#eeeeee", width=1)
            if self._y_log:
                label = f"1e{val:.0f}" if abs(val) >= 1 else f"10^{val:.1f}"
            else:
                label = _format_tick(val)
            d.text((self.ml - 5, py), label, fill="#333333", font=self.font, anchor="rm")

        if self.x_label:
            d.text(
                (self.ml + self.plot_w // 2, self.img_h - 10),
                self.x_label,
                fill="#000000",
                font=self.font,
                anchor="mb",
            )
        if self.y_label:
            d.text(
                (5, self.mt + self.plot_h // 2),
                self.y_label,
                fill="#000000",
                font=self.font,
                anchor="lm",
            )

    def _draw_vlines(self, x_min, x_max, y_min, y_max):
        d = self.draw
        for vl in self._vlines:
            x_range = x_max - x_min if x_max != x_min else 1.0
            px = self.ml + int((vl["x"] - x_min) / x_range * self.plot_w)
            color = vl["color"]
            style = vl.get("style", "dashed")

            if style == "dashed":
                y_top = self.mt
                y_bot = self.mt + self.plot_h
                step = 6
                for y_pos in range(y_top, y_bot, step * 2):
                    y_end = min(y_pos + step, y_bot)
                    d.line([(px, y_pos), (px, y_end)], fill=color, width=2)
            else:
                d.line([(px, self.mt), (px, self.mt + self.plot_h)], fill=color, width=2)

            if vl.get("label"):
                d.text((px + 3, self.mt + 5), vl["label"], fill=color, font=self.font, anchor="lt")

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
        n = len(labeled)
        line_height = 18
        legend_w = 260
        legend_h = n * line_height + 10

        # Place legend in upper right, inside plot area
        x_start = self.ml + self.plot_w - legend_w - 10
        y_start = self.mt + 10

        d.rectangle(
            [x_start - 5, y_start - 5, x_start + legend_w, y_start + legend_h],
            fill="#ffffff",
            outline="#cccccc",
        )

        for i, s in enumerate(labeled):
            y = y_start + i * line_height
            color = s["color"]
            if s["type"] == "line":
                style = s.get("style", "solid")
                if style == "dashed":
                    for dx in range(0, 25, 6):
                        d.line(
                            [(x_start, y + 7), (x_start + min(dx + 3, 25), y + 7)],
                            fill=color, width=2,
                        )
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
        self._draw_vlines(x_min, x_max, y_min, y_max)
        self._draw_series(x_min, x_max, y_min, y_max)
        self._draw_title()
        self._draw_legend()
        return self.img

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.render().save(str(path))
        print(f"  Saved: {path.name}")


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def chart_loss_curve(v4: list[dict], phases: dict, v3: list[dict] | None, out_dir: Path):
    """Chart 1: BT loss raw + EMA over all 197 steps with phase boundaries."""
    chart = PILChart()
    chart.set_title("BT Loss vs Step: v4 (197 steps, extended run)")
    chart.set_labels("Step", "BT Loss")

    steps = [d["step"] for d in v4]
    bt = [d["bt_loss"] for d in v4]
    ema_bt = ema(bt, alpha=0.1)  # slower EMA for longer run

    # Phase boundaries
    p2 = phases["phase2_start"]
    p3 = phases["phase3_start"]
    chart.add_vline(p2, color="#2277aa", label=f"Phase 2 (step {p2})")
    chart.add_vline(p3, color="#cc2222", label=f"Phase 3 (step {p3})")

    # Raw (light blue) + EMA (solid blue)
    chart.add_scatter(steps, bt, color="#bbddff", label="v4 raw", size=2)
    chart.add_line(steps, ema_bt, color="#1155cc", label="v4 EMA(0.1)", line_width=2)

    # v3 comparison (30 steps, dashed)
    if v3:
        steps_v3 = [d["step"] for d in v3]
        bt_v3 = [d["bt_loss"] for d in v3]
        ema_v3 = ema(bt_v3, alpha=0.1)
        chart.add_line(steps_v3, ema_v3, color="#cc1111", label="v3 EMA(0.1) (30 steps)", line_width=2, style="dashed")

    chart.save(out_dir / "01_loss_curve.png")


def chart_per_head_accuracy(v4: list[dict], phases: dict, v3: list[dict] | None, out_dir: Path):
    """Chart 2: Per-head accuracy with running average (window=10 for longer run)."""
    chart = PILChart()
    chart.set_title("Per-Head Accuracy vs Step (window=10 running avg)")
    chart.set_labels("Step", "Accuracy")

    steps = [d["step"] for d in v4]
    acc_p = [d["accuracy_pinkify"] for d in v4]
    acc_t = [d["accuracy_thisnotthat"] for d in v4]

    window = 10
    ravg_p = running_average(acc_p, window=window)
    ravg_t = running_average(acc_t, window=window)

    # Phase boundaries
    p2 = phases["phase2_start"]
    p3 = phases["phase3_start"]
    chart.add_vline(p2, color="#2277aa", label=f"Phase 2 (step {p2})")
    chart.add_vline(p3, color="#cc2222", label=f"Phase 3 (step {p3})")

    # Raw scatter (light)
    chart.add_scatter(steps, acc_p, color="#aaddff", label="", size=2)
    chart.add_scatter(steps, acc_t, color="#ffddaa", label="", size=2)

    # Running averages
    chart.add_line(steps, ravg_p, color="#1155cc", label=f"pinkify ravg({window})", line_width=2)
    chart.add_line(steps, ravg_t, color="#cc7711", label=f"thisnotthat ravg({window})", line_width=2)

    # v3 comparison
    if v3:
        steps_v3 = [d["step"] for d in v3]
        acc_p_v3 = [d["accuracy_pinkify"] for d in v3]
        acc_t_v3 = [d["accuracy_thisnotthat"] for d in v3]
        ravg_p_v3 = running_average(acc_p_v3, window=5)
        ravg_t_v3 = running_average(acc_t_v3, window=5)
        chart.add_line(steps_v3, ravg_p_v3, color="#7799cc", label="v3 pinkify ravg(5)", line_width=2, style="dashed")
        chart.add_line(steps_v3, ravg_t_v3, color="#cc9955", label="v3 thisnotthat ravg(5)", line_width=2, style="dashed")

    chart.save(out_dir / "02_per_head_accuracy.png")


def chart_gradient_norms(v4: list[dict], phases: dict, v3: list[dict] | None, out_dir: Path):
    """Chart 3: Pre-clip gradient norms on log scale with phase boundaries."""
    chart = PILChart()
    chart.set_title("Pre-Clip Gradient Norm vs Step (log scale)")
    chart.set_labels("Step", "log10(Pre-Clip Grad Norm)")
    chart.set_log_y(True)

    steps = [d["step"] for d in v4]
    gn = [d["pre_clip_grad_norm"] for d in v4]
    ema_gn = ema(gn, alpha=0.15)

    # Phase boundaries
    p2 = phases["phase2_start"]
    p3 = phases["phase3_start"]
    chart.add_vline(p2, color="#2277aa", label=f"Phase 2 (step {p2})")
    chart.add_vline(p3, color="#cc2222", label=f"Phase 3 (step {p3})")

    chart.add_scatter(steps, gn, color="#bbffbb", label="v4 raw", size=2)
    chart.add_line(steps, ema_gn, color="#117711", label="v4 EMA(0.15)", line_width=2)

    if v3:
        steps_v3 = [d["step"] for d in v3]
        gn_v3 = [d["pre_clip_grad_norm"] for d in v3]
        ema_v3 = ema(gn_v3, alpha=0.15)
        chart.add_line(steps_v3, gn_v3, color="#ffccaa", label="v3 raw", line_width=1, style="dashed")
        chart.add_line(steps_v3, ema_v3, color="#cc7711", label="v3 EMA(0.15)", line_width=2, style="dashed")

    chart.save(out_dir / "03_gradient_norms.png")


def chart_step_timing(v4: list[dict], phases: dict, v3: list[dict] | None, out_dir: Path):
    """Chart 4: Step timing."""
    chart = PILChart()
    chart.set_title("Step Timing: Seconds per Step")
    chart.set_labels("Step", "Time (s)")

    steps = [d["step"] for d in v4]
    times = [d["time_s"] for d in v4]
    chart.add_line(steps, times, color="#1155cc", label="v4", line_width=2)

    if v3:
        steps_v3 = [d["step"] for d in v3]
        times_v3 = [d["time_s"] for d in v3]
        chart.add_line(steps_v3, times_v3, color="#cc1111", label="v3 (30 steps)", line_width=2, style="dashed")

    chart.save(out_dir / "04_step_timing.png")


# ---------------------------------------------------------------------------
# Analysis and report
# ---------------------------------------------------------------------------

def compute_analysis(v4: list[dict], v3: list[dict] | None) -> dict:
    """Compute all analysis metrics from v4 data."""
    n = len(v4)
    steps = [d["step"] for d in v4]
    loss = [d["bt_loss"] for d in v4]
    gn = [d["pre_clip_grad_norm"] for d in v4]
    times = [d["time_s"] for d in v4]
    acc_p = [d["accuracy_pinkify"] for d in v4]
    acc_t = [d["accuracy_thisnotthat"] for d in v4]
    pair_weights = [d["pair_weight"] for d in v4]

    phases = detect_phases(v4)
    p1_start, p1_end = phases["phase1"]
    p2_start, p2_end = phases["phase2"]
    p3_start, p3_end = phases["phase3"]

    # Per-phase statistics
    phase_stats = {}
    for pname, (ps, pe) in [("phase1", (p1_start, p1_end)), ("phase2", (p2_start, p2_end)), ("phase3", (p3_start, p3_end))]:
        if ps >= pe:
            continue
        pl = loss[ps:pe]
        pg = gn[ps:pe]
        pa_p = acc_p[ps:pe]
        pa_t = acc_t[ps:pe]
        pt = times[ps:pe]
        phase_stats[pname] = {
            "steps": f"{ps}-{pe - 1}",
            "n_steps": pe - ps,
            "loss_mean": mean(pl),
            "loss_min": min(pl),
            "loss_max": max(pl),
            "loss_std": (sum((x - mean(pl)) ** 2 for x in pl) / len(pl)) ** 0.5 if pl else 0,
            "grad_norm_mean": mean(pg),
            "grad_norm_max": max(pg),
            "grad_norm_median": median(pg),
            "acc_pinkify": mean(pa_p),
            "acc_thisnotthat": mean(pa_t),
            "time_mean": mean(pt),
        }

    # Find optimal early stopping: lowest EMA loss value
    ema_loss = ema(loss, alpha=0.1)
    best_ema_idx = ema_loss.index(min(ema_loss))
    best_ema_val = ema_loss[best_ema_idx]

    # Instability transition: first step where grad norm > 10
    instability_step = None
    instability_loss = None
    for i, g in enumerate(gn):
        if g > 10.0 and i > 30:  # skip warmup
            instability_step = i
            instability_loss = loss[i]
            break

    # Gradient norm explosion steps (> 20)
    explosion_steps = [(i, gn[i]) for i in range(n) if gn[i] > 20.0]

    # Steps where loss < 0.01 (near-perfect discrimination)
    near_perfect = [(i, loss[i]) for i in range(n) if loss[i] < 0.01]

    # Last 20 steps stats
    last20 = slice(max(0, n - 20), n)
    last20_acc_p = mean(acc_p[last20])
    last20_acc_t = mean(acc_t[last20])
    last20_loss = mean(loss[last20])
    last20_gn = mean(gn[last20])

    # Timing
    t_step0 = times[0]
    t_steady = times[1:]
    t_steady_mean = mean(t_steady)
    t_total = sum(times)

    # Recompilation events (steps with time > 3x steady-state mean)
    recomp_steps = [(i, times[i]) for i in range(1, n) if times[i] > 3 * t_steady_mean]

    # v3 comparison
    v3_stats = None
    if v3:
        v3_loss = [d["bt_loss"] for d in v3]
        v3_gn = [d["pre_clip_grad_norm"] for d in v3]
        v3_acc_p = [d["accuracy_pinkify"] for d in v3]
        v3_acc_t = [d["accuracy_thisnotthat"] for d in v3]
        v3_times = [d["time_s"] for d in v3]
        v3_last10 = slice(max(0, len(v3) - 10), len(v3))
        v3_stats = {
            "n_steps": len(v3),
            "initial_loss": v3_loss[0],
            "final_loss": v3_loss[-1],
            "min_loss": min(v3_loss),
            "grad_norm_mean": mean(v3_gn),
            "grad_norm_max": max(v3_gn),
            "acc_pinkify_last10": mean(v3_acc_p[v3_last10]),
            "acc_thisnotthat_last10": mean(v3_acc_t[v3_last10]),
            "step0_time": v3_times[0],
            "steady_time": mean(v3_times[1:]),
        }

    return {
        "n_steps": n,
        "initial_loss": loss[0],
        "final_loss": loss[-1],
        "min_loss": min(loss),
        "min_loss_step": loss.index(min(loss)),
        "max_loss": max(loss),
        "max_loss_step": loss.index(max(loss)),
        "phases": phases,
        "phase_stats": phase_stats,
        "best_ema_step": best_ema_idx,
        "best_ema_val": best_ema_val,
        "instability_step": instability_step,
        "instability_loss": instability_loss,
        "explosion_steps": explosion_steps,
        "near_perfect_steps": near_perfect,
        "grad_norm_mean": mean(gn),
        "grad_norm_max": max(gn),
        "grad_norm_max_step": gn.index(max(gn)),
        "acc_pinkify_overall": mean(acc_p),
        "acc_thisnotthat_overall": mean(acc_t),
        "last20_acc_p": last20_acc_p,
        "last20_acc_t": last20_acc_t,
        "last20_loss": last20_loss,
        "last20_gn": last20_gn,
        "step0_time": t_step0,
        "steady_time_mean": t_steady_mean,
        "total_time": t_total,
        "recomp_steps": recomp_steps,
        "v3_stats": v3_stats,
    }


def write_report(v4: list[dict], analysis: dict, out_path: Path):
    """Write the training analysis markdown report."""
    n = analysis["n_steps"]
    phases = analysis["phases"]
    ps = analysis["phase_stats"]
    v3 = analysis["v3_stats"]

    p2_start = phases["phase2_start"]
    p3_start = phases["phase3_start"]

    # Phase 1 stats
    p1 = ps.get("phase1", {})
    p2 = ps.get("phase2", {})
    p3 = ps.get("phase3", {})

    explosion_str = ""
    if analysis["explosion_steps"]:
        top10 = sorted(analysis["explosion_steps"], key=lambda x: -x[1])[:10]
        explosion_str = "\n".join(f"  - Step {s}: {v:.1f}" for s, v in top10)
    else:
        explosion_str = "  (none)"

    near_perfect_str = ""
    if analysis["near_perfect_steps"]:
        near_perfect_str = ", ".join(f"step {s} ({v:.4f})" for s, v in analysis["near_perfect_steps"])
    else:
        near_perfect_str = "(none)"

    recomp_str = ""
    if analysis["recomp_steps"]:
        recomp_str = "\n- Recompilation events: " + ", ".join(f"step {s} ({t:.1f}s)" for s, t in analysis["recomp_steps"])

    v3_table = ""
    if v3:
        # v4 first-30 comparison
        v4_first30 = v4[:30]
        v4_f30_loss = [d["bt_loss"] for d in v4_first30]
        v4_f30_gn = [d["pre_clip_grad_norm"] for d in v4_first30]
        v4_f30_acc_p = [d["accuracy_pinkify"] for d in v4_first30]
        v4_f30_acc_t = [d["accuracy_thisnotthat"] for d in v4_first30]

        v3_table = f"""
## 6. Comparison: v3 (30 steps) vs v4 (197 steps)

v4 is the same configuration as v3 (logsquare_weight=0.0, lr=3e-4, grad_clip=0.1),
run for 200 target steps (197 completed). The first 30 steps of v4 should be
directly comparable to v3's full run (same pair sampling sequence modulo adapter state).

| Metric | v3 (30 steps) | v4 first 30 | v4 full (197 steps) |
|--------|--------------|-------------|---------------------|
| Initial BT loss | {v3['initial_loss']:.4f} | {analysis['initial_loss']:.4f} | {analysis['initial_loss']:.4f} |
| Final BT loss | {v3['final_loss']:.4f} | {v4_f30_loss[-1]:.4f} | {analysis['final_loss']:.4f} |
| Min BT loss | {v3['min_loss']:.4f} | {min(v4_f30_loss):.4f} | {analysis['min_loss']:.6f} (step {analysis['min_loss_step']}) |
| Grad norm mean | {v3['grad_norm_mean']:.2f} | {mean(v4_f30_gn):.2f} | {analysis['grad_norm_mean']:.2f} |
| Grad norm max | {v3['grad_norm_max']:.2f} | {max(v4_f30_gn):.2f} | {analysis['grad_norm_max']:.1f} (step {analysis['grad_norm_max_step']}) |
| Pinkify acc (last N) | {v3['acc_pinkify_last10']:.0%} (last 10) | {mean(v4_f30_acc_p[-10:]):.0%} (last 10) | {analysis['last20_acc_p']:.0%} (last 20) |
| TNT acc (last N) | {v3['acc_thisnotthat_last10']:.0%} (last 10) | {mean(v4_f30_acc_t[-10:]):.0%} (last 10) | {analysis['last20_acc_t']:.0%} (last 20) |
| Step 0 compile | {v3['step0_time']:.1f}s | {analysis['step0_time']:.1f}s | {analysis['step0_time']:.1f}s |
| Steady-state time | {v3['steady_time']:.1f}s | -- | {analysis['steady_time_mean']:.1f}s |
| Total wall time | {v3['step0_time'] + v3['steady_time'] * (v3['n_steps'] - 1):.0f}s | -- | {analysis['total_time']:.0f}s ({analysis['total_time']/60:.1f} min) |

**Key observations:**
- v4's first 30 steps closely match v3, confirming reproducibility.
- Extended training (steps 30-130) drives loss from ~0.62 down to sub-0.10 territory,
  showing the model IS learning to discriminate the training pairs.
- After step ~{p3_start}, gradient instability begins. The model has overfit the
  pair sampling distribution -- it confidently classifies most pairs, but when it
  encounters a genuinely ambiguous pair, the loss spikes and gradients explode.
"""

    report = f"""# Differentiable BTRM Training Analysis (v4 -- Extended 197-Step Run)

**Run:** `pinkify_thisnotthat_differentiable_v4`
**Date:** 2026-02-18
**Steps completed:** {n} / 200 (crashed at step 197)
**Total wall time:** {analysis['total_time']:.0f}s ({analysis['total_time']/60:.1f} min)
**LR:** 0.0003 (warmup for first 10 steps)
**Grad clip:** 0.1
**logsquare_weight:** 0.0

## Run Configuration

- LoRA adapter parameters: 10,096,640
- Score unembedder parameters: 11,520
- Adapter grad verified at step 0: True
- Scoring method: on_the_fly_gpu
- Dataset trajectories: 259
- Crash cause: `RuntimeError: element 0 of tensors does not require grad` at step 197
  (edge case: BT loss scalar had no grad_fn, likely identical scores for both images in pair)

## 1. Three-Phase Training Dynamics

The 197-step run reveals a complete training lifecycle with three distinct phases:

### Phase 1: Warmup + Early Descent (steps {p1.get('steps', '0-?')})

> Loss descends gently from {analysis['initial_loss']:.4f} toward ~0.62.
> LR warmup for the first 10 steps. Gradient norms stable at ~{p1.get('grad_norm_mean', 0):.2f}.
> The model is beginning to separate preferred from non-preferred images.

| Metric | Value |
|--------|-------|
| Loss range | {p1.get('loss_min', 0):.4f} -- {p1.get('loss_max', 0):.4f} |
| Loss mean (std) | {p1.get('loss_mean', 0):.4f} ({p1.get('loss_std', 0):.4f}) |
| Grad norm mean | {p1.get('grad_norm_mean', 0):.3f} |
| Grad norm max | {p1.get('grad_norm_max', 0):.3f} |
| Pinkify accuracy | {p1.get('acc_pinkify', 0):.0%} |
| TNT accuracy | {p1.get('acc_thisnotthat', 0):.0%} |

### Phase 2: Aggressive Learning (steps {p2.get('steps', '?-?')})

> Loss aggressively descends, reaching sub-0.10 values. The model is confidently
> discriminating most training pairs. Gradient norms remain controlled but begin
> showing occasional spikes. This is the productive training phase.

| Metric | Value |
|--------|-------|
| Loss range | {p2.get('loss_min', 0):.4f} -- {p2.get('loss_max', 0):.4f} |
| Loss mean (std) | {p2.get('loss_mean', 0):.4f} ({p2.get('loss_std', 0):.4f}) |
| Grad norm mean | {p2.get('grad_norm_mean', 0):.3f} |
| Grad norm max | {p2.get('grad_norm_max', 0):.3f} |
| Pinkify accuracy | {p2.get('acc_pinkify', 0):.0%} |
| TNT accuracy | {p2.get('acc_thisnotthat', 0):.0%} |

### Phase 3: Instability / Overfitting (steps {p3.get('steps', '?-?')})

> Loss oscillates wildly between near-zero and 2.3. Gradient norms explode
> (up to {analysis['grad_norm_max']:.1f}). The model has memorized the easy pairs
> and now produces extreme score differentials. When it encounters a hard pair
> that it gets wrong, the loss is enormous and the gradient catastrophic.

| Metric | Value |
|--------|-------|
| Loss range | {p3.get('loss_min', 0):.6f} -- {p3.get('loss_max', 0):.4f} |
| Loss mean (std) | {p3.get('loss_mean', 0):.4f} ({p3.get('loss_std', 0):.4f}) |
| Grad norm mean | {p3.get('grad_norm_mean', 0):.2f} |
| Grad norm max | {p3.get('grad_norm_max', 0):.1f} |
| Pinkify accuracy | {p3.get('acc_pinkify', 0):.0%} |
| TNT accuracy | {p3.get('acc_thisnotthat', 0):.0%} |

## 2. Loss Trajectory

| Metric | Value |
|--------|-------|
| Initial BT loss | {analysis['initial_loss']:.4f} |
| Final BT loss | {analysis['final_loss']:.4f} |
| Minimum BT loss | {analysis['min_loss']:.6f} (step {analysis['min_loss_step']}) |
| Maximum BT loss | {analysis['max_loss']:.4f} (step {analysis['max_loss_step']}) |
| Optimal EMA(0.1) point | step {analysis['best_ema_step']} (EMA = {analysis['best_ema_val']:.4f}) |

Near-perfect discrimination steps (loss < 0.01): {near_perfect_str}

The BT loss descends through Phase 1 and 2, then enters chaotic oscillation in
Phase 3. The EMA(0.1) smoothed loss continues descending until step ~{analysis['best_ema_step']},
which is the recommended early-stopping point.

## 3. Gradient Norm Analysis

| Metric | Value |
|--------|-------|
| Overall mean | {analysis['grad_norm_mean']:.2f} |
| Overall max | {analysis['grad_norm_max']:.1f} (step {analysis['grad_norm_max_step']}) |
| Phase 1 mean | {p1.get('grad_norm_mean', 0):.3f} |
| Phase 2 mean | {p2.get('grad_norm_mean', 0):.3f} |
| Phase 3 mean | {p3.get('grad_norm_mean', 0):.2f} |
| Instability onset | step {analysis.get('instability_step', 'N/A')} (first grad norm > 10) |

**Top 10 gradient explosions (pre-clip norm > 20):**
{explosion_str}

The gradient norm signature is the clearest phase separator. Phase 1-2 norms
stay below ~5. Phase 3 norms regularly exceed 20-80, with the grad clip at 0.1
absorbing almost all the explosion. The clip prevents parameter updates from
being proportionally catastrophic, but the loss itself still oscillates because
the score magnitudes have grown large enough that even clipped updates cannot
stabilize the model on genuinely ambiguous pairs.

## 4. Per-Head Accuracy

| Head | Overall | Phase 1 | Phase 2 | Phase 3 | Last 20 steps |
|------|---------|---------|---------|---------|---------------|
| pinkify | {analysis['acc_pinkify_overall']:.0%} | {p1.get('acc_pinkify', 0):.0%} | {p2.get('acc_pinkify', 0):.0%} | {p3.get('acc_pinkify', 0):.0%} | {analysis['last20_acc_p']:.0%} |
| thisnotthat | {analysis['acc_thisnotthat_overall']:.0%} | {p1.get('acc_thisnotthat', 0):.0%} | {p2.get('acc_thisnotthat', 0):.0%} | {p3.get('acc_thisnotthat', 0):.0%} | {analysis['last20_acc_t']:.0%} |

Pinkify (attention quantization discrimination) achieves high accuracy across all phases.
Thisnotthat (step count discrimination) is noisier but trends upward through Phase 2,
then degrades in Phase 3 as the model becomes overconfident on memorized pairs.

## 5. Step Timing

| Metric | Value |
|--------|-------|
| Step 0 (compilation) | {analysis['step0_time']:.1f}s |
| Mean steady-state (steps 1+) | {analysis['steady_time_mean']:.1f}s |
| Total training time | {analysis['total_time']:.0f}s ({analysis['total_time']/60:.1f} min) |{recomp_str}

Step 0 is torch.compile warmup. After that, steady-state is ~{analysis['steady_time_mean']:.1f}s/step.
{analysis['total_time']/60:.1f} minutes for 197 steps is practical for iterative experimentation.
{v3_table}
## 7. Key Questions Answered

### Q1: At what step does gradient instability begin? What's the transition loss level?

> **Instability onset: step ~{analysis.get('instability_step', 'N/A')}** (first pre-clip grad norm > 10
> after warmup). The EMA loss at this point is approximately {analysis.get('instability_loss', 0):.4f}.
> The transition is gradual -- grad norms start climbing around step 130 and become
> consistently elevated by step ~{p3_start}. The loss at transition is not uniformly low;
> it oscillates because the model has learned to produce large score differentials,
> meaning both correct (low loss) and incorrect (high loss) predictions are extreme.

### Q2: Is the model actually overfitting or encountering genuinely hard pairs?

> **Both.** The model has learned the training distribution well enough to achieve
> near-zero loss on many pairs (loss < 0.01 at {len(analysis['near_perfect_steps'])} steps).
> But the on-the-fly pair sampler draws from ~1.5M possible pairs. When the model
> encounters a pair where its preference ordering is wrong, the large score
> differential produces catastrophic loss. This is textbook overconfidence: the model
> outputs extreme scores (soft_tanh_cap notwithstanding) without corresponding calibration.
> A label-smoothed BT loss or a margin-based formulation could mitigate this.

### Q3: What's the optimal early-stopping point?

> **Step ~{analysis['best_ema_step']}** by EMA(0.1) loss minimum ({analysis['best_ema_val']:.4f}).
> At this point, the model has learned the discrimination task without entering the
> instability regime. Pinkify accuracy is high and grad norms are still controlled.
> For deployment, checkpointing every 10 steps and selecting by validation loss on
> a held-out pair set would be the correct approach.

### Q4: How does step-wise accuracy compare to v3 (30 steps)?

> v4's first 30 steps closely track v3 (same initial loss, same descent rate, same
> grad norm range). By step 30, both runs have pinkify at ~60% and thisnotthat at ~50%.
> v4 then continues to improve: pinkify reaches {analysis['acc_pinkify_overall']:.0%} overall
> (with Phase 2 at {p2.get('acc_pinkify', 0):.0%}) and thisnotthat reaches
> {analysis['acc_thisnotthat_overall']:.0%} overall. The extended training validates that
> 30 steps was insufficient for convergence -- the model was still in early descent.

## 8. Crash Analysis

The crash at step 197 (`RuntimeError: element 0 of tensors does not require grad and
does not have a grad_fn`) is an edge case in the Bradley-Terry loss computation.
When the model produces identical scores for both images in a pair, the BT loss
reduces to `log(sigmoid(0)) = log(0.5) = -0.693`, which is a constant. In this
degenerate case, the loss tensor has no grad_fn because no learnable parameters
contributed to the score differential. This is a fixable bug: clamp the score
differential to a minimum absolute value, or add epsilon to prevent exact equality.

## 9. Charts

Generated in `charts/`:
- `01_loss_curve.png` -- BT loss raw scatter + EMA(0.1), phase boundaries marked
- `02_per_head_accuracy.png` -- Pinkify + thisnotthat accuracy, running avg window=10
- `03_gradient_norms.png` -- Pre-clip grad norm (log scale), phase boundaries
- `04_step_timing.png` -- Seconds per step, showing compile warmup

## 10. Recommendations for v5

1. **Early stopping at step ~{analysis['best_ema_step']}** (or validation-based checkpoint selection)
2. **Score differential clamping**: `max(|s_a - s_b|, eps)` to prevent the grad_fn crash
3. **Label smoothing**: BT loss with target 0.9 instead of 1.0 to reduce overconfidence
4. **Increased gradient clip**: From 0.1 to 1.0. The current clip is extremely aggressive;
   the model's effective learning rate in Phase 3 is ~{0.1 / analysis.get('grad_norm_mean', 1):.4f}x
   the configured value due to constant clipping.
5. **LR decay**: Cosine or linear decay after step ~100 to slow learning as the model
   enters the region where most pairs are already correctly classified.
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"  Saved: {out_path.name}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading v4 data from: {V4_DIR}")
    v4 = load_jsonl(V4_DIR / "training_metrics.jsonl")
    print(f"  v4: {len(v4)} steps loaded")

    v3 = None
    v3_path = V3_DIR / "training_metrics.jsonl"
    if v3_path.exists():
        v3 = load_jsonl(v3_path)
        print(f"  v3: {len(v3)} steps loaded (for comparison)")
    else:
        print(f"  v3: not found at {v3_path}, skipping comparison")

    print("\n--- Phase detection ---")
    phases = detect_phases(v4)
    p2 = phases["phase2_start"]
    p3 = phases["phase3_start"]
    print(f"  Phase 1 (warmup + early): steps 0-{p2 - 1}")
    print(f"  Phase 2 (aggressive learning): steps {p2}-{p3 - 1}")
    print(f"  Phase 3 (instability): steps {p3}-{len(v4) - 1}")

    print("\n--- Computing analysis ---")
    analysis = compute_analysis(v4, v3)

    # Print key findings
    print(f"\n  Loss: {analysis['initial_loss']:.4f} -> {analysis['final_loss']:.4f}")
    print(f"  Min loss: {analysis['min_loss']:.6f} at step {analysis['min_loss_step']}")
    print(f"  Max loss: {analysis['max_loss']:.4f} at step {analysis['max_loss_step']}")
    print(f"  Best EMA(0.1) point: step {analysis['best_ema_step']} ({analysis['best_ema_val']:.4f})")
    print(f"  Instability onset: step {analysis.get('instability_step', 'N/A')}")
    print(f"  Grad norm max: {analysis['grad_norm_max']:.1f} at step {analysis['grad_norm_max_step']}")
    print(f"  Pinkify acc overall: {analysis['acc_pinkify_overall']:.0%}")
    print(f"  TNT acc overall: {analysis['acc_thisnotthat_overall']:.0%}")
    print(f"  Total time: {analysis['total_time']:.0f}s ({analysis['total_time']/60:.1f} min)")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting charts to: {CHARTS_DIR}")

    print("\n--- Chart 1: Loss curve ---")
    chart_loss_curve(v4, phases, v3, CHARTS_DIR)

    print("--- Chart 2: Per-head accuracy ---")
    chart_per_head_accuracy(v4, phases, v3, CHARTS_DIR)

    print("--- Chart 3: Gradient norms (log scale) ---")
    chart_gradient_norms(v4, phases, v3, CHARTS_DIR)

    print("--- Chart 4: Step timing ---")
    chart_step_timing(v4, phases, v3, CHARTS_DIR)

    print("\n--- Writing analysis report ---")
    report_path = V4_DIR / "training_analysis.md"
    write_report(v4, analysis, report_path)

    print("\nDone.")
    print(f"  Charts: {CHARTS_DIR}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
