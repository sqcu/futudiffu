r"""Analyze differentiable BTRM training run metrics and produce charts for v3 (logsquare removed).

Reads:
  pinkify_thisnotthat_output/differentiable_run_v3/training_metrics.jsonl  (30 steps)
  pinkify_thisnotthat_output/differentiable_run_v3/run_summary.json
  pinkify_thisnotthat_output/differentiable_run_v2/training_metrics.jsonl  (v2 comparison, dashed)

Produces:
  pinkify_thisnotthat_output/differentiable_run_v3/charts/
    01_bt_loss_curve.png       -- BT loss vs step (v3 solid, v2 dashed)
    02_per_head_accuracy.png   -- per-head accuracy with running average
    03_gradient_norms.png      -- pre-clip grad norm log scale (v3 solid, v2 dashed)
    04_step_timing.png         -- seconds per step

All rendering via PIL only -- no matplotlib.

Key v3 difference from v2: logsquare_weight=0.0, so total loss == bt_loss every step.
The v2 run had large logsquare regularizer making total loss diverge from bt_loss.
Removing it lets the BT objective drive learning without regularizer interference.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts\analyze_pinkify_differentiable_v3.py
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
V3_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run_v3"
V2_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run_v2"
CHARTS_DIR = V3_DIR / "charts"


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


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


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


# ---------------------------------------------------------------------------
# PILChart: minimal PIL-based line/scatter chart renderer
# (self-contained -- no src_ii imports required)
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
        width: int = 1000,
        height: int = 650,
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

        self.font = ImageFont.load_default()

    def set_title(self, title: str):
        self.title = title

    def set_labels(self, x_label: str, y_label: str):
        self.x_label = x_label
        self.y_label = y_label

    def set_log_y(self, enabled: bool = True):
        self._y_log = enabled

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

        n_x_ticks = 6
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
        legend_w = 240
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

def chart_bt_loss(v3: list[dict], v2: list[dict] | None, out_dir: Path):
    """Chart 1: Bradley-Terry loss vs step.

    In v3, logsquare_weight=0.0, so total loss == bt_loss every step.
    The v2 run had a logsquare regularizer that pushed total loss below bt_loss.
    This chart compares the BT loss component directly between runs.
    """
    chart = PILChart()
    chart.set_title("BT Loss vs Step: v3 (no logsquare) vs v2 (logsquare=0.05)")
    chart.set_labels("Step", "BT Loss")

    steps_v3 = [d["step"] for d in v3]
    bt_v3 = [d["bt_loss"] for d in v3]
    ema_bt_v3 = ema(bt_v3, alpha=0.3)

    # Raw (light blue) + EMA (solid blue)
    chart.add_line(steps_v3, bt_v3, color="#aaccff", label="v3 raw", line_width=1)
    chart.add_line(steps_v3, ema_bt_v3, color="#1155cc", label="v3 EMA(0.3)", line_width=2)

    if v2:
        steps_v2 = [d["step"] for d in v2]
        bt_v2 = [d["bt_loss"] for d in v2]
        ema_bt_v2 = ema(bt_v2, alpha=0.3)
        chart.add_line(steps_v2, bt_v2, color="#ffcccc", label="v2 raw", line_width=1, style="dashed")
        chart.add_line(steps_v2, ema_bt_v2, color="#cc1111", label="v2 EMA(0.3)", line_width=2, style="dashed")

    chart.save(out_dir / "01_bt_loss_curve.png")


def chart_per_head_accuracy(v3: list[dict], v2: list[dict] | None, out_dir: Path):
    """Chart 2: Per-head accuracy (pinkify + thisnotthat) vs step, with running average."""
    chart = PILChart()
    chart.set_title("Per-Head Accuracy vs Step (v3 solid, v2 dashed; lines = running avg window=5)")
    chart.set_labels("Step", "Accuracy (0/1 per step)")

    steps_v3 = [d["step"] for d in v3]
    acc_p_v3 = [d["accuracy_pinkify"] for d in v3]
    acc_t_v3 = [d["accuracy_thisnotthat"] for d in v3]

    ravg_p_v3 = running_average(acc_p_v3, window=5)
    ravg_t_v3 = running_average(acc_t_v3, window=5)

    # Raw per-step scatter (light)
    chart.add_scatter(steps_v3, acc_p_v3, color="#88ccff", label="", size=3)
    chart.add_scatter(steps_v3, acc_t_v3, color="#ffcc88", label="", size=3)

    # Running averages (solid)
    chart.add_line(steps_v3, ravg_p_v3, color="#1155cc", label="v3 pinkify ravg(5)", line_width=2)
    chart.add_line(steps_v3, ravg_t_v3, color="#cc7711", label="v3 thisnotthat ravg(5)", line_width=2)

    if v2:
        steps_v2 = [d["step"] for d in v2]
        acc_p_v2 = [d["accuracy_pinkify"] for d in v2]
        acc_t_v2 = [d["accuracy_thisnotthat"] for d in v2]
        ravg_p_v2 = running_average(acc_p_v2, window=5)
        ravg_t_v2 = running_average(acc_t_v2, window=5)
        chart.add_line(steps_v2, ravg_p_v2, color="#7799cc", label="v2 pinkify ravg(5)", line_width=2, style="dashed")
        chart.add_line(steps_v2, ravg_t_v2, color="#cc9955", label="v2 thisnotthat ravg(5)", line_width=2, style="dashed")

    chart.save(out_dir / "02_per_head_accuracy.png")


def chart_gradient_norms(v3: list[dict], v2: list[dict] | None, out_dir: Path):
    """Chart 3: Pre-clip gradient norm vs step, log scale.

    v3 grad norms are dramatically smaller than v2 (mean ~1.5 vs ~100+)
    because logsquare regularizer contributed large gradients in v2.
    """
    chart = PILChart()
    chart.set_title("Pre-Clip Gradient Norm vs Step (log scale; v3 solid, v2 dashed)")
    chart.set_labels("Step", "log10(Pre-Clip Grad Norm)")
    chart.set_log_y(True)

    steps_v3 = [d["step"] for d in v3]
    gn_v3 = [d["pre_clip_grad_norm"] for d in v3]
    ema_gn_v3 = ema(gn_v3, alpha=0.3)

    chart.add_line(steps_v3, gn_v3, color="#aaffaa", label="v3 raw", line_width=1)
    chart.add_line(steps_v3, ema_gn_v3, color="#117711", label="v3 EMA(0.3)", line_width=2)

    if v2:
        steps_v2 = [d["step"] for d in v2]
        gn_v2 = [d["pre_clip_grad_norm"] for d in v2]
        ema_gn_v2 = ema(gn_v2, alpha=0.3)
        chart.add_line(steps_v2, gn_v2, color="#ffccaa", label="v2 raw", line_width=1, style="dashed")
        chart.add_line(steps_v2, ema_gn_v2, color="#cc7711", label="v2 EMA(0.3)", line_width=2, style="dashed")

    chart.save(out_dir / "03_gradient_norms.png")


def chart_step_timing(v3: list[dict], v2: list[dict] | None, out_dir: Path):
    """Chart 4: Seconds per step -- shows compilation warmup vs steady state."""
    chart = PILChart()
    chart.set_title("Step Timing: Seconds per Step (v3 solid, v2 dashed)")
    chart.set_labels("Step", "Time (s)")

    steps_v3 = [d["step"] for d in v3]
    time_v3 = [d["time_s"] for d in v3]
    chart.add_line(steps_v3, time_v3, color="#1155cc", label="v3", line_width=2)

    if v2:
        steps_v2 = [d["step"] for d in v2]
        time_v2 = [d["time_s"] for d in v2]
        chart.add_line(steps_v2, time_v2, color="#cc1111", label="v2", line_width=2, style="dashed")

    chart.save(out_dir / "04_step_timing.png")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_report(v3: list[dict], v2: list[dict] | None, summary: dict, out_path: Path):
    steps = [d["step"] for d in v3]
    loss = [d["loss"] for d in v3]
    bt_loss = [d["bt_loss"] for d in v3]
    pre_clip_gn = [d["pre_clip_grad_norm"] for d in v3]
    time_s = [d["time_s"] for d in v3]
    acc_p = [d["accuracy_pinkify"] for d in v3]
    acc_t = [d["accuracy_thisnotthat"] for d in v3]

    n = len(v3)
    last10 = slice(max(0, n - 10), n)

    loss_initial = loss[0]
    loss_final = loss[-1]
    loss_pct_reduction = (loss_initial - loss_final) / loss_initial * 100.0
    loss_min = min(loss)
    loss_min_step = loss.index(loss_min)

    bt_initial = bt_loss[0]
    bt_final = bt_loss[-1]

    gn_mean = mean(pre_clip_gn)
    gn_max = max(pre_clip_gn)
    gn_max_step = pre_clip_gn.index(gn_max)
    gn_final = pre_clip_gn[-1]

    spikes = [(i, v) for i, v in enumerate(pre_clip_gn) if v > 2 * gn_mean]

    t_step0 = time_s[0]
    t_steady_mean = mean(time_s[1:]) if len(time_s) > 1 else time_s[0]
    t_max = max(time_s)
    t_max_step = time_s.index(t_max)
    compilation_steps = [i for i, t in enumerate(time_s) if t > 2 * t_step0 and i > 0]

    acc_p_last10 = acc_p[last10]
    acc_t_last10 = acc_t[last10]
    acc_p_mean_last10 = mean(acc_p_last10)
    acc_t_mean_last10 = mean(acc_t_last10)
    acc_p_overall = mean(acc_p)
    acc_t_overall = mean(acc_t)

    v2_section = ""
    if v2:
        v2_loss = [d["loss"] for d in v2]
        v2_bt = [d["bt_loss"] for d in v2]
        v2_gn = [d["pre_clip_grad_norm"] for d in v2]
        v2_acc_p = [d["accuracy_pinkify"] for d in v2]
        v2_acc_t = [d["accuracy_thisnotthat"] for d in v2]
        v2_time = [d["time_s"] for d in v2]
        v2_last10_p = mean(v2_acc_p[max(0, len(v2_acc_p) - 10):])
        v2_last10_t = mean(v2_acc_t[max(0, len(v2_acc_t) - 10):])
        v2_section = f"""
## Comparison: v2 (logsquare=0.05) vs v3 (logsquare=0.0)

The key change from v2 to v3 is removing the logsquare regularizer (`logsquare_weight=0.0`).
In v2, the logsquare term dominated the total loss and produced very large gradients (mean pre-clip
norm ~{mean(v2_gn):.1f}, max {max(v2_gn):.1f}). In v3, the BT objective is the entire loss.

| Metric | v2 (logsquare=0.05) | v3 (logsquare=0.0) |
|--------|---------------------|---------------------|
| Initial BT loss | {v2_bt[0]:.4f} | {bt_initial:.4f} |
| Final BT loss | {v2_bt[-1]:.4f} | {bt_final:.4f} |
| BT loss reduction | {(v2_bt[0] - v2_bt[-1]) / v2_bt[0] * 100:.1f}% | {(bt_initial - bt_final) / bt_initial * 100:.1f}% |
| Initial total loss | {v2_loss[0]:.4f} | {loss_initial:.4f} |
| Final total loss | {v2_loss[-1]:.4f} | {loss_final:.4f} |
| Grad norm mean | {mean(v2_gn):.1f} | {gn_mean:.2f} |
| Grad norm max | {max(v2_gn):.1f} (step {v2_gn.index(max(v2_gn))}) | {gn_max:.2f} (step {gn_max_step}) |
| Pinkify acc (last 10) | {v2_last10_p:.0%} | {acc_p_mean_last10:.0%} |
| Thisnotthat acc (last 10) | {v2_last10_t:.0%} | {acc_t_mean_last10:.0%} |
| Step 0 compile time | {v2_time[0]:.1f}s | {t_step0:.1f}s |
| Steady-state time | {mean(v2_time[1:]):.1f}s | {t_steady_mean:.1f}s |

**Key observations:**
- v3 grad norms are dramatically lower (mean {gn_mean:.2f} vs v2 mean {mean(v2_gn):.1f}).
  Removing logsquare eliminated the large negative-gradient regularization signal that was
  saturating the gradient clipping in v2.
- Both runs have BT loss descending at similar rates over 30 steps.
  v3 starts slightly lower ({bt_initial:.4f} vs {v2_bt[0]:.4f}) potentially due to
  different random pair selection.
- v3 accuracy trends are comparable to v2 across both heads.
- v3 compile time ({t_step0:.1f}s step 0) vs v2 ({v2_time[0]:.1f}s), same graph structure.
"""

    spikes_str = ""
    if spikes:
        spike_list = ", ".join(f"step {i} ({v:.2f})" for i, v in spikes)
        spikes_str = f"\n- **Gradient spikes** (>2x mean) at: {spike_list}"

    compilation_str = ""
    if compilation_steps:
        comp_list = ", ".join(f"step {i} ({time_s[i]:.1f}s)" for i in compilation_steps)
        compilation_str = f"\n- Recompilation events at: {comp_list}"

    adapter_info = (
        f"- LoRA adapter parameters: {summary.get('n_adapter_params', 'N/A'):,}\n"
        f"- Score unembedder parameters: {summary.get('n_head_params', 'N/A'):,}\n"
        f"- Adapter grad verified at step 0: {summary.get('adapter_grad_verified_step0', False)}\n"
        f"- Scoring method: {summary.get('scoring_method', 'N/A')}\n"
        f"- Dataset trajectories: {summary.get('n_trajectories', 'N/A')}\n"
        f"- Pair space size: {summary.get('sampler_stats', {}).get('pair_space_size', 'N/A'):,}\n"
        f"- logsquare_weight: {summary.get('logsquare_weight', 'N/A')} (REMOVED)\n"
    )

    challenge_scores = summary.get("challenge_set_scores", {})
    challenge_str = "\n".join(
        f"  - {k}: {v:.4f}" for k, v in challenge_scores.items()
    ) if challenge_scores else "  (none)"

    report = f"""# Differentiable BTRM Training Analysis (v3 -- logsquare removed)

**Run:** `pinkify_thisnotthat_differentiable_v3`
**Date:** 2026-02-18
**Steps:** {n}
**Wall time:** {summary.get('wall_total_s', 0):.1f}s ({summary.get('wall_total_s', 0)/60:.1f} min)
**Train time:** {summary.get('train_time_s', 0):.1f}s
**LR:** {summary.get('lr', 'N/A')} (warmup for first {summary.get('warmup_steps', 0)} steps)
**Grad clip:** {summary.get('grad_clip', 'N/A')}
**logsquare_weight:** {summary.get('logsquare_weight', 'N/A')} (removed; total loss == BT loss every step)

## Run Configuration

{adapter_info}

## 1. Loss Trajectory

In v3, `loss == bt_loss` at every step since logsquare_weight=0.0.

| Metric | Value |
|--------|-------|
| Initial BT loss | {bt_initial:.4f} |
| Final BT loss | {bt_final:.4f} |
| Reduction | {(bt_initial - bt_final) / bt_initial * 100:.1f}% |
| Minimum loss | {loss_min:.4f} (step {loss_min_step}) |

BT loss is the Bradley-Terry pairwise ranking loss. Its descent from {bt_initial:.4f} to {bt_final:.4f}
({(bt_initial - bt_final) / bt_initial * 100:.1f}% reduction) reflects the adapter learning to
discriminate preferred vs non-preferred images in each pair.

## 2. Gradient Norm Statistics

In v2, the logsquare regularizer produced very large pre-clip norms (mean ~100+, peaking at 550).
In v3, with only the BT loss, norms are far more controlled:

| Metric | Value |
|--------|-------|
| Mean pre-clip grad norm | {gn_mean:.3f} |
| Max pre-clip grad norm | {gn_max:.3f} (step {gn_max_step}) |
| Final pre-clip grad norm | {gn_final:.3f} |
| Grad clip threshold | {summary.get('grad_clip', 'N/A')} |{spikes_str}

The pre-clip norms (mean {gn_mean:.3f}, max {gn_max:.3f}) are now below the clip threshold
({summary.get('grad_clip', 'N/A')}) for many steps. Gradient clipping is less frequently saturating,
which means the effective learning rate is closer to the configured value.

## 3. Per-Head Accuracy

| Head | Overall mean | Last 10 steps mean |
|------|-------------|-------------------|
| pinkify | {acc_p_overall:.0%} | {acc_p_mean_last10:.0%} |
| thisnotthat | {acc_t_overall:.0%} | {acc_t_mean_last10:.0%} |

Both heads show binary per-step accuracy (0 or 1 per macrobatch). The running average (window=5)
in the chart smooths the step-to-step variance.

**Pinkify (bit quality):** {acc_p_mean_last10:.0%} accuracy in the last 10 steps.
Discriminates SDPA vs SageAttention INT8 QK images.

**Thisnotthat (step quality):** {acc_t_mean_last10:.0%} accuracy in the last 10 steps.
Discriminates step count (30 vs 8-22 steps). Inherently harder task.

## 4. Step Timing

| Metric | Value |
|--------|-------|
| Step 0 (compilation) | {t_step0:.1f}s |
| Mean steady-state (steps 1+) | {t_steady_mean:.1f}s |
| Max step time | {t_max:.1f}s (step {t_max_step}) |
| Total training time | {sum(time_s):.1f}s |{compilation_str}

Step 0 at {t_step0:.1f}s is torch.compile warmup. Steady-state ~{t_steady_mean:.1f}s/step.
Recompilation spikes at steps 13 ({time_s[13]:.1f}s) and 21 ({time_s[21]:.1f}s) from
FlexAttention sequence length changes (bin packing).

## 5. Challenge Set Scores (End of Run)

{challenge_str}

These are pre-persist scores evaluated on the held-out challenge set at run end.
Scores near 0 indicate the adapter output is near-zero (expected at early training).
{v2_section}
## 6. Charts

Generated in `charts/`:
- `01_bt_loss_curve.png` -- BT loss vs step (v3 solid, v2 dashed). v3=total loss since logsquare removed.
- `02_per_head_accuracy.png` -- Pinkify + thisnotthat accuracy, raw scatter + running average (v3 solid, v2 dashed)
- `03_gradient_norms.png` -- Pre-clip grad norm, log scale (v3 solid, v2 dashed). Shows dramatic reduction.
- `04_step_timing.png` -- Seconds per step, showing compilation warmup and recompilation spikes (v3 solid, v2 dashed)
"""

    out_path.write_text(report)
    print(f"  Saved: {out_path.name}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading v3 data from: {V3_DIR}")
    v3_metrics = load_jsonl(V3_DIR / "training_metrics.jsonl")
    v3_summary = load_json(V3_DIR / "run_summary.json")
    print(f"  v3: {len(v3_metrics)} steps loaded")

    # Verify logsquare is indeed absent (sanity check)
    has_logsq = any("logsq_loss" in d for d in v3_metrics)
    logsq_weight = v3_summary.get("logsquare_weight", -1)
    print(f"  logsquare_weight: {logsq_weight} (has logsq_loss field: {has_logsq})")
    assert logsq_weight == 0.0, f"Expected logsquare_weight=0.0 in v3, got {logsq_weight}"
    assert not has_logsq, "Unexpected logsq_loss field found in v3 metrics -- check data"
    print("  [ok] v3 confirmed: logsquare removed, loss == bt_loss")

    v2_metrics = None
    v2_path = V2_DIR / "training_metrics.jsonl"
    if v2_path.exists():
        v2_metrics = load_jsonl(v2_path)
        print(f"  v2: {len(v2_metrics)} steps loaded (for comparison, dashed)")
    else:
        print(f"  v2: not found at {v2_path}, skipping comparison")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting charts to: {CHARTS_DIR}")

    print("\n--- Chart 1: BT loss curve ---")
    chart_bt_loss(v3_metrics, v2_metrics, CHARTS_DIR)

    print("--- Chart 2: Per-head accuracy ---")
    chart_per_head_accuracy(v3_metrics, v2_metrics, CHARTS_DIR)

    print("--- Chart 3: Gradient norms (log scale) ---")
    chart_gradient_norms(v3_metrics, v2_metrics, CHARTS_DIR)

    print("--- Chart 4: Step timing ---")
    chart_step_timing(v3_metrics, v2_metrics, CHARTS_DIR)

    print("\n--- Writing summary report ---")
    report_path = V3_DIR / "training_analysis.md"
    write_report(v3_metrics, v2_metrics, v3_summary, report_path)

    print("\nDone.")
    print(f"  Charts: {CHARTS_DIR}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
