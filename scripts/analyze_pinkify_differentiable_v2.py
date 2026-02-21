r"""Analyze differentiable BTRM training run metrics and produce charts + summary report.

Reads:
  pinkify_thisnotthat_output/differentiable_run_v2/training_metrics.jsonl  (30 steps)
  pinkify_thisnotthat_output/differentiable_run_v2/run_summary.json
  pinkify_thisnotthat_output/differentiable_run/training_metrics.jsonl     (v1 comparison)

Produces:
  pinkify_thisnotthat_output/differentiable_run_v2/charts/
    01_loss_curve.png
    02_bt_loss_curve.png
    03_per_head_accuracy.png
    04_gradient_norms.png
    05_step_timing.png
  pinkify_thisnotthat_output/differentiable_run_v2/training_analysis.md

All rendering via PIL only -- no matplotlib.

Execution:
  PYTHONUNBUFFERED=1 /mnt/f/dox/repos/ai/futudiffu/.venv/Scripts/python.exe ^
      F:\dox\repos\ai\futudiffu\scripts\analyze_pinkify_differentiable_v2.py
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
V2_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run_v2"
V1_DIR = REPO_ROOT / "pinkify_thisnotthat_output" / "differentiable_run"
CHARTS_DIR = V2_DIR / "charts"


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
# (copied/inlined from scripts_ii/plot_sweep_curves.py to avoid src_ii import
#  path issues -- this script is self-contained)
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
            # If log scale, show as 10^val
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
        legend_w = 220
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

def chart_loss_curve(v2: list[dict], v1: list[dict] | None, out_dir: Path):
    """Chart 1: Total loss vs step."""
    chart = PILChart()
    chart.set_title("Training Loss (Differentiable BTRM v2)")
    chart.set_labels("Step", "Total Loss")

    steps_v2 = [d["step"] for d in v2]
    loss_v2 = [d["loss"] for d in v2]
    ema_v2 = ema(loss_v2, alpha=0.3)

    # Raw loss (light)
    chart.add_line(steps_v2, loss_v2, color="#aaccff", label="v2 raw", line_width=1)
    # EMA smoothed (solid blue)
    chart.add_line(steps_v2, ema_v2, color="#1155cc", label="v2 EMA(0.3)", line_width=2)

    if v1:
        steps_v1 = [d["step"] for d in v1]
        loss_v1 = [d["loss"] for d in v1]
        ema_v1 = ema(loss_v1, alpha=0.3)
        chart.add_line(steps_v1, loss_v1, color="#ffcccc", label="v1 raw", line_width=1)
        chart.add_line(steps_v1, ema_v1, color="#cc1111", label="v1 EMA(0.3)", line_width=2, style="dashed")

    chart.save(out_dir / "01_loss_curve.png")


def chart_bt_loss(v2: list[dict], v1: list[dict] | None, out_dir: Path):
    """Chart 2: Bradley-Terry loss component vs step."""
    chart = PILChart()
    chart.set_title("Bradley-Terry Loss Component (Differentiable BTRM v2)")
    chart.set_labels("Step", "BT Loss")

    steps_v2 = [d["step"] for d in v2]
    bt_v2 = [d["bt_loss"] for d in v2]
    ema_bt_v2 = ema(bt_v2, alpha=0.3)

    chart.add_line(steps_v2, bt_v2, color="#aaccff", label="v2 raw", line_width=1)
    chart.add_line(steps_v2, ema_bt_v2, color="#1155cc", label="v2 EMA(0.3)", line_width=2)

    if v1:
        steps_v1 = [d["step"] for d in v1]
        bt_v1 = [d["bt_loss"] for d in v1]
        ema_bt_v1 = ema(bt_v1, alpha=0.3)
        chart.add_line(steps_v1, bt_v1, color="#ffcccc", label="v1 raw", line_width=1)
        chart.add_line(steps_v1, ema_bt_v1, color="#cc1111", label="v1 EMA(0.3)", line_width=2, style="dashed")

    chart.save(out_dir / "02_bt_loss_curve.png")


def chart_per_head_accuracy(v2: list[dict], v1: list[dict] | None, out_dir: Path):
    """Chart 3: Per-head accuracy (pinkify + thisnotthat) vs step, with running average."""
    chart = PILChart()
    chart.set_title("Per-Head Accuracy vs Step (Differentiable BTRM v2)")
    chart.set_labels("Step", "Accuracy (0/1 per step; lines = running avg window=5)")

    steps_v2 = [d["step"] for d in v2]
    acc_p_v2 = [d["accuracy_pinkify"] for d in v2]
    acc_t_v2 = [d["accuracy_thisnotthat"] for d in v2]

    ravg_p_v2 = running_average(acc_p_v2, window=5)
    ravg_t_v2 = running_average(acc_t_v2, window=5)

    # Raw per-step (scatter dots, light)
    chart.add_scatter(steps_v2, acc_p_v2, color="#88ccff", label="", size=3)
    chart.add_scatter(steps_v2, acc_t_v2, color="#ffcc88", label="", size=3)

    # Running averages (solid lines)
    chart.add_line(steps_v2, ravg_p_v2, color="#1155cc", label="pinkify ravg(5)", line_width=2)
    chart.add_line(steps_v2, ravg_t_v2, color="#cc7711", label="thisnotthat ravg(5)", line_width=2)

    if v1:
        steps_v1 = [d["step"] for d in v1]
        acc_p_v1 = [d["accuracy_pinkify"] for d in v1]
        acc_t_v1 = [d["accuracy_thisnotthat"] for d in v1]
        ravg_p_v1 = running_average(acc_p_v1, window=5)
        ravg_t_v1 = running_average(acc_t_v1, window=5)
        chart.add_line(steps_v1, ravg_p_v1, color="#7799cc", label="v1 pinkify ravg(5)", line_width=2, style="dashed")
        chart.add_line(steps_v1, ravg_t_v1, color="#cc9955", label="v1 thisnotthat ravg(5)", line_width=2, style="dashed")

    chart.save(out_dir / "03_per_head_accuracy.png")


def chart_gradient_norms(v2: list[dict], v1: list[dict] | None, out_dir: Path):
    """Chart 4: Pre-clip grad norm vs step (log scale)."""
    chart = PILChart()
    chart.set_title("Pre-Clip Gradient Norm vs Step (log scale)")
    chart.set_labels("Step", "log10(Pre-Clip Grad Norm)")
    chart.set_log_y(True)

    steps_v2 = [d["step"] for d in v2]
    gn_v2 = [d["pre_clip_grad_norm"] for d in v2]
    ema_gn_v2 = ema(gn_v2, alpha=0.3)

    chart.add_line(steps_v2, gn_v2, color="#aaffaa", label="v2 raw", line_width=1)
    chart.add_line(steps_v2, ema_gn_v2, color="#117711", label="v2 EMA(0.3)", line_width=2)

    if v1:
        steps_v1 = [d["step"] for d in v1]
        gn_v1 = [d["pre_clip_grad_norm"] for d in v1]
        ema_gn_v1 = ema(gn_v1, alpha=0.3)
        chart.add_line(steps_v1, gn_v1, color="#ffccaa", label="v1 raw", line_width=1)
        chart.add_line(steps_v1, ema_gn_v1, color="#cc7711", label="v1 EMA(0.3)", line_width=2, style="dashed")

    chart.save(out_dir / "04_gradient_norms.png")


def chart_step_timing(v2: list[dict], v1: list[dict] | None, out_dir: Path):
    """Chart 5: Seconds per step -- shows compilation warmup vs steady state."""
    chart = PILChart()
    chart.set_title("Step Timing: Seconds per Step (Differentiable BTRM v2)")
    chart.set_labels("Step", "Time (s)")

    steps_v2 = [d["step"] for d in v2]
    time_v2 = [d["time_s"] for d in v2]
    chart.add_line(steps_v2, time_v2, color="#1155cc", label="v2", line_width=2)

    if v1:
        steps_v1 = [d["step"] for d in v1]
        time_v1 = [d["time_s"] for d in v1]
        chart.add_line(steps_v1, time_v1, color="#cc1111", label="v1", line_width=2, style="dashed")

    chart.save(out_dir / "05_step_timing.png")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_report(v2: list[dict], v1: list[dict] | None, summary: dict, out_path: Path):
    steps = [d["step"] for d in v2]
    loss = [d["loss"] for d in v2]
    bt_loss = [d["bt_loss"] for d in v2]
    pre_clip_gn = [d["pre_clip_grad_norm"] for d in v2]
    time_s = [d["time_s"] for d in v2]
    acc_p = [d["accuracy_pinkify"] for d in v2]
    acc_t = [d["accuracy_thisnotthat"] for d in v2]

    n = len(v2)
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

    # Spikes: steps where pre_clip_gn > 2 * mean
    spikes = [(i, v) for i, v in enumerate(pre_clip_gn) if v > 2 * gn_mean]

    # Timing
    t_step0 = time_s[0]
    t_steady_mean = mean(time_s[1:]) if len(time_s) > 1 else time_s[0]
    t_max = max(time_s)
    t_max_step = time_s.index(t_max)
    # Steps 13 and 21 in v2 are the recompilation spikes (seen in data: 19s, 17.5s)
    compilation_steps = [i for i, t in enumerate(time_s) if t > 2 * t_step0 and i > 0]

    # Accuracy (last 10)
    acc_p_last10 = acc_p[last10]
    acc_t_last10 = acc_t[last10]
    acc_p_mean_last10 = mean(acc_p_last10)
    acc_t_mean_last10 = mean(acc_t_last10)
    acc_p_overall = mean(acc_p)
    acc_t_overall = mean(acc_t)

    # v1 comparison
    v1_section = ""
    if v1:
        v1_loss = [d["loss"] for d in v1]
        v1_bt = [d["bt_loss"] for d in v1]
        v1_gn = [d["pre_clip_grad_norm"] for d in v1]
        v1_acc_p = [d["accuracy_pinkify"] for d in v1]
        v1_acc_t = [d["accuracy_thisnotthat"] for d in v1]
        v1_time = [d["time_s"] for d in v1]
        v1_last10_p = mean(v1_acc_p[max(0, len(v1_acc_p) - 10):])
        v1_last10_t = mean(v1_acc_t[max(0, len(v1_acc_t) - 10):])
        v1_section = f"""
## Comparison: v1 vs v2 Run

The v1 run used pre-computed scores (not on-the-fly GPU scoring). The v2 run uses
on-the-fly GPU scoring with the full adapter+backbone differentiated.

| Metric | v1 (pre-scored) | v2 (on-the-fly GPU) |
|--------|----------------|---------------------|
| Initial loss | {v1_loss[0]:.4f} | {loss_initial:.4f} |
| Final loss | {v1_loss[-1]:.4f} | {loss_final:.4f} |
| Loss reduction | {(v1_loss[0] - v1_loss[-1]) / v1_loss[0] * 100:.1f}% | {loss_pct_reduction:.1f}% |
| Initial BT loss | {v1_bt[0]:.4f} | {bt_initial:.4f} |
| Final BT loss | {v1_bt[-1]:.4f} | {bt_final:.4f} |
| Grad norm mean | {mean(v1_gn):.1f} | {gn_mean:.1f} |
| Grad norm max | {max(v1_gn):.1f} (step {v1_gn.index(max(v1_gn))}) | {gn_max:.1f} (step {gn_max_step}) |
| Pinkify acc (last 10) | {v1_last10_p:.0%} | {acc_p_mean_last10:.0%} |
| Thisnotthat acc (last 10) | {v1_last10_t:.0%} | {acc_t_mean_last10:.0%} |
| Step 0 time | {v1_time[0]:.1f}s | {t_step0:.1f}s |
| Steady-state time | {mean(v1_time[1:]):.1f}s | {t_steady_mean:.1f}s |

**Key differences:**
- v1 pre-clip grad norms: mean={mean(v1_gn):.1f}, max={max(v1_gn):.1f}.
  v2 has significantly larger grad norms (mean={gn_mean:.1f}, max={gn_max:.1f}) due to on-the-fly GPU
  scoring adding scoring gradients into the full backward pass.
- v1 final loss ({v1_loss[-1]:.4f}) is lower than v2 final loss ({loss_final:.4f}).
  However, v2 starts from a random adapter init whereas v1 may have benefited from different pair selection
  or initialization. Both runs share the same pair sequence (same steps sampled).
- v2 on-the-fly scoring adds ~{summary.get('avg_score_time_ms', 0):.0f}ms per score call
  ({summary.get('total_score_calls', 0)} calls total, {summary.get('total_score_time_s', 0):.1f}s total
  score time out of {summary.get('train_time_s', 0):.1f}s train time).
"""

    spikes_str = ""
    if spikes:
        spike_list = ", ".join(f"step {i} ({v:.0f})" for i, v in spikes)
        spikes_str = f"\n- **Gradient spikes** at: {spike_list}"

    compilation_str = ""
    if compilation_steps:
        comp_list = ", ".join(f"step {i} ({time_s[i]:.1f}s)" for i in compilation_steps)
        compilation_str = f"\n- Recompilation events observed at: {comp_list}"

    adapter_info = (
        f"- Adapter parameters: {summary.get('n_adapter_params', 'N/A'):,} (LoRA)\n"
        f"- Score unembedder parameters: {summary.get('n_head_params', 'N/A'):,}\n"
        f"- Adapter grad verified at step 0: {summary.get('adapter_grad_verified_step0', False)}\n"
        f"- Scoring method: {summary.get('scoring_method', 'N/A')}\n"
        f"- Dataset trajectories: {summary.get('n_trajectories', 'N/A')}\n"
        f"- Pair space size: {summary.get('sampler_stats', {}).get('pair_space_size', 'N/A'):,}\n"
    )

    report = f"""# Differentiable BTRM Training Analysis (v2)

**Run:** `pinkify_thisnotthat_differentiable_v2`
**Date:** 2026-02-18
**Steps:** {n}
**Wall time:** {summary.get('wall_total_s', 0):.1f}s ({summary.get('wall_total_s', 0)/60:.1f} min)
**Train time:** {summary.get('train_time_s', 0):.1f}s
**LR:** {summary.get('lr', 'N/A')} (warmup for first {summary.get('warmup_steps', 0)} steps)
**Grad clip:** {summary.get('grad_clip', 'N/A')}

## Run Configuration

{adapter_info}

## 1. Loss Trajectory

| Metric | Value |
|--------|-------|
| Initial total loss | {loss_initial:.4f} |
| Final total loss | {loss_final:.4f} |
| Reduction | {loss_pct_reduction:.1f}% |
| Minimum loss | {loss_min:.4f} (step {loss_min_step}) |
| Initial BT loss | {bt_initial:.4f} |
| Final BT loss | {bt_final:.4f} |
| BT loss reduction | {(bt_initial - bt_final) / bt_initial * 100:.1f}% |

The total loss reduction of {loss_pct_reduction:.1f}% over 30 steps indicates the adapter is learning.
The BT loss component (which drives the pairwise ranking) drops from {bt_initial:.4f} to {bt_final:.4f}.

**Note on logsquare regularizer:** The logsq_loss term is large and negative (ranges from
{min(d["logsq_loss"] for d in v2):.2f} to {max(d["logsq_loss"] for d in v2):.2f}), pulling total
loss below BT loss. This reflects the logsquare regularizer with weight={summary.get('logsquare_weight', 0.05)}.

## 2. Gradient Norm Statistics

| Metric | Value |
|--------|-------|
| Mean pre-clip grad norm | {gn_mean:.1f} |
| Max pre-clip grad norm | {gn_max:.1f} (step {gn_max_step}) |
| Final pre-clip grad norm | {gn_final:.1f} |
| Post-clip grad norm | ~{mean([d["grad_norm"] for d in v2]):.4f} (clipped to {summary.get('grad_clip', 'N/A')}) |{spikes_str}

The pre-clip grad norm is consistently high (mean {gn_mean:.1f}), indicating the adapter is
producing large gradients relative to the clip threshold ({summary.get('grad_clip', 'N/A')}).
Grad clipping is active on nearly every step -- the post-clip norm is essentially always at
the clip ceiling of {summary.get('grad_clip', 'N/A')}.

The large pre-clip norms (especially steps 15-23 where they reach 200-550) suggest the adapter
is in an aggressive learning phase. The final step's pre-clip norm drops to {gn_final:.1f}, which
may indicate convergence toward a local optimum or the pair sampler selecting an "easy" pair.

## 3. Per-Head Accuracy

| Head | Overall mean | Last 10 steps mean |
|------|-------------|-------------------|
| pinkify | {acc_p_overall:.0%} | {acc_p_mean_last10:.0%} |
| thisnotthat | {acc_t_overall:.0%} | {acc_t_mean_last10:.0%} |

**Pinkify head:** {acc_p_mean_last10:.0%} accuracy in the last 10 steps.
Pinkify discrimination (SDPA vs SageAttention INT8 QK) is the "bit quality" head.

**Thisnotthat head:** {acc_t_mean_last10:.0%} accuracy in the last 10 steps.
Thisnotthat discrimination (step count: 30 vs 8-22) is the "step quality" head.

Note: Per-step accuracy is binary (0 or 1) since each macrobatch is a single pair. Both
heads show erratic step-by-step accuracy; the running average (window=5) in the chart
shows the trend. The final reported accuracy per head ({summary.get('final_accuracy_pinkify', 0):.0%}
pinkify, {summary.get('final_accuracy_thisnotthat', 0):.0%} thisnotthat) matches the last step value.

The asymmetry between heads (pinkify learning faster than thisnotthat) mirrors what was
observed in run02: scrongle (now thisnotthat) is the harder discrimination task.

## 4. Step Timing

| Metric | Value |
|--------|-------|
| Step 0 (compilation) | {t_step0:.1f}s |
| Mean steady-state (steps 1+) | {t_steady_mean:.1f}s |
| Max step time | {t_max:.1f}s (step {t_max_step}) |
| Total training time | {sum(time_s):.1f}s |{compilation_str}

Step 0 at {t_step0:.1f}s is the torch.compile warmup. Steady state settles at ~{t_steady_mean:.1f}s/step.
The spikes at steps 13 ({time_s[13]:.1f}s) and 21 ({time_s[21]:.1f}s) correspond to
recompilation events when the FlexAttention sequence length changes (new bin packing configuration).
{v1_section}
## 5. Anomalies and Concerns

1. **Grad norm climbing mid-run:** Pre-clip grad norm rises from ~15 at step 1 to a peak of {gn_max:.0f}
   at step {gn_max_step}, then partially subsides. The grad clip at {summary.get('grad_clip', 'N/A')}
   bounds the effective update size, but the large pre-clip norms suggest the loss landscape has
   high-curvature regions being traversed during the LR warmup phase.

2. **Thisnotthat accuracy instability:** Thisnotthat accuracy (step quality head) oscillates 0/1
   throughout the run. Overall accuracy {acc_t_overall:.0%} vs pinkify {acc_p_overall:.0%}. This is expected --
   step count discrimination requires the model to distinguish images generated with 30 steps vs 8-22
   steps, which is a subtler difference than the attention quantization artifact targeted by pinkify.

3. **BT loss not descending strongly:** Final BT loss ({bt_final:.4f}) vs initial ({bt_initial:.4f})
   is only a {(bt_initial - bt_final) / bt_initial * 100:.1f}% reduction. The logsquare regularizer
   is dominating the loss signal, which may be competing with the pairwise ranking objective.
   Reducing `logsquare_weight` from {summary.get('logsquare_weight', 0.05)} in future runs may help
   the BT loss descend more aggressively.

4. **Run is short (30 steps):** 30 macrobatches is diagnostic/validation, not convergence. Both
   run02 (30 macrobatches) and this run show loss reduction but don't reach saturation. Extended
   training on the full {summary.get('n_trajectories', 259)}-trajectory dataset is the next step.

## 6. Charts

Generated in `charts/`:
- `01_loss_curve.png` -- Total loss vs step (raw + EMA, v1 comparison)
- `02_bt_loss_curve.png` -- BT loss component vs step (raw + EMA, v1 comparison)
- `03_per_head_accuracy.png` -- Pinkify + thisnotthat accuracy, raw scatter + running average
- `04_gradient_norms.png` -- Pre-clip grad norm vs step, log scale (v1 comparison)
- `05_step_timing.png` -- Seconds per step, showing compilation warmup and recompilation spikes
"""

    out_path.write_text(report)
    print(f"  Saved: {out_path.name}")
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading v2 data from: {V2_DIR}")
    v2_metrics = load_jsonl(V2_DIR / "training_metrics.jsonl")
    v2_summary = load_json(V2_DIR / "run_summary.json")
    print(f"  v2: {len(v2_metrics)} steps loaded")

    v1_metrics = None
    v1_path = V1_DIR / "training_metrics.jsonl"
    if v1_path.exists():
        v1_metrics = load_jsonl(v1_path)
        print(f"  v1: {len(v1_metrics)} steps loaded (for comparison)")
    else:
        print(f"  v1: not found at {v1_path}, skipping comparison")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting charts to: {CHARTS_DIR}")

    print("\n--- Chart 1: Loss curve ---")
    chart_loss_curve(v2_metrics, v1_metrics, CHARTS_DIR)

    print("--- Chart 2: BT loss curve ---")
    chart_bt_loss(v2_metrics, v1_metrics, CHARTS_DIR)

    print("--- Chart 3: Per-head accuracy ---")
    chart_per_head_accuracy(v2_metrics, v1_metrics, CHARTS_DIR)

    print("--- Chart 4: Gradient norms (log scale) ---")
    chart_gradient_norms(v2_metrics, v1_metrics, CHARTS_DIR)

    print("--- Chart 5: Step timing ---")
    chart_step_timing(v2_metrics, v1_metrics, CHARTS_DIR)

    print("\n--- Writing summary report ---")
    report_path = V2_DIR / "training_analysis.md"
    write_report(v2_metrics, v1_metrics, v2_summary, report_path)

    print("\nDone.")
    print(f"  Charts: {CHARTS_DIR}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
