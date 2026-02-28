"""Training run artifact manager: JSONL logging, checkpoints, charts, analysis.

Canonical module for all persisted outputs from a BTRM training run.
Every training script uses this via 1-line import + 1-3 line invocation
instead of inlining JSONL writing, chart rendering, and stats computation.

Usage:
    from src_ii.training_artifacts import TrainingArtifacts

    artifacts = TrainingArtifacts(output_dir="training_output/run03/", run_name="funfetti_v1")

    # During training (called per step, either directly or via callback):
    artifacts.log_step(step, loss, accuracy_dict, grad_norm, lr, extra_metrics)

    # At checkpoints:
    artifacts.save_checkpoint(step, model, adapter_name="rtheta")

    # After training:
    artifacts.generate_analysis()  # produces charts/ + training_analysis.md

Import constraints:
    - PIL for chart rendering (PILChart, no matplotlib)
    - json for JSONL / manifest writing
    - DOES NOT import: model_manager, server, client, torch (except for checkpoint saving)
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Statistics helpers (extracted from inline analysis scripts)
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _running_average(xs: list[float], window: int) -> list[float]:
    """Causal unweighted running average."""
    out = []
    for i in range(len(xs)):
        start = max(0, i - window + 1)
        window_vals = xs[start:i + 1]
        out.append(sum(window_vals) / len(window_vals))
    return out


def _ema(values: list[float], alpha: float) -> list[float]:
    """Exponential moving average. alpha is weight on the new sample."""
    if not values:
        return []
    out = []
    s = values[0]
    for v in values:
        s = alpha * v + (1.0 - alpha) * s
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# PILChart: PIL-only line/scatter chart renderer
# Extracted from scripts/analyze_pinkify_differentiable_v4.py to be reusable.
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
    """Minimal line/scatter chart renderer using PIL (no matplotlib)."""

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
        from PIL import Image, ImageDraw, ImageFont

        self.img_w = width
        self.img_h = height
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
            "type": "line", "xs": list(xs), "ys": list(ys),
            "color": color, "label": label, "line_width": line_width, "style": style,
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
            "type": "scatter", "xs": list(xs), "ys": list(ys),
            "color": color, "label": label, "size": size,
        })

    def _transform_y(self, y: float) -> float:
        if self._y_log:
            return math.log10(max(y, 1e-10))
        return y

    def _compute_bounds(self):
        all_xs, all_ys = [], []
        for s in self.series:
            all_xs.extend(s["xs"])
            all_ys.extend(self._transform_y(v) for v in s["ys"])
        for vl in self._vlines:
            all_xs.append(vl["x"])
        if not all_xs or not all_ys:
            return 0, 1, 0, 1
        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys), max(all_ys)
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        return x_min - 0.05 * x_range, x_max + 0.05 * x_range, y_min - 0.05 * y_range, y_max + 0.05 * y_range

    def _data_to_pixel(self, x, y, x_min, x_max, y_min, y_max):
        ty = self._transform_y(y)
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        px = self.ml + (x - x_min) / x_range * self.plot_w
        py = self.mt + (1.0 - (ty - y_min) / y_range) * self.plot_h
        return int(px), int(py)

    def _draw_axes(self, x_min, x_max, y_min, y_max):
        d = self.draw
        d.rectangle([self.ml, self.mt, self.ml + self.plot_w, self.mt + self.plot_h], outline="#cccccc", width=1)
        for i in range(9):
            frac = i / 8
            val = x_min + frac * (x_max - x_min)
            px = self.ml + int(frac * self.plot_w)
            d.line([(px, self.mt), (px, self.mt + self.plot_h)], fill="#eeeeee", width=1)
            d.text((px, self.mt + self.plot_h + 5), _format_tick(val), fill="#333333", font=self.font, anchor="mt")
        for i in range(7):
            frac = i / 6
            val = y_min + frac * (y_max - y_min)
            py = self.mt + int((1.0 - frac) * self.plot_h)
            d.line([(self.ml, py), (self.ml + self.plot_w, py)], fill="#eeeeee", width=1)
            if self._y_log:
                label = f"1e{val:.0f}" if abs(val) >= 1 else f"10^{val:.1f}"
            else:
                label = _format_tick(val)
            d.text((self.ml - 5, py), label, fill="#333333", font=self.font, anchor="rm")
        if self.x_label:
            d.text((self.ml + self.plot_w // 2, self.img_h - 10), self.x_label, fill="#000000", font=self.font, anchor="mb")
        if self.y_label:
            d.text((5, self.mt + self.plot_h // 2), self.y_label, fill="#000000", font=self.font, anchor="lm")

    def _draw_vlines(self, x_min, x_max, y_min, y_max):
        d = self.draw
        for vl in self._vlines:
            x_range = x_max - x_min if x_max != x_min else 1.0
            px = self.ml + int((vl["x"] - x_min) / x_range * self.plot_w)
            color = vl["color"]
            if vl.get("style") == "dashed":
                for y_pos in range(self.mt, self.mt + self.plot_h, 12):
                    y_end = min(y_pos + 6, self.mt + self.plot_h)
                    d.line([(px, y_pos), (px, y_end)], fill=color, width=2)
            else:
                d.line([(px, self.mt), (px, self.mt + self.plot_h)], fill=color, width=2)
            if vl.get("label"):
                d.text((px + 3, self.mt + 5), vl["label"], fill=color, font=self.font, anchor="lt")

    def _draw_title(self):
        if self.title:
            self.draw.text((self.ml + self.plot_w // 2, 10), self.title, fill="#000000", font=self.font, anchor="mt")

    def _draw_legend(self):
        labeled = [s for s in self.series if s.get("label")]
        if not labeled:
            return
        d = self.draw
        n = len(labeled)
        line_height = 18
        legend_w = 260
        legend_h = n * line_height + 10
        x_start = self.ml + self.plot_w - legend_w - 10
        y_start = self.mt + 10
        d.rectangle([x_start - 5, y_start - 5, x_start + legend_w, y_start + legend_h], fill="#ffffff", outline="#cccccc")
        for i, s in enumerate(labeled):
            y = y_start + i * line_height
            color = s["color"]
            if s["type"] == "line":
                if s.get("style") == "dashed":
                    for dx in range(0, 25, 6):
                        d.line([(x_start, y + 7), (x_start + min(dx + 3, 25), y + 7)], fill=color, width=2)
                else:
                    d.line([(x_start, y + 7), (x_start + 25, y + 7)], fill=color, width=2)
            else:
                d.ellipse([x_start + 8, y + 3, x_start + 17, y + 12], fill=color)
            d.text((x_start + 30, y + 2), s["label"], fill="#333333", font=self.font)

    def _draw_series(self, x_min, x_max, y_min, y_max):
        d = self.draw
        for s in self.series:
            xs, ys, color = s["xs"], s["ys"], s["color"]
            if s["type"] == "line":
                lw = s.get("line_width", 2)
                style = s.get("style", "solid")
                points = [self._data_to_pixel(x, y, x_min, x_max, y_min, y_max) for x, y in zip(xs, ys)]
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
                    d.ellipse([px - sz, py - sz, px + sz, py + sz], fill=color)

    def render(self):
        x_min, x_max, y_min, y_max = self._compute_bounds()
        self._draw_axes(x_min, x_max, y_min, y_max)
        self._draw_vlines(x_min, x_max, y_min, y_max)
        self._draw_series(x_min, x_max, y_min, y_max)
        self._draw_title()
        self._draw_legend()
        return self.img

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.render().save(str(path))


# ---------------------------------------------------------------------------
# TrainingArtifacts: the main class
# ---------------------------------------------------------------------------

class TrainingArtifacts:
    """Manages all persisted outputs for a BTRM training run.

    Handles:
        - Streaming JSONL metrics (one line per training step)
        - Checkpoint saving (adapter + head safetensors)
        - Post-training analysis: charts (PNG via PIL) + markdown report
        - Wall time tracking per step and cumulative

    The module is transport-agnostic: it writes files and produces images.
    It does not import torch at module level; torch is only used for checkpoint
    saving (safetensors) and is imported lazily.
    """

    def __init__(
        self,
        output_dir: str | Path,
        head_names: Sequence[str],
        run_name: str = "btrm_training",
    ):
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.head_names = list(head_names)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = self.output_dir / "training_metrics.jsonl"
        self._metrics_file = open(str(self._metrics_path), "w")

        self._steps: list[dict] = []
        self._wall_start = time.perf_counter()
        self._step_times: list[float] = []

    # -----------------------------------------------------------------------
    # Per-step logging
    # -----------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        loss: float,
        accuracy_dict: dict[str, float],
        grad_norm: float,
        lr: float,
        extra_metrics: dict[str, Any] | None = None,
        step_time: float | None = None,
    ):
        """Log a single training step's metrics.

        Args:
            step: Optimizer step index.
            loss: BT loss value.
            accuracy_dict: {head_name: accuracy} for each scoring head.
            grad_norm: Pre-clip gradient norm.
            lr: Current learning rate.
            extra_metrics: Any additional metrics to log (optional).
            step_time: Time for this step in seconds. If None, computed from
                wall clock since last log_step call.
        """
        elapsed = time.perf_counter() - self._wall_start

        if step_time is None and self._step_times:
            step_time = elapsed - sum(self._step_times)
        elif step_time is None:
            step_time = elapsed

        self._step_times.append(step_time)

        entry = {
            "step": step,
            "loss": loss,
            "bt_loss": loss,
            "pre_clip_grad_norm": grad_norm,
            "lr": lr,
            "time_s": step_time,
            "elapsed_s": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for name in self.head_names:
            entry[f"accuracy_{name}"] = accuracy_dict.get(name, 0.0)

        if extra_metrics:
            entry.update(extra_metrics)

        self._steps.append(entry)

        # Stream to JSONL
        self._metrics_file.write(json.dumps(entry, default=str) + "\n")
        self._metrics_file.flush()

    def make_callback(self):
        """Return a callback function compatible with train_btrm_differentiable().

        The callback signature is: callback(step: int, entry: dict)
        where entry is the dict from the training loop containing at minimum:
        step, loss, bt_loss, pre_clip_grad_norm, lr, time_s, accuracy_<head>.
        """
        def _callback(step: int, entry: dict):
            acc_dict = {}
            for name in self.head_names:
                key = f"accuracy_{name}"
                acc_dict[name] = entry.get(key, 0.0)

            # Separate known keys from extra
            known_keys = {
                "step", "loss", "bt_loss", "pre_clip_grad_norm", "grad_norm",
                "lr", "time_s", "pair_weight", "elapsed_s", "timestamp",
            }
            known_keys.update(f"accuracy_{n}" for n in self.head_names)
            extra = {k: v for k, v in entry.items() if k not in known_keys}

            self.log_step(
                step=step,
                loss=entry.get("loss", entry.get("bt_loss", 0.0)),
                accuracy_dict=acc_dict,
                grad_norm=entry.get("pre_clip_grad_norm", 0.0),
                lr=entry.get("lr", 0.0),
                extra_metrics=extra if extra else None,
                step_time=entry.get("time_s"),
            )

        return _callback

    # -----------------------------------------------------------------------
    # Checkpoint saving
    # -----------------------------------------------------------------------

    def save_checkpoint(
        self,
        step: int,
        model,
        adapter_name: str = "rtheta",
    ) -> Path:
        """Save adapter + head state at a checkpoint step.

        Args:
            step: Current training step.
            model: ZImageRLAIF model instance.
            adapter_name: LoRA adapter name to persist.

        Returns:
            Path to the checkpoint directory.
        """
        from src_ii.btrm_lifecycle import persist_btrm

        ckpt_dir = self.output_dir / f"checkpoint_step{step:03d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        persist_btrm(model, adapter_name, str(ckpt_dir), head_names=self.head_names)
        return ckpt_dir

    # -----------------------------------------------------------------------
    # Post-training analysis
    # -----------------------------------------------------------------------

    def generate_analysis(self, run_config: dict[str, Any] | None = None) -> Path:
        """Generate charts and markdown analysis from logged metrics.

        Produces:
            {output_dir}/charts/01_loss_curve.png
            {output_dir}/charts/02_per_head_accuracy.png
            {output_dir}/charts/03_gradient_norms.png
            {output_dir}/charts/04_learning_rate.png
            {output_dir}/charts/05_step_timing.png
            {output_dir}/training_analysis.md

        Args:
            run_config: Optional dict of run configuration to include in the report.

        Returns:
            Path to the generated training_analysis.md.
        """
        # Close the metrics file if still open
        if self._metrics_file and not self._metrics_file.closed:
            self._metrics_file.close()

        data = self._steps
        if not data:
            # Try loading from JSONL on disk
            data = _load_jsonl(self._metrics_path)

        if not data:
            # Nothing to analyze
            report_path = self.output_dir / "training_analysis.md"
            report_path.write_text(f"# {self.run_name}\n\nNo training data recorded.\n")
            return report_path

        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        steps = [d["step"] for d in data]
        losses = [d.get("loss", d.get("bt_loss", 0.0)) for d in data]
        grad_norms = [d.get("pre_clip_grad_norm", 0.0) for d in data]
        lrs = [d.get("lr", 0.0) for d in data]
        times = [d.get("time_s", 0.0) for d in data]

        per_head_accs = {}
        for name in self.head_names:
            per_head_accs[name] = [d.get(f"accuracy_{name}", 0.0) for d in data]

        # Chart 1: Loss curve
        self._chart_loss(steps, losses, charts_dir / "01_loss_curve.png")

        # Chart 2: Per-head accuracy
        self._chart_accuracy(steps, per_head_accs, charts_dir / "02_per_head_accuracy.png")

        # Chart 3: Gradient norms
        self._chart_grad_norms(steps, grad_norms, charts_dir / "03_gradient_norms.png")

        # Chart 4: Learning rate
        self._chart_lr(steps, lrs, charts_dir / "04_learning_rate.png")

        # Chart 5: Step timing
        self._chart_timing(steps, times, charts_dir / "05_step_timing.png")

        # ---- Funfetti diagnostics (Plots A-F) ----
        # Only produced when funfetti metadata is available in the step data.
        funfetti_steps = [d for d in data if "funfetti" in d]
        if funfetti_steps:
            self._chart_resolution_pdf(funfetti_steps, charts_dir / "06_resolution_pdf.png")
            self._chart_aspect_ratio_pdf(funfetti_steps, charts_dir / "07_aspect_ratio_pdf.png")
            self._chart_loss_by_resolution(data, funfetti_steps, charts_dir / "08_metrics_by_resolution.png")
            self._chart_microbatch_pair_count(data, funfetti_steps, charts_dir / "09_microbatch_pairs.png")
            self._chart_context_length(data, funfetti_steps, charts_dir / "10_context_length.png")
            self._chart_flops_normalized_resolution(funfetti_steps, charts_dir / "11_flops_normalized_resolution.png")

        # Write markdown report
        report_path = self.output_dir / "training_analysis.md"
        report = self._build_report(data, steps, losses, grad_norms, lrs, times, per_head_accs, run_config)
        report_path.write_text(report)

        return report_path

    # -----------------------------------------------------------------------
    # Chart rendering (PIL only, no matplotlib)
    # -----------------------------------------------------------------------

    def _chart_loss(self, steps, losses, out_path):
        chart = PILChart()
        chart.set_title(f"{self.run_name}: BT Loss (per-term) vs Step")
        chart.set_labels("Step", "Loss (per-term avg)")

        ema_losses = _ema(losses, alpha=0.1)

        chart.add_scatter(steps, losses, color="#bbddff", label="raw loss", size=2)
        chart.add_line(steps, ema_losses, color="#1155cc", label="EMA(0.1)", line_width=2)
        chart.save(out_path)

    def _chart_accuracy(self, steps, per_head_accs, out_path):
        chart = PILChart()
        chart.set_title(f"{self.run_name}: Per-Head Accuracy vs Step")
        chart.set_labels("Step", "Accuracy")

        colors = ["#1155cc", "#cc7711", "#11aa44", "#cc1111", "#7722cc", "#117777"]
        scatter_colors = ["#aaddff", "#ffddaa", "#aaffcc", "#ffaaaa", "#ddaaff", "#aaffff"]

        for i, name in enumerate(self.head_names):
            accs = per_head_accs[name]
            ci = i % len(colors)
            window = max(5, len(steps) // 20)
            ravg = _running_average(accs, window=window)

            chart.add_scatter(steps, accs, color=scatter_colors[ci], size=2)
            chart.add_line(steps, ravg, color=colors[ci], label=f"{name} ravg({window})", line_width=2)

        chart.save(out_path)

    def _chart_grad_norms(self, steps, grad_norms, out_path):
        chart = PILChart()
        chart.set_title(f"{self.run_name}: Pre-Clip Gradient Norm (log scale)")
        chart.set_labels("Step", "log10(Grad Norm)")
        chart.set_log_y(True)

        ema_gn = _ema(grad_norms, alpha=0.15)

        chart.add_scatter(steps, grad_norms, color="#bbffbb", label="raw", size=2)
        chart.add_line(steps, ema_gn, color="#117711", label="EMA(0.15)", line_width=2)
        chart.save(out_path)

    def _chart_lr(self, steps, lrs, out_path):
        chart = PILChart()
        chart.set_title(f"{self.run_name}: Learning Rate Schedule")
        chart.set_labels("Step", "LR")

        chart.add_line(steps, lrs, color="#cc1111", label="LR", line_width=2)
        chart.save(out_path)

    def _chart_timing(self, steps, times, out_path):
        chart = PILChart()
        chart.set_title(f"{self.run_name}: Step Timing")
        chart.set_labels("Step", "Time (s)")

        chart.add_line(steps, times, color="#1155cc", label="time/step", line_width=2)
        chart.save(out_path)

    # -----------------------------------------------------------------------
    # Funfetti diagnostic charts (Plots A-E)
    # -----------------------------------------------------------------------

    def _chart_resolution_pdf(self, funfetti_steps, out_path):
        """Plot A: Empirical PDF of resolution scale trained on.

        Shows how many pairs at each pixel count were consumed during training.
        Validates FLOPS-weighted sampling: small images should dominate pair count.
        """
        from collections import Counter

        pixel_counts = Counter()
        for d in funfetti_steps:
            fm = d["funfetti"]
            for res in fm.get("resolutions", []):
                pixels = res["pixels"]
                pixel_counts[pixels] += 1

        if not pixel_counts:
            return

        total = sum(pixel_counts.values())
        # Sort by pixel count
        sorted_pixels = sorted(pixel_counts.keys())
        labels_px = sorted_pixels
        proportions = [pixel_counts[px] / total for px in sorted_pixels]

        chart = PILChart()
        chart.set_title(f"{self.run_name}: Resolution PDF (image count)")
        chart.set_labels("Resolution (pixels)", "Proportion")

        # Use categorical x positions for clarity
        xs = list(range(len(sorted_pixels)))
        chart.add_scatter(xs, proportions, color="#1155cc", label="proportion", size=6)

        # Connect with lines
        if len(xs) > 1:
            chart.add_line(xs, proportions, color="#1155cc", label="", line_width=2)

        chart.save(out_path)

        # Also save the raw data as JSON for the essay
        pdf_data = {
            "total_images": total,
            "per_resolution": {
                str(px): {"count": pixel_counts[px], "proportion": pixel_counts[px] / total}
                for px in sorted_pixels
            }
        }
        pdf_path = out_path.parent.parent / "resolution_pdf.json"
        import json
        with open(str(pdf_path), "w") as f:
            json.dump(pdf_data, f, indent=2)

    def _chart_aspect_ratio_pdf(self, funfetti_steps, out_path):
        """Plot B: Empirical PDF of aspect ratio (W/H) in training data."""
        from collections import Counter

        ratio_counts = Counter()
        for d in funfetti_steps:
            fm = d["funfetti"]
            for res in fm.get("resolutions", []):
                w, h = res["width"], res["height"]
                ratio = round(w / max(h, 1), 2)
                ratio_counts[ratio] += 1

        if not ratio_counts:
            return

        total = sum(ratio_counts.values())
        sorted_ratios = sorted(ratio_counts.keys())
        proportions = [ratio_counts[r] / total for r in sorted_ratios]

        chart = PILChart()
        chart.set_title(f"{self.run_name}: Aspect Ratio PDF (W/H)")
        chart.set_labels("Aspect Ratio (W/H)", "Proportion")

        chart.add_scatter(sorted_ratios, proportions, color="#cc7711", label="proportion", size=6)
        if len(sorted_ratios) > 1:
            chart.add_line(sorted_ratios, proportions, color="#cc7711", label="", line_width=2)
        chart.save(out_path)

    def _chart_loss_by_resolution(self, data, funfetti_steps, out_path):
        """Plot C: Loss and grad norm distribution by resolution bucket.

        For each resolution present, plots the per-step loss and grad norm.
        """
        from collections import defaultdict

        # Build resolution -> list of (step, loss, grad_norm, time) mappings
        res_to_metrics = defaultdict(list)

        for d in data:
            fm = d.get("funfetti")
            if not fm:
                continue
            step = d["step"]
            loss = d.get("loss", d.get("bt_loss", 0.0))
            gnorm = d.get("pre_clip_grad_norm", 0.0)
            step_time = d.get("time_s", 0.0)

            # Determine dominant resolution for this step
            pixel_counts = {}
            for res in fm.get("resolutions", []):
                px = res["pixels"]
                pixel_counts[px] = pixel_counts.get(px, 0) + 1
            if pixel_counts:
                dominant_px = max(pixel_counts, key=pixel_counts.get)
                res_to_metrics[dominant_px].append({
                    "step": step, "loss": loss, "gnorm": gnorm, "time": step_time,
                })

        if not res_to_metrics:
            return

        chart = PILChart()
        chart.set_title(f"{self.run_name}: Loss by Dominant Resolution")
        chart.set_labels("Step", "Loss (per-term avg)")

        colors = ["#1155cc", "#cc7711", "#11aa44", "#cc1111", "#7722cc", "#117777"]
        for i, px in enumerate(sorted(res_to_metrics.keys())):
            entries = res_to_metrics[px]
            xs = [e["step"] for e in entries]
            ys = [e["loss"] for e in entries]
            ci = i % len(colors)
            label = f"{px:,}px ({len(entries)})"
            chart.add_scatter(xs, ys, color=colors[ci], label=label, size=3)

        chart.save(out_path)

    def _chart_microbatch_pair_count(self, data, funfetti_steps, out_path):
        """Plot D: Pairs per microbatch across training.

        Shows how many pairs were in each microbatch. With pairs_per_pack=2
        and grad_accum_steps=2, each macrobatch should have 2 microbatches
        of 2 pairs = 4 pairs total.
        """
        xs = []
        ys = []
        for d in data:
            fm = d.get("funfetti")
            if not fm:
                continue
            step = d["step"]
            for mi, mb in enumerate(fm.get("microbatches", [])):
                xs.append(step + mi * 0.3)  # offset for visibility
                ys.append(mb["n_pairs"])

        if not xs:
            return

        chart = PILChart()
        chart.set_title(f"{self.run_name}: Pairs per Microbatch")
        chart.set_labels("Step", "Pairs per Microbatch")

        chart.add_scatter(xs, ys, color="#1155cc", label="pairs/micro", size=4)

        # Also show total pairs per macrobatch
        macro_xs = []
        macro_ys = []
        for d in data:
            fm = d.get("funfetti")
            if fm:
                macro_xs.append(d["step"])
                macro_ys.append(fm["total_pairs"])

        if macro_xs:
            chart.add_line(macro_xs, macro_ys, color="#cc1111", label="total/macro", line_width=2)

        chart.save(out_path)

    def _chart_context_length(self, data, funfetti_steps, out_path):
        """Plot E: Total context length per microbatch.

        Validates context-length-based accumulation. Two microbatches with small
        images should have very different context lengths than one with large.
        """
        xs = []
        ys = []
        for d in data:
            fm = d.get("funfetti")
            if not fm:
                continue
            step = d["step"]
            for mi, mb in enumerate(fm.get("microbatches", [])):
                xs.append(step + mi * 0.3)
                ys.append(mb["total_context_len"])

        if not xs:
            return

        chart = PILChart()
        chart.set_title(f"{self.run_name}: Context Length per Microbatch")
        chart.set_labels("Step", "Total Context Tokens")

        chart.add_scatter(xs, ys, color="#11aa44", label="tokens/micro", size=4)

        # Also show total context length per macrobatch
        macro_xs = []
        macro_ys = []
        for d in data:
            fm = d.get("funfetti")
            if fm:
                macro_xs.append(d["step"])
                macro_ys.append(fm["total_context_len"])

        if macro_xs:
            chart.add_line(macro_xs, macro_ys, color="#117711", label="total/macro", line_width=2)

        chart.save(out_path)

    def _chart_flops_normalized_resolution(self, funfetti_steps, out_path):
        """Plot F: FLOPS-normalized resolution PDF.

        Shows count_per_resolution * flops_ratio_per_resolution, normalized
        to sum to 1. This directly validates the 33/67 megapixel/small FLOPS
        allocation target. Unlike the raw image-count PDF (Plot A), this
        chart shows WHERE THE COMPUTE WAS SPENT, not how many images were
        drawn.

        The 33/67 target means this chart should show ~33% of the bar height
        on megapixel resolutions (>= 1024^2 pixels) and ~67% on smaller ones.
        """
        from collections import Counter

        pixel_counts = Counter()
        # Also track (width, height) for each pixel area to compute FLOPS ratio
        pixel_to_wh: dict[int, tuple[int, int]] = {}

        for d in funfetti_steps:
            fm = d["funfetti"]
            for res in fm.get("resolutions", []):
                pixels = res["pixels"]
                pixel_counts[pixels] += 1
                if pixels not in pixel_to_wh:
                    pixel_to_wh[pixels] = (res["width"], res["height"])

        if not pixel_counts:
            return

        # Compute FLOPS-weighted proportions
        from src_ii.flops_sampling import _attention_flops_ratio, _MEGAPIXEL_THRESHOLD

        sorted_pixels = sorted(pixel_counts.keys())
        flops_weighted = []
        for px in sorted_pixels:
            w, h = pixel_to_wh[px]
            flops_ratio = _attention_flops_ratio(w, h)
            flops_weighted.append(pixel_counts[px] * flops_ratio)

        total_flops = sum(flops_weighted)
        if total_flops <= 0:
            return

        proportions = [fw / total_flops for fw in flops_weighted]

        # Compute megapixel vs small FLOPS fractions
        mega_flops = sum(
            fw for px, fw in zip(sorted_pixels, flops_weighted)
            if px >= _MEGAPIXEL_THRESHOLD
        )
        small_flops = sum(
            fw for px, fw in zip(sorted_pixels, flops_weighted)
            if px < _MEGAPIXEL_THRESHOLD
        )
        mega_pct = mega_flops / total_flops * 100 if total_flops > 0 else 0
        small_pct = small_flops / total_flops * 100 if total_flops > 0 else 0

        chart = PILChart()
        chart.set_title(
            f"{self.run_name}: FLOPS-Normalized Resolution PDF "
            f"(mega={mega_pct:.1f}% / small={small_pct:.1f}%)"
        )
        chart.set_labels("Resolution (pixels)", "FLOPS Proportion")

        # Use categorical x positions for clarity
        xs = list(range(len(sorted_pixels)))
        chart.add_scatter(xs, proportions, color="#cc1111", label="FLOPS proportion", size=6)
        if len(xs) > 1:
            chart.add_line(xs, proportions, color="#cc1111", label="", line_width=2)

        chart.save(out_path)

        # Save the FLOPS-normalized data as JSON
        flops_pdf_data = {
            "total_flops_weighted": total_flops,
            "megapixel_flops_pct": mega_pct,
            "small_flops_pct": small_pct,
            "per_resolution": {
                str(px): {
                    "count": pixel_counts[px],
                    "flops_ratio": _attention_flops_ratio(*pixel_to_wh[px]),
                    "flops_weighted": flops_weighted[i],
                    "flops_proportion": proportions[i],
                    "is_megapixel": px >= _MEGAPIXEL_THRESHOLD,
                }
                for i, px in enumerate(sorted_pixels)
            },
        }
        pdf_path = out_path.parent.parent / "flops_normalized_resolution.json"
        import json as _json
        with open(str(pdf_path), "w") as f:
            _json.dump(flops_pdf_data, f, indent=2)

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------

    def _build_report(self, data, steps, losses, grad_norms, lrs, times, per_head_accs, run_config):
        n = len(data)
        total_time = sum(times)
        wall_total = time.perf_counter() - self._wall_start

        ema_losses = _ema(losses, alpha=0.1)
        best_ema_idx = ema_losses.index(min(ema_losses)) if ema_losses else 0
        best_ema_val = ema_losses[best_ema_idx] if ema_losses else 0.0

        # Per-head overall and last-20 accuracy
        last_n = min(20, n)
        head_stats = {}
        for name in self.head_names:
            accs = per_head_accs[name]
            head_stats[name] = {
                "overall": _mean(accs),
                "last_n": _mean(accs[-last_n:]) if accs else 0.0,
            }

        # Config section
        config_section = ""
        if run_config:
            config_lines = "\n".join(f"- **{k}**: {v}" for k, v in run_config.items())
            config_section = f"\n## Run Configuration\n\n{config_lines}\n"

        # Head accuracy table rows
        head_rows = ""
        for name in self.head_names:
            s = head_stats[name]
            head_rows += f"| {name} | {s['overall']:.0%} | {s['last_n']:.0%} |\n"

        report = f"""# Training Analysis: {self.run_name}

**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
**Steps completed:** {n}
**Total training time:** {total_time:.1f}s ({total_time / 60:.1f} min)
**Wall time:** {wall_total:.1f}s ({wall_total / 60:.1f} min)
{config_section}
## Loss Trajectory

| Metric | Value |
|--------|-------|
| Initial BT loss | {losses[0]:.4f} |
| Final BT loss | {losses[-1]:.4f} |
| Minimum BT loss | {min(losses):.6f} (step {losses.index(min(losses))}) |
| Maximum BT loss | {max(losses):.4f} (step {losses.index(max(losses))}) |
| EMA(0.1) best | {best_ema_val:.4f} (step {best_ema_idx}) |
| Mean loss | {_mean(losses):.4f} |
| Std loss | {_std(losses):.4f} |

## Per-Head Accuracy

| Head | Overall | Last {last_n} Steps |
|------|---------|-----------|
{head_rows}
## Gradient Norm Analysis

| Metric | Value |
|--------|-------|
| Mean | {_mean(grad_norms):.3f} |
| Max | {max(grad_norms):.3f} (step {grad_norms.index(max(grad_norms))}) |
| Min | {min(grad_norms):.6f} |
| Steps with norm > 10 | {sum(1 for g in grad_norms if g > 10.0)} |

## Learning Rate

| Metric | Value |
|--------|-------|
| Initial LR | {lrs[0]:.2e} |
| Peak LR | {max(lrs):.2e} |
| Final LR | {lrs[-1]:.2e} |

## Step Timing

| Metric | Value |
|--------|-------|
| Step 0 (compilation) | {times[0]:.1f}s |
| Mean steady-state (steps 1+) | {_mean(times[1:]):.1f}s |
| Total training time | {total_time:.1f}s ({total_time / 60:.1f} min) |

## Charts

Generated in `charts/`:
- `01_loss_curve.png` -- BT loss raw scatter + EMA(0.1)
- `02_per_head_accuracy.png` -- Per-head accuracy with running average
- `03_gradient_norms.png` -- Pre-clip gradient norm (log scale)
- `04_learning_rate.png` -- LR schedule
- `05_step_timing.png` -- Seconds per step
"""
        # Check for funfetti charts
        funfetti_steps = [d for d in data if "funfetti" in d]
        if funfetti_steps:
            report += """
## Funfetti Diagnostic Charts

- `06_resolution_pdf.png` -- Empirical PDF of resolution scale trained on (image count)
- `07_aspect_ratio_pdf.png` -- Aspect ratio PDF (W/H)
- `08_metrics_by_resolution.png` -- Loss distribution by resolution bucket
- `09_microbatch_pairs.png` -- Pairs per microbatch across training
- `10_context_length.png` -- Context length per microbatch
- `11_flops_normalized_resolution.png` -- FLOPS-normalized resolution PDF (validates 33/67 split)

### Resolution Summary

"""
            from collections import Counter
            pixel_counts = Counter()
            for d in funfetti_steps:
                fm = d["funfetti"]
                for res in fm.get("resolutions", []):
                    pixel_counts[res["pixels"]] += 1
            total_img = sum(pixel_counts.values())
            report += "| Resolution (pixels) | Count | Proportion |\n"
            report += "|---------------------|-------|------------|\n"
            for px in sorted(pixel_counts.keys()):
                ct = pixel_counts[px]
                prop = ct / max(total_img, 1)
                report += f"| {px:,} | {ct} | {prop:.1%} |\n"

            # Pair count summary
            total_pairs = sum(d["funfetti"]["total_pairs"] for d in funfetti_steps)
            total_nfes = sum(d["funfetti"]["total_nfes"] for d in funfetti_steps)
            report += f"\n**Total pairs processed:** {total_pairs}\n"
            report += f"**Total NFEs:** {total_nfes}\n"
        return report

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def close(self):
        """Close the metrics file."""
        if self._metrics_file and not self._metrics_file.closed:
            self._metrics_file.close()

    def __del__(self):
        self.close()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    if not path.exists():
        return []
    rows = []
    with open(str(path)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
