"""
Score vs logSNR chart rendering using pure PIL.

Provides draw_score_chart(), which renders an N-series line chart of BTRM
head scores against log signal-to-noise ratio. Intended for comparing policy
adapter configurations (ref, r_theta, ablations, etc.) across a denoising
trajectory.

No torch, numpy, or src_ii imports. Only PIL.Image and PIL.ImageDraw are used,
so this module has no GPU dependencies and loads instantly.
"""

from PIL import Image, ImageDraw


def draw_score_chart(
    logsnrs: list[float],
    named_series: dict[str, dict],
    head_name: str,
    chart_w: int = 800,
    chart_h: int = 400,
) -> "Image.Image":
    """Draw a score vs logSNR chart for N labeled series.

    Args:
        logsnrs: Shared X-axis values (log signal-to-noise ratio), one per
            trajectory step. Must have the same length as each series' values.
        named_series: Ordered dict mapping label strings to series dicts.
            Each series dict must have:
                "values": list[float]  -- Y values, same length as logsnrs
                "color":  tuple[int, int, int]  -- RGB fill color for this series
            The first series is treated as the reference for delta annotations.
        head_name: Reward head name used in the chart title.
        chart_w: Total image width in pixels. Default 800.
        chart_h: Total image height in pixels. Default 400.

    Returns:
        PIL.Image.Image in RGB mode containing the rendered chart.

    Layout:
        - Title centered at top: "{head_name}: score vs logSNR"
        - X-axis label centered at bottom: "logSNR (noisy -> clean ->)"
        - Left margin: Y-axis tick labels (5 evenly spaced grid lines)
        - Top-left of plot area: legend (colored box + label per series)
        - Top-right of plot area: final value + delta annotations per series
        - Plot body: line (2px) with dots (3px radius) per series
        - Light gray horizontal grid lines at the 5 Y tick positions
        - Plot area border in black
    """
    margin_l, margin_r, margin_t, margin_b = 70, 20, 40, 50
    plot_w = chart_w - margin_l - margin_r
    plot_h = chart_h - margin_t - margin_b

    img = Image.new("RGB", (chart_w, chart_h), "white")
    draw = ImageDraw.Draw(img)

    # --- Axis range computation ---
    all_values: list[float] = []
    for s in named_series.values():
        all_values.extend(s["values"])

    x_min, x_max = min(logsnrs), max(logsnrs)
    y_min, y_max = min(all_values), max(all_values)

    y_range = y_max - y_min
    y_pad = max(0.05, y_range * 0.15)
    y_min -= y_pad
    y_max += y_pad

    x_range = x_max - x_min
    x_pad = max(0.1, x_range * 0.05)
    x_min -= x_pad
    x_max += x_pad

    def to_px(xv: float, yv: float) -> tuple[int, int]:
        px = margin_l + int((xv - x_min) / (x_max - x_min) * plot_w)
        py = margin_t + int((1.0 - (yv - y_min) / (y_max - y_min)) * plot_h)
        return px, py

    # --- Grid lines and Y-axis labels ---
    n_grid = 5
    for i in range(n_grid):
        frac = i / (n_grid - 1)
        yv = y_min + frac * (y_max - y_min)
        _, py = to_px(x_min, yv)
        draw.line([(margin_l, py), (margin_l + plot_w, py)], fill=(220, 220, 220))
        draw.text((5, py - 6), f"{yv:.3f}", fill="black")

    # --- Plot area border ---
    draw.rectangle(
        [margin_l, margin_t, margin_l + plot_w, margin_t + plot_h],
        outline="black",
    )

    # --- Series: lines and dots ---
    for series_info in named_series.values():
        values = series_info["values"]
        color = series_info["color"]
        points = [to_px(logsnrs[i], values[i]) for i in range(len(logsnrs))]
        for i in range(len(points) - 1):
            draw.line([points[i], points[i + 1]], fill=color, width=2)
        for p in points:
            r = 3
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)

    # --- Legend (top-left of plot area) ---
    legend_x = margin_l + 10
    legend_y = margin_t + 5
    box_w, box_h = 15, 10
    row_h = 15
    for label, series_info in named_series.items():
        color = series_info["color"]
        draw.rectangle(
            [legend_x, legend_y, legend_x + box_w, legend_y + box_h],
            fill=color,
        )
        draw.text((legend_x + box_w + 5, legend_y - 2), label, fill=color)
        legend_y += row_h

    # --- Annotations (top-right of plot area) ---
    # "final {label}={value:.4f}" per series, then "d({label})={delta:+.4f}" per non-first
    annot_x = margin_l + plot_w - 185
    annot_y = margin_t + 5
    labels_list = list(named_series.keys())
    first_label = labels_list[0]
    first_final = named_series[first_label]["values"][-1]

    for label, series_info in named_series.items():
        color = series_info["color"]
        final_val = series_info["values"][-1]
        draw.text(
            (annot_x, annot_y),
            f"final {label}={final_val:.4f}",
            fill=color,
        )
        annot_y += 15

    for label in labels_list[1:]:
        final_val = named_series[label]["values"][-1]
        delta = final_val - first_final
        draw.text(
            (annot_x, annot_y),
            f"d({label})={delta:+.4f}",
            fill="black",
        )
        annot_y += 15

    # --- Title ---
    title = f"{head_name}: score vs logSNR"
    draw.text((chart_w // 2 - len(title) * 3, 5), title, fill="black")

    # --- X-axis label ---
    x_label = "logSNR (noisy -> clean ->)"
    draw.text(
        (chart_w // 2 - len(x_label) * 3, chart_h - 18),
        x_label,
        fill="gray",
    )

    return img
