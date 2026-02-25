"""
Generic image composite construction using pure PIL.

Provides two layout functions for assembling multi-panel output figures:

  build_comparison_composite -- horizontal panel row + chart row + optional stats text.
  build_grid_composite        -- 2-D grid with column/row labels.

No torch, numpy, or src_ii imports. Only PIL.Image and PIL.ImageDraw are used,
so this module has no GPU dependencies and loads instantly.
"""

from PIL import Image, ImageDraw

# Pixels between adjacent panels / cells and between rows.
_GAP = 5
# Height reserved at the top of every composite for the title string.
_HEADER_H = 30
# Vertical spacing between consecutive stats text lines.
_LINE_H = 15


def build_comparison_composite(
    image_panels: list["Image.Image"],
    panel_labels: list[str],
    charts: list["Image.Image"],
    title: str,
    stats_lines: list[str] | None = None,
    target_row_height: int = 256,
) -> "Image.Image":
    """Build a multi-row composite: image panels, charts, optional stats text.

    Row 1: image_panels scaled to target_row_height, side by side with 5px gaps.
    Row 2: charts side by side (unscaled width, scaled to match row 1 total width).
    Row 3 (if stats_lines): text lines rendered at 15px line height.

    Labels appear centered above each panel in row 1.
    Title appears at top of composite.

    Args:
        image_panels: Ordered list of PIL images to display as panel columns.
        panel_labels: One label string per panel. Centered above each panel.
        charts: Ordered list of PIL chart images placed in row 2.
        title: String rendered in the 30px header at the top of the composite.
        stats_lines: Optional list of plain-text lines appended as row 3.
            Each line occupies 15px of vertical space.
        target_row_height: Height in pixels each panel image is scaled to.
            Charts are scaled vertically to match the total row 1 height.

    Returns:
        PIL.Image.Image in RGB mode on a white background.
    """
    n_panels = len(image_panels)

    # --- Row 1: scale panels to target_row_height preserving aspect ratio ---
    scaled_panels: list[Image.Image] = []
    for img in image_panels:
        orig_w, orig_h = img.size
        new_w = max(1, int(orig_w * target_row_height / orig_h))
        scaled_panels.append(img.resize((new_w, target_row_height), Image.LANCZOS))

    # Label row height: 15px above the panel images, inside the header after title.
    label_row_h = _LINE_H

    row1_w = sum(p.size[0] for p in scaled_panels) + _GAP * max(0, n_panels - 1)
    row1_h = target_row_height + label_row_h  # panels + label strip

    # --- Row 2: arrange charts, then scale width to match row 1 ---
    if charts:
        chart_native_w = sum(c.size[0] for c in charts) + _GAP * max(0, len(charts) - 1)
        # First, create native-width chart column image.
        chart_native_h = max(c.size[1] for c in charts)
        chart_row_native = Image.new("RGB", (chart_native_w, chart_native_h), "white")
        cx = 0
        for c in charts:
            chart_row_native.paste(c, (cx, 0))
            cx += c.size[0] + _GAP

        # Scale chart row width to match row 1 width (height follows proportionally).
        if chart_native_w != row1_w and chart_native_w > 0:
            scale = row1_w / chart_native_w
            new_h = max(1, int(chart_native_h * scale))
            chart_row = chart_row_native.resize((row1_w, new_h), Image.LANCZOS)
        else:
            chart_row = chart_row_native
        row2_h = chart_row.size[1]
    else:
        chart_row = None
        row2_h = 0

    # --- Row 3: stats text ---
    row3_h = len(stats_lines) * _LINE_H if stats_lines else 0

    # --- Total composite dimensions ---
    total_w = row1_w
    total_h = _HEADER_H + row1_h + (_GAP + row2_h if row2_h else 0) + (_GAP + row3_h if row3_h else 0)

    composite = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(composite)

    # Title in header.
    draw.text((10, 8), title, fill="black")

    # Paste row 1: label above each panel.
    y_row1 = _HEADER_H
    x = 0
    for panel, label in zip(scaled_panels, panel_labels):
        pw = panel.size[0]
        # Center label text (estimate 6px per character at default font).
        label_x = x + max(0, (pw - len(label) * 6) // 2)
        draw.text((label_x, y_row1 + 1), label, fill="black")
        composite.paste(panel, (x, y_row1 + label_row_h))
        x += pw + _GAP

    # Paste row 2: charts.
    if chart_row is not None:
        y_row2 = _HEADER_H + row1_h + _GAP
        composite.paste(chart_row, (0, y_row2))

    # Render row 3: stats lines.
    if stats_lines:
        y_text = _HEADER_H + row1_h + (_GAP + row2_h if row2_h else 0) + _GAP
        for line in stats_lines:
            draw.text((10, y_text), line, fill=(60, 60, 60))
            y_text += _LINE_H

    return composite


def build_grid_composite(
    grid: list[list["Image.Image"]],  # grid[col][row]
    col_labels: list[str],
    row_labels: list[str],
    title: str,
    target_row_height: int = 256,
) -> "Image.Image":
    """Build a grid composite for cross-resolution comparison.

    Columns represent resolutions; rows represent configurations. Each cell is
    thumbnailed to target_row_height preserving aspect ratio. Column labels are
    rendered centered above each column; row labels are rendered at the left of
    each row. The title is placed in the 30px header at the top.

    Args:
        grid: Nested list indexed as grid[col][row]. All columns must have the
            same number of rows. Missing or None cells are left blank (white).
        col_labels: One label per column. Centered above the column.
        row_labels: One label per row. Rendered left of the row.
        title: String rendered in the 30px header.
        target_row_height: Each cell is thumbnailed to this height.

    Returns:
        PIL.Image.Image in RGB mode on a white background.
    """
    n_cols = len(grid)
    n_rows = max(len(col) for col in grid) if grid else 0

    # --- Thumbnail all cells ---
    # cells[col][row] = (scaled PIL image | None)
    cells: list[list[Image.Image | None]] = []
    for col_idx, col in enumerate(grid):
        col_cells: list[Image.Image | None] = []
        for row_idx in range(n_rows):
            if row_idx < len(col) and col[row_idx] is not None:
                img = col[row_idx]
                orig_w, orig_h = img.size
                new_w = max(1, int(orig_w * target_row_height / orig_h))
                col_cells.append(img.resize((new_w, target_row_height), Image.LANCZOS))
            else:
                col_cells.append(None)
        cells.append(col_cells)

    # --- Column widths = max cell width in that column ---
    col_widths: list[int] = []
    for col_idx in range(n_cols):
        max_w = 1
        for row_idx in range(n_rows):
            cell = cells[col_idx][row_idx]
            if cell is not None:
                max_w = max(max_w, cell.size[0])
        col_widths.append(max_w)

    # Row height is uniform (target_row_height) since all cells are thumbnailed to it.
    cell_h = target_row_height

    # Estimate label strip heights.
    col_label_h = _LINE_H  # above each column's top row
    row_label_w = 80       # pixels reserved at left for row labels

    grid_content_w = row_label_w + sum(col_widths) + _GAP * max(0, n_cols - 1)
    grid_content_h = col_label_h + n_rows * cell_h + _GAP * max(0, n_rows - 1)

    total_w = grid_content_w
    total_h = _HEADER_H + grid_content_h

    composite = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(composite)

    # Title.
    draw.text((10, 8), title, fill="black")

    # Column labels (centered above each column).
    x = row_label_w
    for col_idx in range(n_cols):
        cw = col_widths[col_idx]
        label = col_labels[col_idx] if col_idx < len(col_labels) else ""
        label_x = x + max(0, (cw - len(label) * 6) // 2)
        draw.text((label_x, _HEADER_H + 1), label, fill="black")
        x += cw + _GAP

    # Paste cells row by row.
    for row_idx in range(n_rows):
        y = _HEADER_H + col_label_h + row_idx * (cell_h + _GAP)

        # Row label at the left margin.
        row_label = row_labels[row_idx] if row_idx < len(row_labels) else ""
        draw.text((2, y + cell_h // 2 - 6), row_label, fill="black")

        x = row_label_w
        for col_idx in range(n_cols):
            cell = cells[col_idx][row_idx]
            if cell is not None:
                composite.paste(cell, (x, y))
            x += col_widths[col_idx] + _GAP

    return composite
