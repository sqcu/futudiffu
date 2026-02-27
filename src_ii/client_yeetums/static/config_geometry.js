/* config_geometry.js — Distributional config space geometry visualization.
 *
 * Three-layer display:
 *   1. Per-factor figures — shape encodes distribution type, AREA encodes
 *      relative log-volume (proportional to the largest factor), fill opacity
 *      encodes per-factor exploration (k/cardinality). Glow/pulse on change.
 *   2. Overview strip — thin bar showing all factors at full scale, with the
 *      k marker and a viewport highlight showing the zoomed region.
 *   3. Detail strip — zoomed bar connected to the overview by frustum lines.
 *      A zoom slider crops the top-p% of volume (largest factors), letting the
 *      user appreciate the full PRNG seed scale while perceiving relative
 *      cardinality among the smaller tuning parameters.
 *
 * Data flow: config change -> debounce -> POST /api/config/volumes -> update.
 * Zoom slider: immediate rebuild from cached data (no network round-trip).
 */

const ConfigGeometry = (() => {

  const PALETTE = [
    '#e879f9', // magenta
    '#4a9eff', // accent blue
    '#22d3ee', // cyan
    '#4ade80', // green
    '#fbbf24', // amber
    '#fb923c', // orange
    '#f87171', // red
    '#a78bfa', // violet
  ];

  // Animation constants
  const TAU = 1.5;         // decay time constant (seconds)
  const P_EXP = 2.5;       // polynomial exponent
  const REST_PERIOD = 3.0;  // perimeter pulse rest period (seconds)
  const EXCITE_PERIOD = 0.4;
  const PEAK_BLUR = 6;
  const MAX_R = 42;        // max figure radius
  const MIN_R = 8;         // min figure radius

  // Layout constants
  const PAD = 12;
  const FIG_CY = 46;
  const FIG_WIDTH = 100;
  const OVERVIEW_Y = 92;
  const OVERVIEW_H = 5;
  const DETAIL_Y = 110;
  const DETAIL_H = 16;    // taller for per-factor coverage fill levels
  const SUMMARY_Y = 132;
  const TOTAL_H = 144;

  let container = null;
  let svgEl = null;
  let figures = [];
  let prevVolumes = [];
  let abortCtrl = null;
  let debounceTimer = null;
  let rafId = null;

  // Zoom state
  let zoomLevel = 0;       // 0.0 = full view, 1.0 = max zoom (crops largest factors)
  let autoZoomTarget = 0;
  let cachedSorted = [];   // volumes sorted by log_volume descending, with .color
  let cachedK = 1;
  let cachedTotalLog = 0;
  let stripGroup = null;
  let zoomRow = null;
  let zoomSlider = null;

  // ---------------------------------------------------------------------------
  // Shape generators
  // ---------------------------------------------------------------------------

  function hexagonPoints(cx, cy, r) {
    const pts = [];
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 2;
      pts.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
    }
    return pts.join(' ');
  }

  function squarePoints(cx, cy, r) {
    const s = r * 0.85;
    return `${cx - s},${cy - s} ${cx + s},${cy - s} ${cx + s},${cy + s} ${cx - s},${cy + s}`;
  }

  function shapeForKind(kind) {
    if (kind === 'enum' || kind === 'cat' || kind === 'weighted_cat') return 'hexagon';
    if (kind === 'range_int' || kind === 'range_stepped' || kind === 'log_uniform_stepped') return 'square';
    return 'circle';
  }

  // ---------------------------------------------------------------------------
  // Radius computation — area proportional to log_volume / max_log_volume
  // ---------------------------------------------------------------------------

  function computeRadii(volumes) {
    if (volumes.length === 0) return [];
    const logVols = volumes.map(v => v.log_volume);
    const maxLog = Math.max(...logVols);
    if (maxLog <= 0) return volumes.map(() => MIN_R);
    return logVols.map(l => Math.max(MIN_R, MAX_R * Math.sqrt(Math.max(0, l) / maxLog)));
  }

  // ---------------------------------------------------------------------------
  // Perimeter point for pulse dot
  // ---------------------------------------------------------------------------

  function perimeterPoint(shape, cx, cy, r, angle) {
    if (shape === 'circle') {
      return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
    }
    if (shape === 'hexagon') {
      const n = 6;
      const seg = (angle % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
      const segIdx = Math.floor(seg / (2 * Math.PI / n));
      const segFrac = (seg - segIdx * (2 * Math.PI / n)) / (2 * Math.PI / n);
      const a1 = (Math.PI / 3) * segIdx - Math.PI / 2;
      const a2 = (Math.PI / 3) * ((segIdx + 1) % n) - Math.PI / 2;
      const x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
      const x2 = cx + r * Math.cos(a2), y2 = cy + r * Math.sin(a2);
      return { x: x1 + (x2 - x1) * segFrac, y: y1 + (y2 - y1) * segFrac };
    }
    // square
    const s = r * 0.85;
    const perim = 8 * s;
    let d = ((angle % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI)) / (2 * Math.PI) * perim;
    if (d < 2 * s) return { x: cx - s + d, y: cy - s };
    d -= 2 * s;
    if (d < 2 * s) return { x: cx + s, y: cy - s + d };
    d -= 2 * s;
    if (d < 2 * s) return { x: cx + s - d, y: cy + s };
    d -= 2 * s;
    return { x: cx - s, y: cy + s - d };
  }

  // ---------------------------------------------------------------------------
  // SVG helpers
  // ---------------------------------------------------------------------------

  function svgNS() { return 'http://www.w3.org/2000/svg'; }

  function createSVGElement(tag, attrs) {
    const el = document.createElementNS(svgNS(), tag);
    for (const [k, v] of Object.entries(attrs)) {
      el.setAttribute(k, v);
    }
    return el;
  }

  // ---------------------------------------------------------------------------
  // Compact number formatting
  // ---------------------------------------------------------------------------

  function formatNum(n) {
    if (n >= 1e12) return (n / 1e12).toPrecision(3) + 'T';
    if (n >= 1e9)  return (n / 1e9).toPrecision(3) + 'B';
    if (n >= 1e6)  return (n / 1e6).toPrecision(3) + 'M';
    if (n >= 1e3)  return Math.round(n / 1e3) + 'K';
    return Math.round(n).toString();
  }

  // ---------------------------------------------------------------------------
  // Expected marginal coverage: coupon-collector per-factor formula
  // E[coverage_i] = 1 - ((N_i - 1) / N_i)^k
  // ---------------------------------------------------------------------------

  function expectedCoverage(cardinality, k) {
    if (cardinality <= 1) return 1.0;
    if (k <= 0) return 0.0;
    // For very large N, (1-1/N)^k ≈ exp(-k/N) — use log to avoid precision loss
    if (cardinality > 1e6) {
      return 1 - Math.exp(-k / cardinality);
    }
    return 1 - Math.pow((cardinality - 1) / cardinality, k);
  }

  // ---------------------------------------------------------------------------
  // Auto-zoom: target zoom level where second-largest factor ≥ 25% of detail
  // ---------------------------------------------------------------------------

  function computeAutoZoom(sorted, totalLogVol) {
    if (sorted.length < 2 || totalLogVol <= 0) return 0;

    // Want: sorted[1].log_volume / viewBits >= 0.25
    // So viewBits <= 4 * sorted[1].log_volume
    const targetViewBits = Math.min(totalLogVol, 4 * sorted[1].log_volume);
    const targetCropBits = totalLogVol - targetViewBits;

    const minViewBits = sorted[sorted.length - 1].log_volume * 1.5;
    const maxCropBits = Math.max(0, totalLogVol - minViewBits);

    if (maxCropBits <= 0) return 0;
    return Math.min(1.0, Math.max(0, targetCropBits / maxCropBits));
  }

  // ---------------------------------------------------------------------------
  // Compute viewport cropBits from zoom level
  // ---------------------------------------------------------------------------

  function getCropBits(sorted, totalLogVol) {
    if (sorted.length < 2 || totalLogVol <= 0) return 0;
    const minViewBits = sorted[sorted.length - 1].log_volume * 1.5;
    const maxCropBits = Math.max(0, totalLogVol - minViewBits);
    return zoomLevel * maxCropBits;
  }

  // ---------------------------------------------------------------------------
  // Build one figure group
  // ---------------------------------------------------------------------------

  function buildFigure(vol, index, cx, cy, radius) {
    const color = vol.color || PALETTE[index % PALETTE.length];
    const r = radius;
    const shape = shapeForKind(vol.kind);
    const coverage = expectedCoverage(vol.cardinality, vol.k);
    const fillOpacity = 0.15 + 0.85 * coverage;
    const filterId = `glow-${vol.key.replace(/\./g, '-')}`;

    const g = createSVGElement('g', {});

    // Filter
    const filter = createSVGElement('filter', { id: filterId, x: '-50%', y: '-50%', width: '200%', height: '200%' });
    const blur = createSVGElement('feGaussianBlur', { 'in': 'SourceGraphic', stdDeviation: '0', result: 'blur' });
    const colorMat = createSVGElement('feColorMatrix', {
      'in': 'blur', type: 'matrix',
      values: '1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 1 0',
      result: 'tinted',
    });
    const merge = createSVGElement('feMerge', {});
    merge.appendChild(createSVGElement('feMergeNode', { 'in': 'tinted' }));
    merge.appendChild(createSVGElement('feMergeNode', { 'in': 'SourceGraphic' }));
    filter.appendChild(blur);
    filter.appendChild(colorMat);
    filter.appendChild(merge);
    g.appendChild(filter);

    // Shape element
    let shapeEl;
    if (shape === 'circle') {
      shapeEl = createSVGElement('circle', {
        cx, cy, r,
        fill: color, 'fill-opacity': fillOpacity,
        stroke: color, 'stroke-width': '1.5', 'stroke-opacity': '0.8',
        filter: `url(#${filterId})`,
      });
    } else {
      const pts = shape === 'hexagon' ? hexagonPoints(cx, cy, r) : squarePoints(cx, cy, r);
      shapeEl = createSVGElement('polygon', {
        points: pts,
        fill: color, 'fill-opacity': fillOpacity,
        stroke: color, 'stroke-width': '1.5', 'stroke-opacity': '0.8',
        filter: `url(#${filterId})`,
      });
    }
    g.appendChild(shapeEl);

    // Pulse dot
    const pulseDot = createSVGElement('circle', {
      cx, cy: cy - r, r: '2',
      fill: '#ffffff', 'fill-opacity': '0.6',
    });
    g.appendChild(pulseDot);

    // Value text above (k/cardinality)
    const cardStr = vol.cardinality > 1000 ? formatNum(vol.cardinality)
                                           : Math.round(vol.cardinality).toString();

    const valueText = createSVGElement('text', {
      x: cx, y: cy - r - 8,
      'text-anchor': 'middle', 'font-size': '10',
      'font-family': "'Cascadia Code', 'Fira Code', monospace",
      fill: color, 'fill-opacity': '0.9',
    });
    valueText.textContent = `${vol.k}/${cardStr}`;
    g.appendChild(valueText);

    // Label text below
    const labelText = createSVGElement('text', {
      x: cx, y: cy + r + 14,
      'text-anchor': 'middle', 'font-size': '9',
      'font-family': 'system-ui, sans-serif',
      fill: '#8888aa',
    });
    labelText.textContent = vol.label;
    g.appendChild(labelText);

    return {
      key: vol.key, g, shapeEl, blurEl: blur, pulseDot,
      exciteTime: 0, phase: 0, color, shape,
      radius: r, cx, cy,
      cardinality: vol.cardinality,
      exploration: vol.exploration,
    };
  }

  // ---------------------------------------------------------------------------
  // Overview strip — thin bar, always full range, with viewport highlight
  // ---------------------------------------------------------------------------

  function buildOverviewStrip(sorted, k, totalLogVol, stripW, parent) {
    const g = createSVGElement('g', {});
    const y = OVERVIEW_Y;
    const h = OVERVIEW_H;

    // Background
    g.appendChild(createSVGElement('rect', {
      x: PAD, y, width: stripW, height: h,
      fill: '#16162a', stroke: '#2a2a4a', 'stroke-width': '0.5', rx: '2',
    }));

    // Clip
    const clipId = 'overview-clip';
    const clipPath = createSVGElement('clipPath', { id: clipId });
    clipPath.appendChild(createSVGElement('rect', { x: PAD, y, width: stripW, height: h, rx: '2' }));
    g.appendChild(clipPath);

    // Factor segments (sorted by log_volume desc — largest leftmost)
    let segX = PAD;
    sorted.forEach((vol, i) => {
      const segW = (vol.log_volume / totalLogVol) * stripW;
      g.appendChild(createSVGElement('rect', {
        x: segX, y, width: Math.max(0.5, segW), height: h,
        fill: vol.color, 'fill-opacity': '0.45',
        'clip-path': `url(#${clipId})`,
      }));
      if (i < sorted.length - 1) {
        g.appendChild(createSVGElement('line', {
          x1: segX + segW, y1: y, x2: segX + segW, y2: y + h,
          stroke: '#2a2a4a', 'stroke-width': '0.5',
        }));
      }
      segX += segW;
    });

    // k marker
    const logK = Math.log2(Math.max(1, k));
    const kFrac = Math.min(1.0, logK / totalLogVol);
    const kX = PAD + kFrac * stripW;
    g.appendChild(createSVGElement('line', {
      x1: kX, y1: y - 1, x2: kX, y2: y + h + 1,
      stroke: '#ffffff', 'stroke-width': '1', 'stroke-opacity': '0.6',
    }));

    // Viewport highlight — brighter region showing what the detail strip shows
    const cropBits = getCropBits(sorted, totalLogVol);
    const vpLeftX = PAD + (cropBits / totalLogVol) * stripW;
    g.appendChild(createSVGElement('rect', {
      x: vpLeftX, y, width: PAD + stripW - vpLeftX, height: h,
      fill: '#ffffff', 'fill-opacity': '0.12',
      'clip-path': `url(#${clipId})`,
    }));

    // Viewport left edge marker (when zoomed)
    if (cropBits > 0.01) {
      g.appendChild(createSVGElement('line', {
        x1: vpLeftX, y1: y - 1, x2: vpLeftX, y2: y + h + 1,
        stroke: '#e879f9', 'stroke-width': '1', 'stroke-opacity': '0.6',
      }));
    }

    parent.appendChild(g);
    return vpLeftX;
  }

  // ---------------------------------------------------------------------------
  // Frustum connector — trapezoid between overview viewport and detail strip
  // ---------------------------------------------------------------------------

  function buildFrustum(vpLeftX, stripW, parent) {
    const g = createSVGElement('g', {});

    const overviewBottom = OVERVIEW_Y + OVERVIEW_H;
    const vpRightX = PAD + stripW;
    const dtLeft = PAD;
    const dtRight = PAD + stripW;

    // Trapezoid fill
    const points = [
      `${vpLeftX},${overviewBottom}`,
      `${vpRightX},${overviewBottom}`,
      `${dtRight},${DETAIL_Y}`,
      `${dtLeft},${DETAIL_Y}`,
    ].join(' ');

    g.appendChild(createSVGElement('polygon', {
      points,
      fill: '#ffffff', 'fill-opacity': '0.02',
    }));

    // Left frustum line (diagonal when zoomed — the visual anchor)
    if (Math.abs(vpLeftX - dtLeft) > 2) {
      g.appendChild(createSVGElement('line', {
        x1: vpLeftX, y1: overviewBottom,
        x2: dtLeft, y2: DETAIL_Y,
        stroke: '#e879f9', 'stroke-width': '0.7', 'stroke-opacity': '0.35',
        'stroke-dasharray': '3,2',
      }));

      // Right frustum line (vertical — anchored at right edge)
      g.appendChild(createSVGElement('line', {
        x1: vpRightX, y1: overviewBottom,
        x2: dtRight, y2: DETAIL_Y,
        stroke: '#4a4a6a', 'stroke-width': '0.5', 'stroke-opacity': '0.25',
      }));
    }

    parent.appendChild(g);
  }

  // ---------------------------------------------------------------------------
  // Detail strip — zoomed view of the viewport region
  // ---------------------------------------------------------------------------

  function buildDetailStrip(sorted, k, totalLogVol, cropBits, viewBits, stripW, parent) {
    const g = createSVGElement('g', {});
    const y = DETAIL_Y;
    const h = DETAIL_H;

    // Background
    g.appendChild(createSVGElement('rect', {
      x: PAD, y, width: stripW, height: h,
      fill: '#16162a', stroke: '#2a2a4a', 'stroke-width': '1', rx: '3',
    }));

    // Clip
    const clipId = 'detail-clip';
    const clipPath = createSVGElement('clipPath', { id: clipId });
    clipPath.appendChild(createSVGElement('rect', { x: PAD, y, width: stripW, height: h, rx: '3' }));
    g.appendChild(clipPath);

    // Factor segments with per-factor marginal coverage fill levels
    let cumBits = 0;
    sorted.forEach((vol, i) => {
      const fStart = cumBits;
      const fEnd = cumBits + vol.log_volume;
      cumBits = fEnd;

      // Intersection of [fStart, fEnd] with [cropBits, totalLogVol]
      const visStart = Math.max(fStart, cropBits);
      const visEnd = fEnd;
      const visBits = visEnd - visStart;

      if (visBits < 0.001) return; // fully cropped

      const segX = PAD + ((visStart - cropBits) / viewBits) * stripW;
      const segW = (visBits / viewBits) * stripW;

      // Base segment (dim)
      g.appendChild(createSVGElement('rect', {
        x: segX, y, width: Math.max(0.5, segW), height: h,
        fill: vol.color, 'fill-opacity': '0.12',
        'clip-path': `url(#${clipId})`,
      }));

      // Per-factor marginal coverage: E[cov] = 1 - ((N-1)/N)^k
      // Rendered as a "water level" fill from the bottom of the segment.
      // High coverage (small enum) → nearly full. Low coverage (PRNG) → sliver.
      const cov = expectedCoverage(vol.cardinality, k);
      const fillH = h * cov;

      if (fillH > 0.3) {
        g.appendChild(createSVGElement('rect', {
          x: segX, y: y + h - fillH, width: Math.max(0.5, segW), height: fillH,
          fill: vol.color, 'fill-opacity': '0.45',
          'clip-path': `url(#${clipId})`,
        }));
      }

      // Separator at right edge (except last)
      if (fEnd < totalLogVol - 0.001) {
        const sepX = PAD + ((fEnd - cropBits) / viewBits) * stripW;
        if (sepX > PAD && sepX < PAD + stripW) {
          g.appendChild(createSVGElement('line', {
            x1: sepX, y1: y, x2: sepX, y2: y + h,
            stroke: '#2a2a4a', 'stroke-width': '1',
          }));
        }
      }

      // Label inside segment (if wide enough)
      if (segW > 30) {
        // Show label + coverage percentage
        const covPct = cov >= 0.995 ? '99+' : cov >= 0.01 ? Math.round(cov * 100) : '<1';
        const lbl = createSVGElement('text', {
          x: segX + segW / 2, y: y + h / 2 + 3,
          'text-anchor': 'middle', 'font-size': '8',
          'font-family': "'Cascadia Code', 'Fira Code', monospace",
          fill: '#ffffff', 'fill-opacity': '0.85',
        });
        lbl.textContent = `${vol.label} ${covPct}%`;
        g.appendChild(lbl);
      } else if (segW > 14) {
        // Just coverage percentage
        const covPct = cov >= 0.995 ? '99+' : cov >= 0.01 ? Math.round(cov * 100) : '<1';
        const lbl = createSVGElement('text', {
          x: segX + segW / 2, y: y + h / 2 + 3,
          'text-anchor': 'middle', 'font-size': '7',
          'font-family': "'Cascadia Code', 'Fira Code', monospace",
          fill: '#ffffff', 'fill-opacity': '0.7',
        });
        lbl.textContent = `${covPct}%`;
        g.appendChild(lbl);
      }
    });

    // Cropped indicator at left edge (small triangle showing "more off-screen")
    if (cropBits > 0.01) {
      const triH = h * 0.5;
      const triW = 5;
      const triCy = y + h / 2;
      g.appendChild(createSVGElement('polygon', {
        points: `${PAD + 1},${triCy} ${PAD + triW + 1},${triCy - triH / 2} ${PAD + triW + 1},${triCy + triH / 2}`,
        fill: '#e879f9', 'fill-opacity': '0.45',
      }));
    }

    parent.appendChild(g);
  }

  // ---------------------------------------------------------------------------
  // Summary text below the detail strip
  // ---------------------------------------------------------------------------

  function buildSummaryText(sorted, k, totalLogVol, cropBits, svgWidth, parent) {
    const g = createSVGElement('g', {});
    const y = SUMMARY_Y;

    // Product expression and coverage
    const totalProduct = sorted.reduce((p, v) => p * v.cardinality, 1);
    const coveragePct = (k / totalProduct) * 100;
    let coverageStr;
    if (coveragePct >= 1) coverageStr = coveragePct.toFixed(1) + '%';
    else if (coveragePct >= 0.01) coverageStr = coveragePct.toFixed(2) + '%';
    else coverageStr = '< 0.01%';

    const productParts = sorted.map(v => formatNum(Math.round(v.cardinality)));
    const productExpr = productParts.join(' \u00d7 ');

    const summary = createSVGElement('text', {
      x: svgWidth / 2, y,
      'text-anchor': 'middle', 'font-size': '9',
      'font-family': "'Cascadia Code', 'Fira Code', monospace",
      fill: '#8888aa',
    });
    summary.textContent = `${productExpr} = ${formatNum(totalProduct)} \u00b7 ${coverageStr} coverage`;
    g.appendChild(summary);

    // Log-coverage (right-aligned)
    const logK = Math.log2(Math.max(1, k));
    const logNote = createSVGElement('text', {
      x: svgWidth - PAD, y,
      'text-anchor': 'end', 'font-size': '8',
      'font-family': "'Cascadia Code', 'Fira Code', monospace",
      fill: '#666688',
    });
    logNote.textContent = `${logK.toFixed(1)}/${totalLogVol.toFixed(1)} bits`;
    g.appendChild(logNote);

    // Viewport annotation when zoomed (left-aligned)
    if (cropBits > 0.01) {
      const viewBits = totalLogVol - cropBits;
      const viewNote = createSVGElement('text', {
        x: PAD, y,
        'text-anchor': 'start', 'font-size': '8',
        'font-family': "'Cascadia Code', 'Fira Code', monospace",
        fill: '#666688',
      });
      viewNote.textContent = `\u25c0 ${viewBits.toFixed(1)}/${totalLogVol.toFixed(1)} bits visible`;
      g.appendChild(viewNote);
    }

    parent.appendChild(g);
  }

  // ---------------------------------------------------------------------------
  // Rebuild the strip group (called on zoom slider changes, no network)
  // ---------------------------------------------------------------------------

  function rebuildStripGroup(svgWidth) {
    if (!stripGroup || cachedSorted.length === 0) return;
    while (stripGroup.firstChild) stripGroup.removeChild(stripGroup.firstChild);

    const sorted = cachedSorted;
    const k = cachedK;
    const totalLogVol = cachedTotalLog;
    if (totalLogVol <= 0) return;

    const stripW = svgWidth - 2 * PAD;
    const cropBits = getCropBits(sorted, totalLogVol);
    const viewBits = totalLogVol - cropBits;

    // 1. Overview strip (thin, full range)
    const vpLeftX = buildOverviewStrip(sorted, k, totalLogVol, stripW, stripGroup);

    // 2. Frustum connector
    buildFrustum(vpLeftX, stripW, stripGroup);

    // 3. Detail strip (zoomed, with per-factor marginal coverage fill levels)
    buildDetailStrip(sorted, k, totalLogVol, cropBits, viewBits, stripW, stripGroup);

    // 4. Summary
    buildSummaryText(sorted, k, totalLogVol, cropBits, svgWidth, stripGroup);
  }

  // ---------------------------------------------------------------------------
  // Update figures from volume data (called when API returns new data)
  // ---------------------------------------------------------------------------

  function updateFigures(volumes) {
    if (!svgEl) return;

    const now = performance.now() / 1000;

    // Sort by log_volume descending, assign stable colors
    const sorted = [...volumes]
      .sort((a, b) => b.log_volume - a.log_volume)
      .map((v, i) => ({ ...v, color: PALETTE[i % PALETTE.length] }));

    // Cache for zoom slider rebuilds
    cachedSorted = sorted;
    cachedK = sorted.length > 0 ? sorted[0].k : 1;
    cachedTotalLog = sorted.reduce((s, v) => s + v.log_volume, 0);
    autoZoomTarget = computeAutoZoom(sorted, cachedTotalLog);

    // Compute radii
    const radii = computeRadii(sorted);

    // Layout
    const figAreaWidth = Math.max(240, sorted.length * FIG_WIDTH);
    const totalHeight = sorted.length > 0 ? TOTAL_H : 0;

    svgEl.setAttribute('viewBox', `0 0 ${figAreaWidth} ${totalHeight}`);
    svgEl.setAttribute('width', figAreaWidth);
    svgEl.setAttribute('height', totalHeight);

    // Diff against previous volumes for change detection
    const prevMap = {};
    for (const pv of prevVolumes) prevMap[pv.key] = pv;

    // Clear SVG
    while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);
    const newFigures = [];

    // Build per-factor figures (sorted order — largest leftmost)
    sorted.forEach((vol, i) => {
      const cx = FIG_WIDTH / 2 + i * FIG_WIDTH;
      const fig = buildFigure(vol, i, cx, FIG_CY, radii[i] || MIN_R);

      // Change detection for excite animation
      const prev = prevMap[vol.key];
      if (prev && (prev.cardinality !== vol.cardinality || prev.exploration !== vol.exploration)) {
        fig.exciteTime = now;
      } else if (!prev) {
        fig.exciteTime = now;
      } else {
        const oldFig = figures.find(f => f.key === vol.key);
        if (oldFig) fig.exciteTime = oldFig.exciteTime;
      }

      svgEl.appendChild(fig.g);
      newFigures.push(fig);
    });

    // Strip group (overview + frustum + detail + summary)
    stripGroup = createSVGElement('g', {});
    svgEl.appendChild(stripGroup);

    if (sorted.length > 0) {
      rebuildStripGroup(figAreaWidth);
    }

    figures = newFigures;
    prevVolumes = sorted.map(v => ({ key: v.key, cardinality: v.cardinality, exploration: v.exploration }));

    // Show/hide container
    container.style.display = sorted.length > 0 ? '' : 'none';

    // Show/hide zoom controls (only useful with ≥2 factors)
    if (zoomRow) {
      zoomRow.style.display = sorted.length >= 2 ? '' : 'none';
    }
  }

  // ---------------------------------------------------------------------------
  // Animation loop (rAF)
  // ---------------------------------------------------------------------------

  function animationFrame(timestamp) {
    const now = timestamp / 1000;

    for (const fig of figures) {
      const elapsed = fig.exciteTime > 0 ? now - fig.exciteTime : TAU + 1;
      const t = Math.max(0, elapsed);
      const intensity = Math.pow(Math.max(0, 1 - t / TAU), P_EXP);

      // Glow blur
      const blurVal = PEAK_BLUR * intensity;
      fig.blurEl.setAttribute('stdDeviation', blurVal.toFixed(2));

      // Pulse dot: orbit speed
      const period = REST_PERIOD + (EXCITE_PERIOD - REST_PERIOD) * intensity;
      fig.phase += (1 / 60) / period * 2 * Math.PI;
      const angle = fig.phase;

      const pt = perimeterPoint(fig.shape, fig.cx, fig.cy, fig.radius, angle);
      fig.pulseDot.setAttribute('cx', pt.x.toFixed(1));
      fig.pulseDot.setAttribute('cy', pt.y.toFixed(1));

      // Pulse dot size + opacity
      const dotR = 1.5 + 2.5 * intensity;
      const dotOpacity = 0.3 + 0.7 * intensity;
      fig.pulseDot.setAttribute('r', dotR.toFixed(1));
      fig.pulseDot.setAttribute('fill-opacity', dotOpacity.toFixed(2));
    }

    rafId = requestAnimationFrame(animationFrame);
  }

  // ---------------------------------------------------------------------------
  // Request update (called from config_flow)
  // ---------------------------------------------------------------------------

  function requestUpdate() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      if (abortCtrl) abortCtrl.abort();
      abortCtrl = new AbortController();

      const config = ConfigFlow.getConfig();
      const k = Math.max(1, Math.min(16, config.k || 1));

      try {
        const resp = await fetch('/api/config/volumes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config, k }),
          signal: abortCtrl.signal,
        });
        if (resp.ok) {
          const data = await resp.json();
          updateFigures(data.volumes);
        }
      } catch (e) {
        if (e.name !== 'AbortError') {
          // Silently fail — geometry is decorative
        }
      }
    }, 200);
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------

  function init() {
    container = document.getElementById('config-geometry-container');
    if (!container) return;

    svgEl = createSVGElement('svg', {
      width: '240',
      height: String(TOTAL_H),
      style: 'display: block; margin: 0 auto;',
    });
    container.appendChild(svgEl);

    // Zoom controls row
    zoomRow = document.createElement('div');
    zoomRow.className = 'geometry-zoom-row';
    zoomRow.style.display = 'none';
    zoomRow.innerHTML = [
      '<span class="geometry-zoom-label">\u25c0\u25b6</span>',
      '<input type="range" class="geometry-zoom-slider" min="0" max="1" step="0.005" value="0">',
      '<button class="geometry-zoom-auto btn btn-small" title="Auto-zoom past dominant factor">Auto</button>',
    ].join('');
    container.appendChild(zoomRow);

    zoomSlider = zoomRow.querySelector('.geometry-zoom-slider');
    zoomSlider.addEventListener('input', () => {
      zoomLevel = parseFloat(zoomSlider.value);
      const svgWidth = parseFloat(svgEl.getAttribute('width'));
      rebuildStripGroup(svgWidth);
    });

    const autoBtn = zoomRow.querySelector('.geometry-zoom-auto');
    autoBtn.addEventListener('click', () => {
      zoomLevel = autoZoomTarget;
      zoomSlider.value = zoomLevel.toFixed(3);
      const svgWidth = parseFloat(svgEl.getAttribute('width'));
      rebuildStripGroup(svgWidth);
    });

    container.style.display = 'none';
    rafId = requestAnimationFrame(animationFrame);

    // Initial fetch
    requestUpdate();
  }

  return { init, requestUpdate };
})();
