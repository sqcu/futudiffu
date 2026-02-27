/* config_flow.js — Diegetic config panel: JSON<->control bidirectional binding.
 *
 * The config JSON editor is always the source of truth. Controls read from it
 * and write to it. Distribution-valued fields get visual annotations.
 *
 * Syntax highlighting: keywords orange, strings green, numbers cyan,
 * distribution markers magenta.
 */

const ConfigFlow = (() => {

  // Current config state
  let config = {};
  let suppressSync = false;

  const editor = document.getElementById('config-editor');

  // ---------------------------------------------------------------------------
  // Distribution detection (mirrors config_distributions.py)
  // ---------------------------------------------------------------------------

  function isDist(v) {
    return v && typeof v === 'object' && !Array.isArray(v) &&
           ('min' in v || 'max' in v || 'values' in v || 'weights' in v);
  }
  function isEnum(v) { return Array.isArray(v); }
  function isDistributional(v) { return isDist(v) || isEnum(v); }

  // ---------------------------------------------------------------------------
  // Syntax highlighting
  // ---------------------------------------------------------------------------

  function highlightJSON(json) {
    // Escape HTML first
    const escaped = json
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    let result = escaped
      // Keys (before colon)
      .replace(/"([^"]+)"(?=\s*:)/g, '<span class="cfg-key">"$1"</span>')
      // Strings (after colon)
      .replace(/:\s*"([^"]*)"/g, ': <span class="cfg-str">"$1"</span>')
      // Numbers
      .replace(/:\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)/gi, ': <span class="cfg-num">$1</span>')
      // Booleans
      .replace(/:\s*(true|false)/g, ': <span class="cfg-bool">$1</span>')
      // Null
      .replace(/:\s*(null)/g, ': <span class="cfg-null">$1</span>')
      // Array contents: numbers get cyan, strings get green
      .replace(/\[([^\]]*)\]/g, (match) => {
        return match
          .replace(/(-?\d+\.?\d*)/g, '<span class="cfg-num">$1</span>')
          .replace(/"([^"]*)"/g, '<span class="cfg-str">"$1"</span>');
      });

    // Highlight distribution dicts: {"min": ..., "max": ...} or {"values": ...}
    // Wrap lines containing distribution keys in magenta
    result = result.replace(
      /("(?:min|max|values|weights|step|distribution)")/g,
      '<span class="cfg-dist">$1</span>'
    );

    return result;
  }

  function renderConfig() {
    suppressSync = true;
    const json = JSON.stringify(config, null, 2);
    editor.innerHTML = highlightJSON(json);
    suppressSync = false;
  }

  // ---------------------------------------------------------------------------
  // Config -> Controls sync
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Unified control metadata — EVERY config field gets affordances
  // ---------------------------------------------------------------------------

  // type: 'slider' | 'number' | 'select'
  // For sliders: min, max, step define the track range and overlays.
  // For selects: options lists the closed set of legal scalar values.
  // For numbers: min, max define the range for distributional conversion.
  // configPath: dotted path into config (e.g. 'resolution.megapixels').
  const FIELD_META = {
    'ctrl-steps':   { type: 'slider', min: 4,    max: 50,   step: 1,    configPath: 'n_steps' },
    'ctrl-cfg':     { type: 'slider', min: 1.0,  max: 15.0, step: 0.5,  configPath: 'cfg' },
    'ctrl-shift':   { type: 'slider', min: 0.1,  max: 8.0,  step: 0.1,  configPath: 'sampling_shift' },
    'ctrl-denoise': { type: 'slider', min: 0.0,  max: 1.0,  step: 0.05, configPath: 'denoise' },
    'ctrl-seed':    { type: 'number', min: 0,    max: 4294967295, step: 1, configPath: 'seed' },
    'ctrl-backend': { type: 'select', options: ['sage', 'sdpa'], configPath: 'attention_backend' },
    'ctrl-anchor':  { type: 'slider', min: 11.09, max: 13.86, step: 0.01, configPath: 'resolution.megapixels',
                      toConfig: (v) => Math.round(Math.exp(v)), fromConfig: (v) => Math.log(v) },
    'ctrl-aspect':  { type: 'slider', min: -0.693, max: 0.693, step: 0.01, configPath: 'resolution.aspect_ratio',
                      toConfig: (v) => Math.exp(v), fromConfig: (v) => Math.log(v) },
  };

  /** Format a large cardinality compactly for badge display. */
  function compactCard(n) {
    if (n > 1e9) return (n / 1e9).toFixed(1) + 'B';
    if (n > 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n > 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toString();
  }

  /** Compute the step to use in config space for a given field.
   * For fields with toConfig transform, derives step from slider positions. */
  function computeConfigStep(meta) {
    if (!meta || !meta.step) return null;
    if (!meta.toConfig) return meta.step;
    // Transformed slider: compute config-space step from slider grid count
    const nPositions = Math.round((meta.max - meta.min) / meta.step);
    if (nPositions <= 0) return meta.step;
    const configMin = meta.toConfig(meta.min);
    const configMax = meta.toConfig(meta.max);
    return parseFloat(((configMax - configMin) / nPositions).toPrecision(3));
  }

  /** Read a dotted path from config (e.g. 'resolution.megapixels'). */
  function getConfigValue(path) {
    const parts = path.split('.');
    let obj = config;
    for (const p of parts) {
      if (obj == null) return undefined;
      obj = obj[p];
    }
    return obj;
  }

  /** Write a dotted path into config. */
  function setConfigValue(path, val) {
    const parts = path.split('.');
    let obj = config;
    for (let i = 0; i < parts.length - 1; i++) {
      if (obj[parts[i]] == null) obj[parts[i]] = {};
      obj = obj[parts[i]];
    }
    obj[parts[parts.length - 1]] = val;
  }

  // ---------------------------------------------------------------------------
  // setDistMode: universal badge + overlays for ANY control
  // ---------------------------------------------------------------------------

  /**
   * Annotate a control with distributional state. Adds clickable badge,
   * slider overlays (range highlights, enum dots), and select overlays.
   * Controls are NEVER disabled — distributional fields stay interactive.
   */
  function setDistMode(controlId, labelSpanId, value) {
    const ctrl = document.getElementById(controlId);
    const label = labelSpanId ? document.getElementById(labelSpanId) : null;
    if (!ctrl) return;

    // Walk up: the control-group div is the scope for badge + overlay cleanup
    const controlGroup = ctrl.closest('.control-group') || ctrl.parentElement;

    // Remove any existing dist badge and overlays (search the whole control group)
    controlGroup.querySelectorAll('.dist-badge').forEach(el => el.remove());
    controlGroup.querySelectorAll('.slider-range-highlight, .slider-enum-dot, .select-dist-overlay, .range-handle, .range-step-row, .range-value-label, .enum-value-label').forEach(el => el.remove());

    // Never disable — remove old class if present
    ctrl.classList.remove('ctrl-dist-mode');

    const meta = FIELD_META[controlId];

    // ---- Badge (ALWAYS visible — scalar=dimmed, distributional=colored) ----
    const badgeTarget = label ? label.parentElement : controlGroup.querySelector('label') || controlGroup;
    const cfgPath = meta?.configPath || controlId.replace('ctrl-', '');
    const badge = document.createElement('span');
    badge.className = 'dist-badge';
    badge.addEventListener('click', (e) => {
      e.stopPropagation();
      cycleDistributionType(cfgPath, controlId);
    });

    if (!isDistributional(value)) {
      badge.textContent = 'FIXED';
      badge.classList.add('dist-badge-scalar');
      badge.title = 'Click to make distributional';
    } else if (isEnum(value)) {
      badge.textContent = `ENUM(${value.length})`;
      badge.title = 'Click to cycle distribution type';
    } else if (isDist(value) && 'values' in value && 'weights' in value) {
      badge.textContent = `WCAT(${value.values.length})`;
      badge.title = 'Click to cycle distribution type';
    } else if (isDist(value) && 'min' in value && 'max' in value) {
      const distLabel = value.distribution === 'log_uniform' ? 'LOG' : 'RANGE';
      const step = value.step;
      if (step && step > 0) {
        badge.textContent = `${distLabel}(${compactCard(Math.floor((value.max - value.min) / step) + 1)})`;
      } else if (Number.isInteger(value.min) && Number.isInteger(value.max)) {
        badge.textContent = `${distLabel}(${compactCard(value.max - value.min + 1)})`;
      } else {
        badge.textContent = distLabel;
      }
      badge.title = 'Click to cycle distribution type';
    } else {
      badge.textContent = 'DIST';
      badge.title = 'Click to cycle distribution type';
    }
    badgeTarget.appendChild(badge);

    if (isDistributional(value)) {
      // ---- Slider-specific overlays ----
      if (meta && (meta.type === 'slider')) {
        const sliderMin = meta.min, sliderMax = meta.max;
        const toTrack = meta.fromConfig
          ? (v) => ((meta.fromConfig(v) - sliderMin) / (sliderMax - sliderMin)) * 100
          : (v) => ((v - sliderMin) / (sliderMax - sliderMin)) * 100;

        if (isDist(value) && 'min' in value && 'max' in value) {
          ensureSliderWrapper(ctrl);
          const wrapper = ctrl.parentElement;
          const pctLo = toTrack(value.min);
          const pctHi = toTrack(value.max);

          // Range highlight bar
          const highlight = document.createElement('div');
          highlight.className = 'slider-range-highlight';
          highlight.style.left = Math.max(0, pctLo) + '%';
          highlight.style.width = Math.min(100, pctHi) - Math.max(0, pctLo) + '%';
          wrapper.appendChild(highlight);

          // Draggable handles at min and max
          const makeHandle = (pct, end) => {
            const h = document.createElement('div');
            h.className = 'range-handle';
            h.dataset.end = end;
            h.style.left = Math.max(0, Math.min(100, pct)) + '%';
            h.addEventListener('mousedown', (e) => {
              e.preventDefault();
              e.stopPropagation();
              startRangeHandleDrag(controlId, end);
            });
            wrapper.appendChild(h);
          };
          makeHandle(pctLo, 'lo');
          makeHandle(pctHi, 'hi');

          // Value labels at handle positions
          const fmtVal = (v) => {
            if (Number.isInteger(v)) return String(v);
            if (Math.abs(v) >= 100) return v.toFixed(0);
            if (Math.abs(v) >= 10) return v.toFixed(1);
            return v.toPrecision(3);
          };
          const makeValLabel = (pct, text) => {
            const lbl = document.createElement('span');
            lbl.className = 'range-value-label';
            lbl.style.left = Math.max(0, Math.min(100, pct)) + '%';
            lbl.textContent = text;
            wrapper.appendChild(lbl);
          };
          makeValLabel(pctLo, fmtVal(value.min));
          makeValLabel(pctHi, fmtVal(value.max));

          // Step input row (granularity control)
          const stepRow = document.createElement('div');
          stepRow.className = 'range-step-row';
          stepRow.innerHTML = '<span>step:</span>';
          const stepInput = document.createElement('input');
          stepInput.type = 'number';
          stepInput.className = 'range-step-input';
          stepInput.step = 'any';
          stepInput.value = value.step != null ? value.step : '';
          stepInput.placeholder = 'auto';
          stepInput.addEventListener('change', () => {
            const sv = parseFloat(stepInput.value);
            const curVal = getConfigValue(cfgPath);
            if (isDist(curVal) && 'min' in curVal && 'max' in curVal) {
              if (!isNaN(sv) && sv > 0) {
                curVal.step = sv;
              } else {
                delete curVal.step;
              }
              setConfigValue(cfgPath, curVal);
              renderConfig();
              syncConfigToControls();
              if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
            }
          });
          stepRow.appendChild(stepInput);

          // Cardinality readout
          const cardSpan = document.createElement('span');
          cardSpan.className = 'range-step-card';
          if (value.step && value.step > 0) {
            const n = Math.floor((value.max - value.min) / value.step) + 1;
            cardSpan.textContent = `= ${n} values`;
          }
          stepRow.appendChild(cardSpan);

          // Insert step row after the wrapper within the control group
          wrapper.parentElement.insertBefore(stepRow, wrapper.nextSibling);
        }

        if (isEnum(value)) {
          ensureSliderWrapper(ctrl);
          const wrapper = ctrl.parentElement;
          const fmtEnumVal = (v) => {
            if (typeof v === 'string') return v;
            if (Number.isInteger(v)) return String(v);
            if (Math.abs(v) >= 10) return v.toFixed(1);
            return v.toPrecision(3);
          };
          for (let ei = 0; ei < value.length; ei++) {
            const v = value[ei];
            const pct = toTrack(v);
            const dot = document.createElement('div');
            dot.className = 'slider-enum-dot slider-enum-dot-draggable';
            dot.style.left = pct + '%';
            dot.dataset.enumIndex = ei;
            dot.addEventListener('mousedown', (e) => {
              e.preventDefault();
              e.stopPropagation();
              startEnumDotDrag(controlId, parseInt(dot.dataset.enumIndex, 10));
            });
            wrapper.appendChild(dot);

            // Value labels: show all if ≤6 values, else first and last
            if (value.length <= 6 || ei === 0 || ei === value.length - 1) {
              const lbl = document.createElement('span');
              lbl.className = 'enum-value-label';
              lbl.style.left = pct + '%';
              lbl.dataset.enumIndex = ei;
              lbl.textContent = fmtEnumVal(v);
              wrapper.appendChild(lbl);
            }
          }
        }

        if (isDist(value) && 'values' in value && 'weights' in value) {
          ensureSliderWrapper(ctrl);
          const wrapper = ctrl.parentElement;
          const maxW = Math.max(...value.weights);
          for (let i = 0; i < value.values.length; i++) {
            const pct = toTrack(value.values[i]);
            const dot = document.createElement('div');
            dot.className = 'slider-enum-dot';
            dot.style.left = pct + '%';
            const size = 4 + 6 * Math.sqrt(value.weights[i] / maxW);
            dot.style.width = size + 'px';
            dot.style.height = size + 'px';
            wrapper.appendChild(dot);
          }
        }
      }

      // ---- Select-specific overlay: show which options are in the distribution ----
      if (meta && (meta.type === 'select')) {
        const activeSet = new Set();
        if (isEnum(value)) value.forEach(v => activeSet.add(String(v)));
        else if (isDist(value) && 'values' in value) value.values.forEach(v => activeSet.add(String(v)));

        if (activeSet.size > 0) {
          const overlay = document.createElement('div');
          overlay.className = 'select-dist-overlay';
          const items = [...activeSet].map(v => `<span class="select-dist-chip">${v}</span>`);
          overlay.innerHTML = items.join(' ');
          // Insert after the select
          ctrl.parentElement.insertBefore(overlay, ctrl.nextSibling);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Prompt distributional badge — textareas get FIXED / ENUM(N) badges
  // ---------------------------------------------------------------------------

  /**
   * Annotate a prompt textarea with distributional state badge.
   * Click badge to cycle: scalar <-> enum (split/join on \n---\n delimiter).
   */
  function setPromptDistMode(controlId, value) {
    const ctrl = document.getElementById(controlId);
    if (!ctrl) return;
    const controlGroup = ctrl.closest('.control-group') || ctrl.parentElement;

    // Cleanup
    controlGroup.querySelectorAll('.dist-badge').forEach(el => el.remove());
    ctrl.classList.remove('ctrl-dist-textarea');

    const configKey = controlId === 'ctrl-prompt' ? 'prompt' : 'negative_prompt';
    const badgeTarget = controlGroup.querySelector('label') || controlGroup;

    const badge = document.createElement('span');
    badge.className = 'dist-badge';
    badge.addEventListener('click', (e) => {
      e.stopPropagation();
      cyclePromptDistribution(configKey, controlId);
    });

    if (!isDistributional(value)) {
      badge.textContent = 'FIXED';
      badge.classList.add('dist-badge-scalar');
      badge.title = 'Click to split into distributional prompt set';
    } else if (isEnum(value)) {
      badge.textContent = `ENUM(${value.length})`;
      badge.title = 'Click to collapse to single prompt';
    } else {
      badge.textContent = 'DIST';
    }
    badgeTarget.appendChild(badge);
  }

  /**
   * Cycle prompt field between scalar and enum.
   * scalar -> enum: split textarea on \n---\n delimiter, or wrap in array
   * enum -> scalar: take first element
   */
  function cyclePromptDistribution(configKey, controlId) {
    const value = config[configKey];
    const ctrl = document.getElementById(controlId);

    if (!isDistributional(value)) {
      // Scalar -> enum: split on delimiter or wrap as single-element array
      const text = ctrl ? ctrl.value : (typeof value === 'string' ? value : '');
      const parts = text.split('\n---\n').map(s => s.trim()).filter(s => s.length > 0);
      config[configKey] = parts.length > 1 ? parts : [text];
    } else if (isEnum(value)) {
      // Enum -> scalar: take first element
      config[configKey] = value[0] || '';
    }

    renderConfig();
    syncConfigToControls();
    if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
  }

  /**
   * Ensure a slider is wrapped in a position:relative div for overlays.
   */
  function ensureSliderWrapper(ctrl) {
    if (ctrl.parentElement.classList.contains('slider-wrapper')) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'slider-wrapper';
    ctrl.parentElement.insertBefore(wrapper, ctrl);
    wrapper.appendChild(ctrl);
  }

  // ---------------------------------------------------------------------------
  // Range handle dragging — diegetic min/max adjustment on slider track
  // ---------------------------------------------------------------------------

  function startRangeHandleDrag(controlId, end) {
    const ctrl = document.getElementById(controlId);
    const meta = FIELD_META[controlId];
    if (!ctrl || !meta) return;

    const wrapper = ctrl.parentElement;
    if (!wrapper) return;

    const cfgPath = meta.configPath;
    const sliderMin = meta.min, sliderMax = meta.max;
    const toConfigVal = meta.toConfig || (v => v);
    const fromConfigVal = meta.fromConfig || (v => v);

    const curVal = getConfigValue(cfgPath);
    if (!curVal || !('min' in curVal) || !('max' in curVal)) return;

    // Current positions in slider space
    let loSlider = fromConfigVal(curVal.min);
    let hiSlider = fromConfigVal(curVal.max);

    const handle = wrapper.querySelector(`.range-handle[data-end="${end}"]`);
    const highlight = wrapper.querySelector('.slider-range-highlight');

    const onMove = (e) => {
      const rect = wrapper.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      let sliderVal = sliderMin + pct * (sliderMax - sliderMin);

      // Snap to slider step grid
      sliderVal = Math.round(sliderVal / meta.step) * meta.step;
      sliderVal = Math.max(sliderMin, Math.min(sliderMax, sliderVal));

      if (end === 'lo') {
        if (sliderVal < hiSlider) loSlider = sliderVal;
      } else {
        if (sliderVal > loSlider) hiSlider = sliderVal;
      }

      // Update visuals
      const loPct = ((loSlider - sliderMin) / (sliderMax - sliderMin)) * 100;
      const hiPct = ((hiSlider - sliderMin) / (sliderMax - sliderMin)) * 100;
      if (handle) handle.style.left = (end === 'lo' ? loPct : hiPct) + '%';
      if (highlight) {
        highlight.style.left = loPct + '%';
        highlight.style.width = (hiPct - loPct) + '%';
      }
    };

    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);

      // Commit to config
      const newMin = toConfigVal(loSlider);
      const newMax = toConfigVal(hiSlider);
      const round = (v) => Number.isInteger(v) ? v : parseFloat(v.toPrecision(6));
      const newVal = { ...curVal, min: round(newMin), max: round(newMax) };
      setConfigValue(cfgPath, newVal);
      renderConfig();
      syncConfigToControls();
      if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }

  // ---------------------------------------------------------------------------
  // Enum dot dragging — reposition individual enum values on slider track
  // ---------------------------------------------------------------------------

  function startEnumDotDrag(controlId, enumIndex) {
    const ctrl = document.getElementById(controlId);
    const meta = FIELD_META[controlId];
    if (!ctrl || !meta) return;

    const wrapper = ctrl.parentElement;
    if (!wrapper) return;

    const cfgPath = meta.configPath;
    const sliderMin = meta.min, sliderMax = meta.max;
    const toConfigVal = meta.toConfig || (v => v);

    const curVal = getConfigValue(cfgPath);
    if (!isEnum(curVal) || enumIndex < 0 || enumIndex >= curVal.length) return;

    const dot = wrapper.querySelector(`.slider-enum-dot[data-enum-index="${enumIndex}"]`);
    const label = wrapper.querySelector(`.enum-value-label[data-enum-index="${enumIndex}"]`);

    const fmtVal = (v) => {
      if (Number.isInteger(v)) return String(v);
      if (Math.abs(v) >= 10) return v.toFixed(1);
      return v.toPrecision(3);
    };

    const onMove = (e) => {
      const rect = wrapper.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      let sliderVal = sliderMin + pct * (sliderMax - sliderMin);

      // Snap to slider step grid
      sliderVal = Math.round(sliderVal / meta.step) * meta.step;
      sliderVal = Math.max(sliderMin, Math.min(sliderMax, sliderVal));

      const configVal = toConfigVal(sliderVal);

      // Update visual position
      const displayPct = ((sliderVal - sliderMin) / (sliderMax - sliderMin)) * 100;
      if (dot) dot.style.left = displayPct + '%';
      if (label) {
        label.style.left = displayPct + '%';
        label.textContent = fmtVal(configVal);
      }
    };

    const onUp = (e) => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);

      // Compute final value
      const rect = wrapper.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      let sliderVal = sliderMin + pct * (sliderMax - sliderMin);
      sliderVal = Math.round(sliderVal / meta.step) * meta.step;
      sliderVal = Math.max(sliderMin, Math.min(sliderMax, sliderVal));
      const configVal = toConfigVal(sliderVal);

      // Round for clean JSON
      const round = (v) => Number.isInteger(v) ? v : parseFloat(v.toPrecision(6));

      // Update the enum array
      const newArr = [...curVal];
      newArr[enumIndex] = round(configVal);
      // Sort and deduplicate
      const unique = [...new Set(newArr)];
      if (typeof unique[0] === 'number') unique.sort((a, b) => a - b);

      setConfigValue(cfgPath, unique.length > 1 ? unique : unique[0]);
      renderConfig();
      syncConfigToControls();
      if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }

  // ---------------------------------------------------------------------------
  // Distribution type cycling: works on ANY field via dotted config path
  // ---------------------------------------------------------------------------

  /**
   * Cycle a config field through distribution types.
   * The cycle depends on the field's control type:
   *   slider/number: scalar -> enum -> range -> scalar
   *   select:        scalar -> enum (all options) -> scalar
   */
  function cycleDistributionType(configPath, controlId) {
    const value = getConfigValue(configPath);
    const meta = controlId ? FIELD_META[controlId] : null;

    if (!isDistributional(value)) {
      // ---- scalar -> enum ----
      if (meta && meta.type === 'slider') {
        const step = meta.step;
        const cur = meta.fromConfig ? meta.fromConfig(value) : value;
        const lo = Math.max(meta.min, cur - step * 2);
        const hi = Math.min(meta.max, cur + step * 2);
        const toVal = meta.toConfig || (v => v);
        const vals = [...new Set([toVal(lo), value, toVal(hi)])];
        if (typeof value === 'number') vals.sort((a, b) => a - b);
        setConfigValue(configPath, vals);
      } else if (meta && (meta.type === 'select')) {
        // Enum over all legal options
        setConfigValue(configPath, [...meta.options]);
      } else if (meta && meta.type === 'number') {
        // Number: create a range around current value (±10% of range)
        const span = (meta.max - meta.min) * 0.1;
        const cur = typeof value === 'number' ? value : meta.min;
        const rangeSpec = {
          min: Math.max(meta.min, Math.round(cur - span)),
          max: Math.min(meta.max, Math.round(cur + span)),
        };
        if (meta.step) rangeSpec.step = meta.step;
        setConfigValue(configPath, rangeSpec);
      } else {
        // Fallback: single-element enum
        setConfigValue(configPath, [value]);
      }
    } else if (isEnum(value)) {
      // ---- enum -> range (for numeric) or -> scalar (for string/select) ----
      const nums = value.filter(v => typeof v === 'number');
      if (nums.length >= 2 && meta && (meta.type === 'slider' || meta.type === 'number')) {
        const rangeSpec = { min: Math.min(...nums), max: Math.max(...nums) };
        const cfgStep = computeConfigStep(meta);
        if (cfgStep) rangeSpec.step = parseFloat(Number(cfgStep).toPrecision(4));
        setConfigValue(configPath, rangeSpec);
      } else {
        // String enums or single-value enums -> collapse to first value
        setConfigValue(configPath, value[0] !== undefined ? value[0] : '');
      }
    } else if (isDist(value)) {
      // ---- range/dist -> scalar ----
      if ('min' in value && 'max' in value) {
        const mid = (value.min + value.max) / 2;
        if (meta && meta.type === 'slider' && meta.step) {
          // For log-scale sliders, compute midpoint in config space
          const configMid = meta.toConfig ? meta.toConfig((meta.fromConfig(value.min) + meta.fromConfig(value.max)) / 2) : mid;
          setConfigValue(configPath, typeof configMid === 'number' ? configMid : Math.round(mid));
        } else {
          setConfigValue(configPath, typeof value.min === 'number' && typeof value.max === 'number' ? Math.round(mid) : mid);
        }
      } else if ('values' in value) {
        setConfigValue(configPath, value.values[0]);
      } else {
        setConfigValue(configPath, 0);
      }
    }

    renderConfig();
    syncConfigToControls();
    if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
  }

  // ---------------------------------------------------------------------------
  // Alt+Drag (or Shift+Drag) to create a range from a scalar slider
  // ---------------------------------------------------------------------------

  let altDragState = null;

  function setupAltDrag(controlId) {
    const ctrl = document.getElementById(controlId);
    const meta = FIELD_META[controlId];
    if (!ctrl || !meta || meta.type !== 'slider') return;

    ctrl.addEventListener('mousedown', (e) => {
      if (!(e.altKey || e.shiftKey)) return;
      const curVal = getConfigValue(meta.configPath);
      if (isDistributional(curVal)) return; // already distributional

      e.preventDefault();
      const startRaw = parseFloat(ctrl.value);
      altDragState = { controlId, meta, startRaw };

      const onMove = (me) => {
        if (!altDragState) return;
        const rect = ctrl.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (me.clientX - rect.left) / rect.width));
        const rawVal = meta.min + pct * (meta.max - meta.min);
        const snapped = Math.round(rawVal / meta.step) * meta.step;

        // Show temporary range highlight
        ensureSliderWrapper(ctrl);
        const wrapper = ctrl.parentElement;
        wrapper.querySelectorAll('.slider-range-highlight').forEach(el => el.remove());
        const lo = Math.min(altDragState.startRaw, snapped);
        const hi = Math.max(altDragState.startRaw, snapped);
        const pctLo = ((lo - meta.min) / (meta.max - meta.min)) * 100;
        const pctHi = ((hi - meta.min) / (meta.max - meta.min)) * 100;
        const highlight = document.createElement('div');
        highlight.className = 'slider-range-highlight';
        highlight.style.left = pctLo + '%';
        highlight.style.width = (pctHi - pctLo) + '%';
        wrapper.appendChild(highlight);
      };

      const onUp = (me) => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        if (!altDragState) return;

        const rect = ctrl.getBoundingClientRect();
        const pct = Math.max(0, Math.min(1, (me.clientX - rect.left) / rect.width));
        const endRaw = Math.round((meta.min + pct * (meta.max - meta.min)) / meta.step) * meta.step;
        const loRaw = Math.min(altDragState.startRaw, endRaw);
        const hiRaw = Math.max(altDragState.startRaw, endRaw);

        // Only convert if drag distance is sufficient (> 1 step)
        if (Math.abs(hiRaw - loRaw) > meta.step * 0.5) {
          const toVal = meta.toConfig || (v => v);
          const loVal = toVal(parseFloat(loRaw.toFixed(6)));
          const hiVal = toVal(parseFloat(hiRaw.toFixed(6)));
          // Ensure min < max even after toConfig transform
          const minVal = Math.min(loVal, hiVal);
          const maxVal = Math.max(loVal, hiVal);
          const rangeSpec = {
            min: parseFloat(minVal.toFixed(4)),
            max: parseFloat(maxVal.toFixed(4)),
          };
          const cfgStep = computeConfigStep(meta);
          if (cfgStep) rangeSpec.step = parseFloat(Number(cfgStep).toPrecision(4));
          setConfigValue(meta.configPath, rangeSpec);
          renderConfig();
          syncConfigToControls();
          if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
        } else {
          // Remove temp highlight
          const wrapper = ctrl.parentElement;
          if (wrapper) wrapper.querySelectorAll('.slider-range-highlight').forEach(el => el.remove());
        }

        altDragState = null;
      };

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });
  }

  function syncConfigToControls() {
    suppressSync = true;

    const $ = (id) => document.getElementById(id);

    // --- Prompt (may be scalar string or enum array) ---
    if (config.prompt !== undefined) {
      setPromptDistMode('ctrl-prompt', config.prompt);
      if (!isDistributional(config.prompt)) {
        $('ctrl-prompt').value = config.prompt;
      } else if (isEnum(config.prompt)) {
        $('ctrl-prompt').value = config.prompt.join('\n---\n');
        $('ctrl-prompt').classList.add('ctrl-dist-textarea');
      }
    }
    if (config.negative_prompt !== undefined) {
      setPromptDistMode('ctrl-neg-prompt', config.negative_prompt);
      if (!isDistributional(config.negative_prompt)) {
        $('ctrl-neg-prompt').value = config.negative_prompt;
      } else if (isEnum(config.negative_prompt)) {
        $('ctrl-neg-prompt').value = config.negative_prompt.join('\n---\n');
        $('ctrl-neg-prompt').classList.add('ctrl-dist-textarea');
      }
    }

    // --- Seed ---
    if (config.seed !== undefined) {
      setDistMode('ctrl-seed', null, config.seed);
      if (!isDistributional(config.seed)) {
        $('ctrl-seed').value = config.seed;
        $('ctrl-seed').placeholder = '-1 = random';
      } else {
        $('ctrl-seed').value = '';
        $('ctrl-seed').placeholder = 'distributional';
      }
    }

    // --- Steps ---
    if (config.n_steps !== undefined) {
      setDistMode('ctrl-steps', 'steps-value', config.n_steps);
      if (!isDistributional(config.n_steps)) {
        $('ctrl-steps').value = config.n_steps;
        $('steps-value').textContent = config.n_steps;
      } else {
        $('steps-value').textContent = '~';
      }
    }

    // --- CFG ---
    if (config.cfg !== undefined) {
      setDistMode('ctrl-cfg', 'cfg-value', config.cfg);
      if (!isDistributional(config.cfg)) {
        $('ctrl-cfg').value = config.cfg;
        $('cfg-value').textContent = config.cfg;
      } else {
        $('cfg-value').textContent = '~';
      }
    }

    // --- Attention backend ---
    if (config.attention_backend !== undefined) {
      setDistMode('ctrl-backend', null, config.attention_backend);
      if (!isDistributional(config.attention_backend)) {
        $('ctrl-backend').value = config.attention_backend;
      }
    }

    // --- Shift ---
    if (config.sampling_shift !== undefined) {
      setDistMode('ctrl-shift', 'shift-value', config.sampling_shift);
      if (!isDistributional(config.sampling_shift)) {
        $('ctrl-shift').value = config.sampling_shift;
        $('shift-value').textContent = config.sampling_shift;
      } else {
        $('shift-value').textContent = '~';
      }
    }

    // --- Denoise ---
    if (config.denoise !== undefined) {
      setDistMode('ctrl-denoise', 'denoise-value', config.denoise);
      if (!isDistributional(config.denoise)) {
        $('ctrl-denoise').value = config.denoise;
        $('denoise-value').textContent = config.denoise;
      } else {
        $('denoise-value').textContent = '~';
      }
    }

    // --- k slider (never distributional) ---
    if (config.k !== undefined) {
      const kVal = Math.max(1, Math.min(16, config.k));
      $('ctrl-k').value = kVal;
      $('k-value').textContent = kVal;
    }

    // --- Resolution ---
    if (config.resolution) {
      const res = config.resolution;

      // Megapixel (px²) slider
      if (res.megapixels !== undefined) {
        setDistMode('ctrl-anchor', 'anchor-value', res.megapixels);
        if (!isDistributional(res.megapixels)) {
          $('ctrl-anchor').value = Math.log(res.megapixels);
          updateAnchorLabel(res.megapixels);
        } else {
          $('anchor-value').textContent = '~';
        }
      }

      // Aspect ratio
      if (res.aspect_ratio !== undefined) {
        setDistMode('ctrl-aspect', 'aspect-label', res.aspect_ratio);
        if (!isDistributional(res.aspect_ratio)) {
          const logVal = Math.log(res.aspect_ratio);
          $('ctrl-aspect').value = logVal;
          updateAspectLabel(res.aspect_ratio);
        } else {
          $('aspect-label').textContent = '~';
        }
      }
    }

    // Show i2i group if denoise < 1
    const i2iGroup = $('i2i-group');
    if (config.denoise !== undefined && typeof config.denoise === 'number' && config.denoise < 1.0) {
      i2iGroup.style.display = '';
    }

    suppressSync = false;
    updateResolutionPreview();
  }

  // ---------------------------------------------------------------------------
  // Controls -> Config sync
  // ---------------------------------------------------------------------------

  function syncControlsToConfig() {
    if (suppressSync) return;

    const $ = (id) => document.getElementById(id);

    // Prompt: only overwrite if scalar. Distributional arrays are edited via
    // the JSON editor or badge cycling, not the textarea.
    if (!isDistributional(config.prompt))
      config.prompt = $('ctrl-prompt').value;
    if (!isDistributional(config.negative_prompt))
      config.negative_prompt = $('ctrl-neg-prompt').value;

    // Only overwrite scalar fields from controls; distributional fields are
    // preserved — the control shows an annotation, not the authoritative value.
    if (!isDistributional(config.seed)) {
      const sv = parseInt($('ctrl-seed').value, 10);
      if (!isNaN(sv)) config.seed = sv;
    }
    if (!isDistributional(config.n_steps))
      config.n_steps = parseInt($('ctrl-steps').value, 10);
    if (!isDistributional(config.cfg))
      config.cfg = parseFloat($('ctrl-cfg').value);
    if (!isDistributional(config.attention_backend))
      config.attention_backend = $('ctrl-backend').value;
    if (!isDistributional(config.sampling_shift))
      config.sampling_shift = parseFloat($('ctrl-shift').value);
    if (!isDistributional(config.denoise))
      config.denoise = parseFloat($('ctrl-denoise').value);

    // k slider (never distributional)
    config.k = parseInt($('ctrl-k').value, 10);

    // Resolution
    if (!config.resolution) config.resolution = {};
    if (!isDistributional(config.resolution.megapixels))
      config.resolution.megapixels = Math.round(Math.exp(parseFloat($('ctrl-anchor').value)));
    if (!isDistributional(config.resolution.aspect_ratio)) {
      config.resolution.aspect_ratio = Math.exp(parseFloat($('ctrl-aspect').value));
    }
    config.resolution.quantize = 32;

    renderConfig();
    if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
  }

  // ---------------------------------------------------------------------------
  // Editor -> Config parse
  // ---------------------------------------------------------------------------

  function parseEditorToConfig() {
    if (suppressSync) return;
    try {
      const text = editor.textContent || editor.innerText;
      const parsed = JSON.parse(text);
      config = parsed;
      syncConfigToControls();
      if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
    } catch (e) {
      // Invalid JSON — don't update controls, user is still typing
    }
  }

  // ---------------------------------------------------------------------------
  // Aspect ratio label
  // ---------------------------------------------------------------------------

  const SNAP_LABELS = [
    [0.5, '1:2'], [0.667, '2:3'], [0.75, '3:4'],
    [1.0, '1:1'],
    [1.333, '4:3'], [1.5, '3:2'], [2.0, '2:1'],
  ];

  function updateAspectLabel(ratio) {
    let best = '?';
    let bestDist = Infinity;
    for (const [val, label] of SNAP_LABELS) {
      const d = Math.abs(ratio - val);
      if (d < bestDist) {
        bestDist = d;
        best = label;
      }
    }
    // If close enough to a snap point, show the label; otherwise show the ratio
    if (bestDist < 0.05) {
      document.getElementById('aspect-label').textContent = best;
    } else {
      document.getElementById('aspect-label').textContent = ratio.toFixed(2);
    }
  }

  function updateAnchorLabel(px2) {
    const el = document.getElementById('anchor-value');
    if (!el) return;
    if (px2 >= 1e6) el.textContent = (px2 / 1e6).toFixed(2) + 'M';
    else if (px2 >= 1e3) el.textContent = Math.round(px2 / 1e3) + 'K';
    else el.textContent = String(px2);
  }

  // ---------------------------------------------------------------------------
  // Resolution preview (calls BFF endpoint)
  // ---------------------------------------------------------------------------

  let resDebounce = null;

  function updateResolutionPreview() {
    clearTimeout(resDebounce);
    resDebounce = setTimeout(async () => {
      const rawAnchor = config.resolution?.megapixels;
      const rawAspect = config.resolution?.aspect_ratio;
      // Skip resolution preview if distributional
      const anchor = (typeof rawAnchor === 'number') ? rawAnchor : 1048576;
      const aspect = (typeof rawAspect === 'number') ? rawAspect : 1.0;

      try {
        const resp = await fetch('/api/resolution', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ anchor_pixels: anchor, aspect_ratio: aspect }),
        });
        if (resp.ok) {
          const data = await resp.json();
          document.getElementById('res-preview').textContent =
            `${data.width} \u00d7 ${data.height}`;
          document.getElementById('res-pixels').textContent =
            `${data.actual_pixels.toLocaleString()} px`;
        }
      } catch (e) {
        // Fallback: simple computation
        const w = Math.round(Math.sqrt(anchor * aspect) / 32) * 32;
        const h = Math.round(Math.sqrt(anchor / aspect) / 32) * 32;
        document.getElementById('res-preview').textContent = `${w} \u00d7 ${h}`;
        document.getElementById('res-pixels').textContent =
          `${(w * h).toLocaleString()} px`;
      }
    }, 100);
  }

  // ---------------------------------------------------------------------------
  // Initialize
  // ---------------------------------------------------------------------------

  function init(defaultConfig) {
    config = JSON.parse(JSON.stringify(defaultConfig));
    renderConfig();
    syncConfigToControls();
    bindControls();
  }

  function bindControls() {
    const $ = (id) => document.getElementById(id);

    // Text inputs — prompt textareas update arrays when distributional
    $('ctrl-prompt').addEventListener('input', () => {
      if (isDistributional(config.prompt)) {
        const parts = $('ctrl-prompt').value.split('\n---\n').map(s => s.trim()).filter(s => s.length > 0);
        if (parts.length > 0) {
          config.prompt = parts.length > 1 ? parts : parts;
          renderConfig();
          if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
        }
      } else {
        syncControlsToConfig();
      }
    });
    $('ctrl-neg-prompt').addEventListener('input', () => {
      if (isDistributional(config.negative_prompt)) {
        const parts = $('ctrl-neg-prompt').value.split('\n---\n').map(s => s.trim()).filter(s => s.length > 0);
        if (parts.length > 0) {
          config.negative_prompt = parts.length > 1 ? parts : parts;
          renderConfig();
          if (typeof ConfigGeometry !== 'undefined') ConfigGeometry.requestUpdate();
        }
      } else {
        syncControlsToConfig();
      }
    });
    $('ctrl-seed').addEventListener('change', syncControlsToConfig);

    // Sliders
    $('ctrl-steps').addEventListener('input', () => {
      $('steps-value').textContent = $('ctrl-steps').value;
      syncControlsToConfig();
    });
    $('ctrl-cfg').addEventListener('input', () => {
      $('cfg-value').textContent = parseFloat($('ctrl-cfg').value).toFixed(1);
      syncControlsToConfig();
    });
    $('ctrl-shift').addEventListener('input', () => {
      $('shift-value').textContent = parseFloat($('ctrl-shift').value).toFixed(1);
      syncControlsToConfig();
    });
    $('ctrl-denoise').addEventListener('input', () => {
      const val = parseFloat($('ctrl-denoise').value).toFixed(2);
      $('denoise-value').textContent = val;
      // Show/hide i2i group
      $('i2i-group').style.display = val < 1.0 ? '' : 'none';
      syncControlsToConfig();
    });

    // k slider
    $('ctrl-k').addEventListener('input', () => {
      $('k-value').textContent = $('ctrl-k').value;
      syncControlsToConfig();
    });

    // Selects
    $('ctrl-backend').addEventListener('change', syncControlsToConfig);

    // px² slider (log scale)
    $('ctrl-anchor').addEventListener('input', () => {
      const px2 = Math.round(Math.exp(parseFloat($('ctrl-anchor').value)));
      updateAnchorLabel(px2);
      syncControlsToConfig();
    });

    // Aspect slider
    $('ctrl-aspect').addEventListener('input', () => {
      const ratio = Math.exp(parseFloat($('ctrl-aspect').value));
      updateAspectLabel(ratio);
      syncControlsToConfig();
    });

    // Alt+Drag (or Shift+Drag) to create ranges on any slider
    for (const [ctrlId, meta] of Object.entries(FIELD_META)) {
      if (meta.type === 'slider') setupAltDrag(ctrlId);
    }

    // Config editor: parse on blur or Ctrl+Enter
    editor.addEventListener('blur', parseEditorToConfig);
    editor.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 'Enter') {
        e.preventDefault();
        parseEditorToConfig();
      }
    });

    // Reset button
    $('btn-reset-config').addEventListener('click', async () => {
      try {
        const resp = await fetch('/api/config/default');
        if (resp.ok) {
          const data = await resp.json();
          config = data.config;
          renderConfig();
          syncConfigToControls();
        }
      } catch (e) {
        // ignore
      }
    });
  }

  function getConfig() {
    return JSON.parse(JSON.stringify(config));
  }

  function setConfig(newConfig) {
    config = JSON.parse(JSON.stringify(newConfig));
    renderConfig();
    syncConfigToControls();
  }

  return {
    init,
    getConfig,
    setConfig,
    renderConfig,
    updateResolutionPreview,
    highlightJSON,
  };
})();
