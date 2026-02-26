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
  // Syntax highlighting
  // ---------------------------------------------------------------------------

  function highlightJSON(json) {
    // Escape HTML first
    const escaped = json
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    return escaped
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
      // Array numbers (in arrays like [8, 30])
      .replace(/\[([^\]]*)\]/g, (match) => {
        return match.replace(/(-?\d+\.?\d*)/g, '<span class="cfg-num">$1</span>');
      });
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

  function syncConfigToControls() {
    suppressSync = true;

    const $ = (id) => document.getElementById(id);

    if (config.prompt !== undefined)
      $('ctrl-prompt').value = config.prompt;
    if (config.negative_prompt !== undefined)
      $('ctrl-neg-prompt').value = config.negative_prompt;
    if (config.seed !== undefined)
      $('ctrl-seed').value = config.seed;
    if (config.n_steps !== undefined) {
      $('ctrl-steps').value = config.n_steps;
      $('steps-value').textContent = config.n_steps;
    }
    if (config.cfg !== undefined) {
      $('ctrl-cfg').value = config.cfg;
      $('cfg-value').textContent = config.cfg;
    }
    if (config.attention_backend !== undefined)
      $('ctrl-backend').value = config.attention_backend;
    if (config.sampling_shift !== undefined) {
      $('ctrl-shift').value = config.sampling_shift;
      $('shift-value').textContent = config.sampling_shift;
    }
    if (config.denoise !== undefined) {
      $('ctrl-denoise').value = config.denoise;
      $('denoise-value').textContent = config.denoise;
    }
    if (config.multiplier !== undefined) {
      // No slider for multiplier yet
    }

    // Resolution
    if (config.resolution) {
      const res = config.resolution;
      if (res.megapixels !== undefined) {
        $('ctrl-anchor').value = res.megapixels;
      }
      if (res.aspect_ratio !== undefined) {
        const logVal = Math.log(res.aspect_ratio);
        $('ctrl-aspect').value = logVal;
        updateAspectLabel(res.aspect_ratio);
      }
    }

    // Show i2i group if denoise < 1
    const i2iGroup = $('i2i-group');
    if (config.denoise !== undefined && config.denoise < 1.0) {
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

    config.prompt = $('ctrl-prompt').value;
    config.negative_prompt = $('ctrl-neg-prompt').value;
    config.seed = parseInt($('ctrl-seed').value, 10);
    config.n_steps = parseInt($('ctrl-steps').value, 10);
    config.cfg = parseFloat($('ctrl-cfg').value);
    config.attention_backend = $('ctrl-backend').value;
    config.sampling_shift = parseFloat($('ctrl-shift').value);
    config.denoise = parseFloat($('ctrl-denoise').value);

    // Resolution
    if (!config.resolution) config.resolution = {};
    config.resolution.megapixels = parseInt($('ctrl-anchor').value, 10);
    config.resolution.aspect_ratio = Math.exp(parseFloat($('ctrl-aspect').value));
    config.resolution.quantize = 32;

    renderConfig();
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

  // ---------------------------------------------------------------------------
  // Resolution preview (calls BFF endpoint)
  // ---------------------------------------------------------------------------

  let resDebounce = null;

  function updateResolutionPreview() {
    clearTimeout(resDebounce);
    resDebounce = setTimeout(async () => {
      const anchor = config.resolution?.megapixels || 1048576;
      const aspect = config.resolution?.aspect_ratio || 1.0;

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

    // Text inputs
    $('ctrl-prompt').addEventListener('input', syncControlsToConfig);
    $('ctrl-neg-prompt').addEventListener('input', syncControlsToConfig);
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

    // Selects
    $('ctrl-backend').addEventListener('change', syncControlsToConfig);
    $('ctrl-anchor').addEventListener('change', syncControlsToConfig);

    // Aspect slider
    $('ctrl-aspect').addEventListener('input', () => {
      const ratio = Math.exp(parseFloat($('ctrl-aspect').value));
      updateAspectLabel(ratio);
      syncControlsToConfig();
    });

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
  };
})();
