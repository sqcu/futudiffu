/* gallery.js — Thumbnail grid with batch grouping, metadata overlay, fullscreen viewer.
 *
 * Renders gallery entries as film-strip thumbnails in the bottom panel.
 * Batch groups (k>1) are rendered as magenta-bordered groups with seed labels.
 * Click to view fullscreen with metadata overlay.
 */

const Gallery = (() => {

  const strip = document.getElementById('gallery-strip');
  const countEl = document.getElementById('gallery-count');
  const overlay = document.getElementById('fullscreen-overlay');
  const fsImg = document.getElementById('fullscreen-img');
  const fsMeta = document.getElementById('fullscreen-meta');
  const fsClose = document.getElementById('fullscreen-close');

  let entries = [];
  // batch_id -> { element, k, jobs, filled }
  const batchGroups = {};

  // ---------------------------------------------------------------------------
  // Add entry (standalone, from generation result)
  // ---------------------------------------------------------------------------

  function addEntry(result) {
    // Route to batch group if it has a batch_id
    if (result.batch_id && batchGroups[result.batch_id]) {
      addToBatchGroup(result);
      return;
    }

    const entry = {
      id: result.gallery_id,
      prompt: result.prompt,
      negative_prompt: result.negative_prompt || (result.resolved_config && result.resolved_config.negative_prompt) || '',
      seed: result.seed,
      width: result.width,
      height: result.height,
      image_url: result.image_url,
      elapsed_s: result.elapsed_s,
      batch_id: result.batch_id || null,
      batch_index: result.batch_index !== undefined ? result.batch_index : null,
      resolved_config: result.resolved_config || null,
    };

    entries.unshift(entry);
    renderThumb(entry, true);
    updateCount();
  }

  // ---------------------------------------------------------------------------
  // Batch group management
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // Resolved config card: show what was sampled from distributions
  // ---------------------------------------------------------------------------

  // Distribution detection (mirrors config_flow.js / config_distributions.py)
  function _isDist(v) {
    return v && typeof v === 'object' && !Array.isArray(v) &&
           ('min' in v || 'max' in v || 'values' in v || 'weights' in v);
  }
  function _isEnum(v) { return Array.isArray(v); }

  /**
   * Build a compact config card showing resolved values for distributional fields.
   * Only shows fields that were distributional in the template — scalar fields
   * are the same for every draw and don't need per-slot display.
   */
  function _buildConfigCard(job, templateConfig) {
    const card = document.createElement('div');
    card.className = 'slot-config-card';

    const resolved = job.resolved_config || {};

    // Always show seed (it's the primary identity of a draw)
    const seedLine = document.createElement('div');
    seedLine.className = 'config-card-line config-card-seed';
    seedLine.textContent = `seed ${job.seed}`;
    card.appendChild(seedLine);

    // Show resolved values for distributional fields
    const distFields = _getDistributionalFields(templateConfig);
    for (const key of distFields) {
      if (key === 'seed') continue; // already shown
      const val = resolved[key];
      if (val === undefined) continue;

      const line = document.createElement('div');
      line.className = 'config-card-line config-card-sampled';
      line.textContent = `${_shortKey(key)} ${_formatVal(val)}`;
      card.appendChild(line);
    }

    // Show resolution if it differs across jobs (distributional resolution)
    if (job.width && job.height) {
      const resLine = document.createElement('div');
      resLine.className = 'config-card-line config-card-dim';
      resLine.textContent = `${job.width}\u00d7${job.height}`;
      card.appendChild(resLine);
    }

    return card;
  }

  /** Identify which top-level fields in the template are distributional. */
  function _getDistributionalFields(config) {
    const dist = [];
    for (const [key, val] of Object.entries(config || {})) {
      if (key === 'resolution' || key === 'prompt' || key === 'negative_prompt' || key === 'k') continue;
      if (_isDist(val) || _isEnum(val)) dist.push(key);
    }
    return dist;
  }

  /** Strip gallery bookkeeping, keep only generation-config fields. */
  const _GALLERY_INTERNAL = new Set([
    'id', 'image_url', 'timestamp', 'batch_id', 'batch_index',
    'resolved_config', 'source_id', 'gallery_id', 'prompt',
  ]);

  function _entryToConfig(entry) {
    const out = {};
    for (const [k, v] of Object.entries(entry)) {
      if (_GALLERY_INTERNAL.has(k)) continue;
      if (v == null) continue;
      out[k] = v;
    }
    return out;
  }

  function _shortKey(key) {
    const shorts = {
      n_steps: 'steps', cfg: 'cfg', sampling_shift: 'shift',
      denoise: 'denoise', multiplier: 'mult', attention_backend: 'attn',
    };
    return shorts[key] || key;
  }

  function _formatVal(val) {
    if (typeof val === 'number') {
      return Number.isInteger(val) ? String(val) : val.toFixed(2);
    }
    return String(val);
  }

  // ---------------------------------------------------------------------------
  // Batch group lifecycle
  // ---------------------------------------------------------------------------

  function startBatchGroup(batchId, k, jobs, templateConfig) {
    // Remove empty message
    const empty = strip.querySelector('.gallery-empty');
    if (empty) empty.remove();

    const group = document.createElement('div');
    group.className = 'gallery-batch-group';
    group.dataset.batchId = batchId;

    const label = document.createElement('div');
    label.className = 'batch-group-label';
    label.textContent = k === 1 ? '1 draw' : `${k} draws`;
    group.appendChild(label);

    const slotsContainer = document.createElement('div');
    slotsContainer.className = 'batch-slots';

    for (let i = 0; i < k; i++) {
      const slot = document.createElement('div');
      slot.className = 'batch-slot';
      slot.dataset.batchIndex = i;

      const job = jobs[i];
      if (job && templateConfig) {
        // Diegetic presentation: show the resolved config card
        slot.appendChild(_buildConfigCard(job, templateConfig));
      } else {
        const pending = document.createElement('div');
        pending.className = 'slot-pending';
        pending.textContent = job ? `seed:${job.seed}` : `#${i}`;
        slot.appendChild(pending);
      }

      slotsContainer.appendChild(slot);
    }

    group.appendChild(slotsContainer);
    strip.prepend(group);
    strip.scrollLeft = 0;

    batchGroups[batchId] = {
      element: group,
      k,
      jobs,
      filled: new Set(),
    };
  }

  function addToBatchGroup(result) {
    const batchId = result.batch_id;
    const batchIndex = result.batch_index;
    const group = batchGroups[batchId];

    if (!group) {
      // Fallback: render as standalone
      addEntry({ ...result, batch_id: null });
      return;
    }

    const slot = group.element.querySelector(
      `.batch-slot[data-batch-index="${batchIndex}"]`
    );
    if (!slot) return;

    // Clear pending text
    slot.innerHTML = '';

    // Add image
    const img = document.createElement('img');
    img.src = result.image_url;
    img.alt = result.prompt || 'Generated';
    img.loading = 'lazy';
    slot.appendChild(img);

    // Add overlay
    const overlayDiv = document.createElement('div');
    overlayDiv.className = 'thumb-overlay';
    overlayDiv.textContent = `${result.seed || '?'} | ${result.width || '?'}x${result.height || '?'}`;
    slot.appendChild(overlayDiv);

    // Click to fullscreen
    const entry = {
      id: result.gallery_id,
      prompt: result.prompt,
      negative_prompt: result.negative_prompt || (result.resolved_config && result.resolved_config.negative_prompt) || '',
      seed: result.seed,
      width: result.width,
      height: result.height,
      image_url: result.image_url,
      elapsed_s: result.elapsed_s,
      batch_id: batchId,
      batch_index: batchIndex,
      resolved_config: result.resolved_config || null,
    };
    slot.addEventListener('click', () => openFullscreen(entry));

    entries.unshift(entry);
    group.filled.add(batchIndex);
    updateCount();
  }

  // ---------------------------------------------------------------------------
  // Render (standalone thumbs)
  // ---------------------------------------------------------------------------

  function renderThumb(entry, prepend) {
    // Remove empty message
    const empty = strip.querySelector('.gallery-empty');
    if (empty) empty.remove();

    const thumb = document.createElement('div');
    thumb.className = 'gallery-thumb';
    thumb.dataset.id = entry.id;

    thumb.innerHTML = `
      <div class="thumb-perf-left"></div>
      <div class="thumb-perf-right"></div>
      <img src="${entry.image_url}" alt="${entry.prompt || 'Generated'}" loading="lazy">
      <div class="thumb-overlay">${entry.seed || '?'} | ${entry.width || '?'}x${entry.height || '?'}</div>
    `;

    thumb.addEventListener('click', () => openFullscreen(entry));

    if (prepend) {
      strip.prepend(thumb);
      // Scroll to start
      strip.scrollLeft = 0;
    } else {
      strip.appendChild(thumb);
    }
  }

  function renderAll() {
    strip.innerHTML = '';
    // Clear batch groups on full re-render
    Object.keys(batchGroups).forEach(k => delete batchGroups[k]);

    if (entries.length === 0) {
      strip.innerHTML = '<div class="gallery-empty">No images yet. Generate something!</div>';
      updateCount();
      return;
    }

    // Group entries by batch_id for reconstruction
    const batches = {};
    const standalone = [];

    for (const entry of entries) {
      if (entry.batch_id) {
        if (!batches[entry.batch_id]) batches[entry.batch_id] = [];
        batches[entry.batch_id].push(entry);
      } else {
        standalone.push(entry);
      }
    }

    // Render batch groups first (newest entries are first in the array)
    const renderedBatchIds = new Set();
    for (const entry of entries) {
      if (entry.batch_id && !renderedBatchIds.has(entry.batch_id)) {
        renderedBatchIds.add(entry.batch_id);
        const batchEntries = batches[entry.batch_id];
        const k = batchEntries.length;

        // Reconstruct batch group
        const jobs = batchEntries.map((e, i) => ({
          seed: e.seed,
          batch_index: e.batch_index !== null ? e.batch_index : i,
        }));
        startBatchGroup(entry.batch_id, k, jobs);

        for (const be of batchEntries) {
          addToBatchGroup({
            gallery_id: be.id,
            prompt: be.prompt,
            seed: be.seed,
            width: be.width,
            height: be.height,
            image_url: be.image_url,
            elapsed_s: be.elapsed_s,
            batch_id: be.batch_id,
            batch_index: be.batch_index,
            resolved_config: be.resolved_config || null,
          });
        }
      } else if (!entry.batch_id) {
        renderThumb(entry, false);
      }
    }

    updateCount();
  }

  function updateCount() {
    countEl.textContent = `${entries.length} image${entries.length !== 1 ? 's' : ''}`;
  }

  // ---------------------------------------------------------------------------
  // Fullscreen
  // ---------------------------------------------------------------------------

  function openFullscreen(entry) {
    fsImg.src = entry.image_url;
    fsMeta.innerHTML = [
      `<strong>${entry.prompt || '(no prompt)'}</strong>`,
      entry.negative_prompt ? `<em class="text-dim">neg: ${entry.negative_prompt}</em>` : '',
      `seed: ${entry.seed} | ${entry.width}x${entry.height}`,
      entry.elapsed_s ? `${entry.elapsed_s}s` : '',
      entry.n_steps ? `${entry.n_steps} steps` : '',
      entry.cfg ? `cfg ${entry.cfg}` : '',
      entry.attention_backend || '',
      entry.batch_id ? `batch: ${entry.batch_id}[${entry.batch_index}]` : '',
    ].filter(Boolean).join(' &middot; ');

    overlay.style.display = 'flex';

    // Show resolved config, or extract config-like fields for pre-feature images
    const cfg = entry.resolved_config || _entryToConfig(entry);
    OutputConfig.show(cfg, `seed ${entry.seed}`);
  }

  function closeFullscreen() {
    overlay.style.display = 'none';
    fsImg.src = '';
  }

  // ---------------------------------------------------------------------------
  // Load existing gallery
  // ---------------------------------------------------------------------------

  async function loadExisting() {
    try {
      const resp = await fetch('/api/gallery?limit=50');
      if (!resp.ok) return;
      const data = await resp.json();
      entries = data.entries || [];
      renderAll();
    } catch (e) {
      // Gallery load failed — not critical
      renderAll();
    }
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------

  function init() {
    fsClose.addEventListener('click', closeFullscreen);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) closeFullscreen();
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && overlay.style.display !== 'none') {
        closeFullscreen();
      }
    });

    loadExisting();
  }

  return { init, addEntry, addToBatchGroup, startBatchGroup, loadExisting };
})();
