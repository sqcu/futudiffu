/* gallery.js — Thumbnail grid, metadata overlay, fullscreen viewer.
 *
 * Renders gallery entries as film-strip thumbnails in the bottom panel.
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

  // ---------------------------------------------------------------------------
  // Add entry (from generation result)
  // ---------------------------------------------------------------------------

  function addEntry(result) {
    const entry = {
      id: result.gallery_id,
      prompt: result.prompt,
      seed: result.seed,
      width: result.width,
      height: result.height,
      image_url: result.image_url,
      elapsed_s: result.elapsed_s,
    };

    entries.unshift(entry);
    renderThumb(entry, true);
    updateCount();
  }

  // ---------------------------------------------------------------------------
  // Render
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
    if (entries.length === 0) {
      strip.innerHTML = '<div class="gallery-empty">No images yet. Generate something!</div>';
      updateCount();
      return;
    }
    for (const entry of entries) {
      renderThumb(entry, false);
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
      `seed: ${entry.seed} | ${entry.width}x${entry.height}`,
      entry.elapsed_s ? `${entry.elapsed_s}s` : '',
      entry.n_steps ? `${entry.n_steps} steps` : '',
      entry.cfg ? `cfg ${entry.cfg}` : '',
      entry.attention_backend || '',
    ].filter(Boolean).join(' &middot; ');

    overlay.style.display = 'flex';
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

  return { init, addEntry, loadExisting };
})();
