/* app.js — Core: status polling, tab routing, SVG arrow setup, initialization.
 *
 * Orchestrates all modules: ConfigFlow, Generate, Gallery, Arrows.
 * Polls the inference server status and updates the accelerator panel.
 */

const App = (() => {

  const $ = (id) => document.getElementById(id);

  let statusPollTimer = null;
  let lastConnected = false;

  // ---------------------------------------------------------------------------
  // Status polling
  // ---------------------------------------------------------------------------

  async function pollStatus() {
    try {
      const resp = await fetch('/api/status');
      if (!resp.ok) throw new Error('BFF unavailable');
      const data = await resp.json();
      updateAcceleratorPanel(data);
    } catch (e) {
      updateAcceleratorPanel({ connected: false });
    }
  }

  function updateAcceleratorPanel(status) {
    const connected = status.connected || false;

    // Top bar LED
    const topLed = $('status-led');
    const topText = $('status-text');
    if (connected) {
      topLed.className = 'led led-on';
      topLed.title = 'Server connected';
      topText.textContent = 'Connected';
    } else {
      topLed.className = 'led led-off';
      topLed.title = 'Server disconnected';
      topText.textContent = 'Disconnected';
    }

    // Accel panel connection LED
    $('accel-connection').className = connected ? 'led led-on' : 'led led-off';
    $('accel-phase').textContent = status.phase || (connected ? 'ready' : '--');
    $('accel-version').textContent = status.server_version || '--';

    // VRAM bar
    const total = status.vram_total_gb || 0;
    const alloc = status.vram_allocated_gb || 0;
    const pct = total > 0 ? (alloc / total * 100) : 0;
    const vramBar = $('vram-bar');
    vramBar.style.width = pct + '%';
    vramBar.className = 'vram-bar' + (pct > 90 ? ' crit' : pct > 70 ? ' warn' : '');
    $('vram-text').textContent = total > 0
      ? `${alloc.toFixed(1)} / ${total.toFixed(1)} GB`
      : '-- / -- GB';

    // Model badges
    const models = status.loaded_models || [];
    const badgeContainer = $('model-badges');
    const modelNames = ['diffusion', 'te', 'vae'];
    const modelLabels = { diffusion: 'diffusion', te: 'text encoder', vae: 'vae' };
    badgeContainer.innerHTML = modelNames.map(m => {
      const on = models.some(n => n.toLowerCase().includes(m));
      return `<span class="badge ${on ? 'badge-on' : 'badge-off'}">${modelLabels[m]}</span>`;
    }).join('');

    // Connection change logging
    if (connected && !lastConnected) {
      logActivity('Server connected', 'ok');
    } else if (!connected && lastConnected) {
      logActivity('Server disconnected', 'err');
    }
    lastConnected = connected;
  }

  // ---------------------------------------------------------------------------
  // Activity log
  // ---------------------------------------------------------------------------

  function logActivity(msg, type) {
    const log = $('activity-log');
    const entry = document.createElement('div');
    entry.className = 'activity-entry';

    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false });

    const typeClass = type === 'ok' ? 'activity-ok' : type === 'err' ? 'activity-err' : 'activity-msg';
    entry.innerHTML = `<span class="activity-time">${time}</span> <span class="${typeClass}">${msg}</span>`;

    log.prepend(entry);

    // Keep max 50 entries
    while (log.children.length > 50) {
      log.removeChild(log.lastChild);
    }
  }

  // ---------------------------------------------------------------------------
  // Tab routing
  // ---------------------------------------------------------------------------

  function setupTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabName = tab.dataset.tab;
        if (tabName === 'gallery') {
          // Expand gallery, collapse main layout
          document.querySelector('.main-layout').style.display = 'none';
          $('panel-gallery').style.maxHeight = 'calc(100vh - 60px)';
          $('panel-gallery').style.height = 'calc(100vh - 60px)';
          Gallery.loadExisting();
        } else {
          document.querySelector('.main-layout').style.display = '';
          $('panel-gallery').style.maxHeight = '';
          $('panel-gallery').style.height = '';
        }
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Arrow setup
  // ---------------------------------------------------------------------------

  function setupArrows() {
    // Config -> Controls
    Arrows.create('arrow-config-controls',
      $('panel-config'), $('panel-controls'));

    // Controls -> Accelerator
    Arrows.create('arrow-controls-accel',
      $('panel-controls'), $('panel-accel'));

    // Accelerator -> Gallery (vertical, bottom)
    Arrows.create('arrow-accel-gallery',
      $('panel-accel'), $('panel-gallery'));

    Arrows.start();
  }

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------

  async function init() {
    logActivity('Yeetums starting...', 'msg');

    // Load default config
    try {
      const resp = await fetch('/api/config/default');
      if (resp.ok) {
        const data = await resp.json();
        ConfigFlow.init(data.config);
      } else {
        ConfigFlow.init({
          prompt: '',
          negative_prompt: '',
          seed: -1,
          n_steps: 30,
          cfg: 4.0,
          attention_backend: 'sage',
          sampling_shift: 1.0,
          multiplier: 1.0,
          denoise: 1.0,
          resolution: { megapixels: 1048576, aspect_ratio: 1.5385, quantize: 32 },
        });
      }
    } catch (e) {
      ConfigFlow.init({
        prompt: '',
        seed: -1,
        n_steps: 30,
        cfg: 4.0,
        resolution: { megapixels: 1048576, aspect_ratio: 1.5385, quantize: 32 },
      });
    }

    Generate.init();
    Gallery.init();
    setupTabs();
    setupArrows();

    // Start status polling
    await pollStatus();
    statusPollTimer = setInterval(pollStatus, 5000);

    logActivity('Ready', 'ok');
  }

  // Run on load
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { logActivity, pollStatus };
})();
