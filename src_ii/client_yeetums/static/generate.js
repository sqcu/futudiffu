/* generate.js — t2i/i2i form submission with SSE streaming progress.
 *
 * Flow:
 *   1. POST /api/generate → {job_id, stream_url, seed, width, height}
 *   2. EventSource(stream_url) → encoding/progress/decoding/complete events
 *   3. On 'gallery_ready' → add image to gallery
 */

const Generate = (() => {

  let isGenerating = false;
  let currentSourceId = null;
  let activeEventSource = null;

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------
  // Submission (queue-based with SSE streaming)
  // ---------------------------------------------------------------------------

  async function submit() {
    if (isGenerating) return;

    const config = ConfigFlow.getConfig();

    if (!config.prompt || !config.prompt.trim()) {
      App.logActivity('No prompt provided', 'err');
      return;
    }

    isGenerating = true;
    const btn = $('btn-generate');
    btn.disabled = true;
    btn.textContent = 'Queuing...';
    btn.classList.add('btn-generating');

    Arrows.setState('arrow-config-controls', 'flowing');
    Arrows.bounce('arrow-config-controls');

    App.logActivity('Submitting: ' + config.prompt.slice(0, 40), 'msg');

    const res = config.resolution || {};
    const body = {
      prompt: config.prompt,
      negative_prompt: config.negative_prompt || '',
      seed: config.seed !== undefined ? config.seed : -1,
      n_steps: config.n_steps || 30,
      cfg: config.cfg || 4.0,
      attention_backend: config.attention_backend || 'sage',
      sampling_shift: config.sampling_shift || 1.0,
      multiplier: config.multiplier || 1.0,
      denoise: config.denoise || 1.0,
      anchor_pixels: res.megapixels || 1048576,
      aspect_ratio: res.aspect_ratio || 1.0,
      source_id: currentSourceId,
    };

    try {
      // Step 1: Enqueue the job
      const resp = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Enqueue failed');
      }

      const result = await resp.json();
      const jobId = result.job_id;

      // Update seed if it was random
      if (config.seed === -1 || config.seed < 0) {
        const newConfig = ConfigFlow.getConfig();
        newConfig.seed = result.seed;
        ConfigFlow.setConfig(newConfig);
      }

      btn.textContent = 'Generating...';
      Arrows.setState('arrow-controls-accel', 'flowing');
      Arrows.bounce('arrow-controls-accel');

      App.logActivity('Job queued: ' + jobId, 'msg');

      // Step 2: Connect to SSE stream
      await streamJob(result.stream_url, result);

    } catch (e) {
      Arrows.setState('arrow-config-controls', 'error');
      Arrows.setState('arrow-controls-accel', 'error');
      App.logActivity('Error: ' + e.message, 'err');
      resetUI();
    }
  }

  // ---------------------------------------------------------------------------
  // SSE event streaming
  // ---------------------------------------------------------------------------

  function streamJob(streamUrl, enqueueResult) {
    return new Promise((resolve, reject) => {
      if (activeEventSource) {
        activeEventSource.close();
      }

      const es = new EventSource(streamUrl);
      activeEventSource = es;

      es.addEventListener('encoding', () => {
        $('btn-generate').textContent = 'Encoding...';
        App.logActivity('Encoding prompts...', 'msg');
      });

      es.addEventListener('progress', (e) => {
        try {
          const d = JSON.parse(e.data);
          const step = d.step || 0;
          const total = d.total_steps || enqueueResult.n_steps || 30;
          $('btn-generate').textContent = `Sampling ${step}/${total}`;
        } catch (err) { /* ignore parse errors */ }
      });

      es.addEventListener('sampling', () => {
        $('btn-generate').textContent = 'Sampling...';
      });

      es.addEventListener('decoding', () => {
        $('btn-generate').textContent = 'Decoding...';
        Arrows.setState('arrow-accel-gallery', 'flowing');
        Arrows.bounce('arrow-accel-gallery');
      });

      es.addEventListener('complete', (e) => {
        App.logActivity('Generation complete, saving to gallery...', 'msg');
      });

      es.addEventListener('gallery_ready', (e) => {
        es.close();
        activeEventSource = null;

        try {
          const d = JSON.parse(e.data);

          Arrows.setState('arrow-config-controls', 'complete');
          Arrows.setState('arrow-controls-accel', 'complete');
          Arrows.setState('arrow-accel-gallery', 'complete');

          App.logActivity(
            `Done: ${d.width}x${d.height}, seed=${d.seed}, ${d.elapsed_s}s`,
            'ok'
          );

          Gallery.addEntry(d);
        } catch (err) {
          App.logActivity('Gallery update error: ' + err.message, 'err');
        }

        resetUI();
        resolve();
      });

      es.addEventListener('error', (e) => {
        es.close();
        activeEventSource = null;

        let msg = 'Generation error';
        if (e.data) {
          try {
            const d = JSON.parse(e.data);
            msg = d.error || msg;
          } catch (err) { /* ignore */ }
        }

        Arrows.setState('arrow-config-controls', 'error');
        Arrows.setState('arrow-controls-accel', 'error');
        App.logActivity('Error: ' + msg, 'err');

        resetUI();
        reject(new Error(msg));
      });

      // EventSource built-in error (connection failure)
      es.onerror = () => {
        if (es.readyState === EventSource.CLOSED) return;
        es.close();
        activeEventSource = null;
        Arrows.setState('arrow-controls-accel', 'error');
        App.logActivity('SSE connection lost', 'err');
        resetUI();
        reject(new Error('SSE connection lost'));
      };
    });
  }

  // ---------------------------------------------------------------------------
  // UI reset
  // ---------------------------------------------------------------------------

  function resetUI() {
    isGenerating = false;
    const btn = $('btn-generate');
    btn.disabled = false;
    btn.textContent = 'Generate';
    btn.classList.remove('btn-generating');
  }

  // ---------------------------------------------------------------------------
  // i2i source upload
  // ---------------------------------------------------------------------------

  async function uploadSource(file) {
    const formData = new FormData();
    formData.append('file', file);

    App.logActivity('Uploading source image...', 'msg');

    try {
      const resp = await fetch('/api/upload_source', {
        method: 'POST',
        body: formData,
      });

      if (!resp.ok) throw new Error('Upload failed');

      const data = await resp.json();
      currentSourceId = data.source_id;

      // Show preview
      const reader = new FileReader();
      reader.onload = (e) => {
        const preview = $('i2i-preview');
        preview.src = e.target.result;
        preview.style.display = '';
      };
      reader.readAsDataURL(file);

      $('i2i-status').textContent = `Source: ${currentSourceId}`;
      App.logActivity('Source uploaded: ' + currentSourceId, 'ok');
    } catch (e) {
      App.logActivity('Upload error: ' + e.message, 'err');
    }
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------

  function init() {
    $('btn-generate').addEventListener('click', submit);

    // Keyboard shortcut: Ctrl+Enter to generate
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.key === 'Enter' && !isGenerating) {
        e.preventDefault();
        submit();
      }
    });

    // i2i upload
    $('btn-upload-source').addEventListener('click', () => {
      $('i2i-file').click();
    });

    $('i2i-file').addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) uploadSource(file);
    });
  }

  return { init, submit };
})();
