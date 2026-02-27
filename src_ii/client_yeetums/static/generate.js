/* generate.js — Batch-first generation: k jobs, k individual SSE streams.
 *
 * Flow:
 *   1. POST /api/batch_generate → {batch_id, k, jobs: [{job_id, stream_url, ...}, ...]}
 *   2. Open k individual EventSource connections (one per job, through /api/stream/{job_id})
 *   3. On each 'gallery_ready' → add image to gallery (with batch grouping if k>1)
 *   4. When all k complete → reset UI
 *
 * No multiplexed streams. ONE streaming path: /api/stream/{job_id}.
 * The queue is the scatter. The gallery is the gather.
 */

const Generate = (() => {

  let isGenerating = false;
  let currentSourceId = null;
  let activeEventSources = [];

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------
  // Submission (always through batch path)
  // ---------------------------------------------------------------------------

  async function submit() {
    if (isGenerating) return;

    const config = ConfigFlow.getConfig();

    // Prompt may be a string (scalar) or array (enum distribution)
    const promptEmpty = Array.isArray(config.prompt)
      ? config.prompt.length === 0 || config.prompt.every(p => !p.trim())
      : !config.prompt || !config.prompt.trim();
    if (promptEmpty) {
      App.logActivity('No prompt provided', 'err');
      return;
    }

    const k = Math.max(1, Math.min(16, config.k || 1));

    isGenerating = true;
    const btn = $('btn-generate');
    btn.disabled = true;
    btn.textContent = k > 1 ? `Queuing ${k} draws...` : 'Queuing...';
    btn.classList.add('btn-generating');

    Arrows.setState('arrow-config-controls', 'flowing');
    Arrows.bounce('arrow-config-controls');

    const promptPreview = Array.isArray(config.prompt)
      ? `[${config.prompt.length}] ${config.prompt[0].slice(0, 30)}`
      : config.prompt.slice(0, 40);
    App.logActivity(`Submitting batch (k=${k}): ` + promptPreview, 'msg');

    try {
      // Step 1: POST distributional config to batch endpoint
      const resp = await fetch('/api/batch_generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Batch enqueue failed');
      }

      const result = await resp.json();
      const batchId = result.batch_id;

      btn.textContent = k > 1 ? `Generating 0/${k}...` : 'Generating...';
      Arrows.setState('arrow-controls-accel', 'flowing');
      Arrows.bounce('arrow-controls-accel');

      App.logActivity(`Batch ${batchId}: ${k} job(s) queued`, 'msg');

      // Prepare batch group in gallery with resolved configs visible.
      // The template config lets the gallery identify distributional fields.
      // Always create the group — even k=1 shows resolved config before image.
      Gallery.startBatchGroup(batchId, k, result.jobs, config);

      // Show first job's resolved config in output panel
      if (result.jobs.length > 0 && result.jobs[0].resolved_config) {
        OutputConfig.show(result.jobs[0].resolved_config, `seed ${result.jobs[0].seed}`);
      }

      // Step 2: Open k individual SSE streams (scatter)
      await streamAllJobs(result.jobs, batchId, k);

    } catch (e) {
      Arrows.setState('arrow-config-controls', 'error');
      Arrows.setState('arrow-controls-accel', 'error');
      App.logActivity('Error: ' + e.message, 'err');
      resetUI();
    }
  }

  // ---------------------------------------------------------------------------
  // k individual SSE streams (one per job, gathered by completion counting)
  // ---------------------------------------------------------------------------

  function streamAllJobs(jobs, batchId, k) {
    return new Promise((resolve) => {
      // Close any leftover streams
      closeAllStreams();

      let completedCount = 0;
      let errorCount = 0;

      for (const job of jobs) {
        const streamUrl = job.stream_url +
          `?batch_id=${encodeURIComponent(batchId)}` +
          `&batch_index=${job.batch_index}`;

        const es = new EventSource(streamUrl);
        activeEventSources.push(es);

        es.addEventListener('encoding', () => {
          if (k === 1) $('btn-generate').textContent = 'Encoding...';
        });

        es.addEventListener('progress', (e) => {
          try {
            const d = JSON.parse(e.data);
            const step = d.step || 0;
            const total = d.total_steps || 30;
            if (k === 1) {
              $('btn-generate').textContent = `Sampling ${step}/${total}`;
            }
          } catch (err) { /* ignore */ }
        });

        es.addEventListener('sampling', () => {
          if (k === 1) $('btn-generate').textContent = 'Sampling...';
        });

        es.addEventListener('decoding', () => {
          Arrows.setState('arrow-accel-gallery', 'flowing');
          Arrows.bounce('arrow-accel-gallery');
          if (k === 1) $('btn-generate').textContent = 'Decoding...';
        });

        es.addEventListener('complete', () => {
          App.logActivity(`[${job.batch_index}] Complete, saving...`, 'msg');
        });

        es.addEventListener('gallery_ready', (e) => {
          es.close();
          removeStream(es);

          try {
            const d = JSON.parse(e.data);
            completedCount++;

            App.logActivity(
              `[${job.batch_index}] ` +
              `${d.width}x${d.height}, seed=${d.seed}, ${d.elapsed_s}s`,
              'ok'
            );

            // Route to gallery: always through batch group (we always create one)
            Gallery.addToBatchGroup(d);

            if (k > 1) {
              $('btn-generate').textContent = `Generating ${completedCount}/${k}...`;
            }
          } catch (err) {
            App.logActivity('Gallery update error: ' + err.message, 'err');
          }

          checkAllDone();
        });

        es.addEventListener('error', (e) => {
          let msg = 'Generation error';
          if (e.data) {
            try {
              const d = JSON.parse(e.data);
              msg = d.error || msg;
            } catch (err) { /* ignore */ }
          }
          App.logActivity(`[${job.batch_index}] Error: ` + msg, 'err');
          es.close();
          removeStream(es);
          errorCount++;
          checkAllDone();
        });

        es.onerror = () => {
          if (es.readyState === EventSource.CLOSED) return;
          es.close();
          removeStream(es);
          App.logActivity(`[${job.batch_index}] SSE connection lost`, 'err');
          errorCount++;
          checkAllDone();
        };
      }

      function checkAllDone() {
        if (completedCount + errorCount >= k) {
          Arrows.setState('arrow-config-controls', 'complete');
          Arrows.setState('arrow-controls-accel', 'complete');
          Arrows.setState('arrow-accel-gallery', 'complete');
          App.logActivity(`Batch done: ${completedCount}/${k} images`, 'ok');
          resetUI();
          resolve();
        }
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Stream cleanup
  // ---------------------------------------------------------------------------

  function closeAllStreams() {
    for (const es of activeEventSources) {
      es.close();
    }
    activeEventSources = [];
  }

  function removeStream(es) {
    const idx = activeEventSources.indexOf(es);
    if (idx >= 0) activeEventSources.splice(idx, 1);
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
