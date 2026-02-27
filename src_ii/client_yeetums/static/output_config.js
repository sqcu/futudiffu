/* output_config.js — Read-only resolved config display panel.
 *
 * Shows the concrete scalar values drawn from distributional configs.
 * Reuses ConfigFlow.highlightJSON() for syntax highlighting.
 */

const OutputConfig = (() => {
  const view = document.getElementById('output-config-view');
  const label = document.getElementById('output-config-label');
  let currentConfig = null;

  function show(config, labelText) {
    currentConfig = config;
    const json = JSON.stringify(config, null, 2);
    view.innerHTML = ConfigFlow.highlightJSON(json);
    label.textContent = labelText || '';
    label.className = 'text-dim';
  }

  function clear() {
    currentConfig = null;
    view.innerHTML = '';
    label.textContent = 'no selection';
    label.className = 'text-dim';
  }

  function getConfig() { return currentConfig; }

  function init() {}

  return { show, clear, getConfig, init };
})();
