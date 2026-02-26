/* arrows.js — Procedural SVG arrow animation engine.
 *
 * Connects DOM elements with animated bezier-curve SVG arrows.
 * Uses spring physics for bounce on submit, steady pulse while processing.
 * Color-codes by state: idle (gray), flowing (blue), complete (green), error (red).
 */

const Arrows = (() => {
  const canvas = document.getElementById('arrow-canvas');
  const arrows = {};
  let animId = null;

  const STATE_COLORS = {
    idle:     { stroke: '#555',    marker: 'arrowhead-idle' },
    flowing:  { stroke: '#4a9eff', marker: 'arrowhead-flow' },
    complete: { stroke: '#4ade80', marker: 'arrowhead-done' },
    error:    { stroke: '#f87171', marker: 'arrowhead-error' },
  };

  function createArrow(id, fromEl, toEl) {
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke-width', '2');
    path.setAttribute('stroke-linecap', 'round');
    canvas.appendChild(path);

    arrows[id] = {
      path,
      fromEl,
      toEl,
      state: 'idle',
      phase: 0,
      spring: { x: 0, v: 0, target: 0 },
      opacity: 0.4,
    };

    updateArrowPath(id);
    applyState(id, 'idle');
    return arrows[id];
  }

  function getCenter(el) {
    const r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }

  function getEdge(el, side) {
    const r = el.getBoundingClientRect();
    switch (side) {
      case 'right':  return { x: r.right,     y: r.top + r.height / 2 };
      case 'left':   return { x: r.left,      y: r.top + r.height / 2 };
      case 'bottom': return { x: r.left + r.width / 2, y: r.bottom };
      case 'top':    return { x: r.left + r.width / 2, y: r.top };
      default:       return getCenter(el);
    }
  }

  function updateArrowPath(id) {
    const a = arrows[id];
    if (!a) return;

    const from = getEdge(a.fromEl, 'right');
    const to = getEdge(a.toEl, 'left');

    const dx = to.x - from.x;
    const cp = Math.max(40, Math.abs(dx) * 0.4);

    // Add spring offset to control points for bounce effect
    const bounce = a.spring.x * 8;

    const d = `M ${from.x} ${from.y} C ${from.x + cp} ${from.y + bounce}, ${to.x - cp} ${to.y - bounce}, ${to.x} ${to.y}`;
    a.path.setAttribute('d', d);
  }

  function applyState(id, state) {
    const a = arrows[id];
    if (!a) return;

    a.state = state;
    const colors = STATE_COLORS[state] || STATE_COLORS.idle;
    a.path.setAttribute('stroke', colors.stroke);
    a.path.setAttribute('marker-end', `url(#${colors.marker})`);

    if (state === 'flowing') {
      a.path.classList.add('arrow-flowing');
      a.path.setAttribute('stroke-width', '3');
    } else {
      a.path.classList.remove('arrow-flowing');
      a.path.setAttribute('stroke-width', '2');
    }

    if (state === 'complete') {
      a.path.setAttribute('stroke-width', '3');
      // Flash green then fade
      setTimeout(() => {
        if (a.state === 'complete') {
          applyState(id, 'idle');
        }
      }, 2000);
    }

    if (state === 'error') {
      a.path.setAttribute('stroke-width', '3');
      setTimeout(() => {
        if (a.state === 'error') {
          applyState(id, 'idle');
        }
      }, 3000);
    }
  }

  function triggerBounce(id) {
    const a = arrows[id];
    if (!a) return;
    a.spring.v = 6; // impulse
  }

  function tick() {
    const dt = 1 / 60;
    const stiffness = 180;
    const damping = 12;

    for (const id in arrows) {
      const a = arrows[id];

      // Spring physics
      const s = a.spring;
      const force = -stiffness * (s.x - s.target) - damping * s.v;
      s.v += force * dt;
      s.x += s.v * dt;

      // Flowing state: gentle oscillation
      if (a.state === 'flowing') {
        a.phase += dt * 3;
        s.target = Math.sin(a.phase) * 0.5;
      } else {
        s.target = 0;
      }

      updateArrowPath(id);
    }

    animId = requestAnimationFrame(tick);
  }

  function start() {
    if (!animId) {
      animId = requestAnimationFrame(tick);
    }
  }

  function stop() {
    if (animId) {
      cancelAnimationFrame(animId);
      animId = null;
    }
  }

  // Re-layout on resize
  window.addEventListener('resize', () => {
    for (const id in arrows) {
      updateArrowPath(id);
    }
  });

  return {
    create: createArrow,
    setState: applyState,
    bounce: triggerBounce,
    start,
    stop,
  };
})();
