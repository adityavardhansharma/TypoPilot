const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const root = document.documentElement;
root.classList.add('js');

const clamp = (value, min = 0, max = 1) => Math.min(max, Math.max(min, value));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = (t) => -(Math.cos(Math.PI * t) - 1) / 2;
const easeOut = (t) => 1 - Math.pow(1 - t, 4);

/* ---------- Masthead ---------- */

const masthead = document.querySelector('[data-header]');
const menuToggle = document.querySelector('.menu-toggle');
if (masthead && menuToggle) {
  const setMenu = (open) => {
    menuToggle.setAttribute('aria-expanded', String(open));
    menuToggle.textContent = open ? 'Close' : 'Menu';
    masthead.classList.toggle('is-open', open);
  };
  menuToggle.addEventListener('click', () => setMenu(menuToggle.getAttribute('aria-expanded') !== 'true'));
  masthead.querySelector('nav')?.addEventListener('click', (event) => {
    if (event.target.closest('a')) setMenu(false);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') setMenu(false);
  });
}

/* ---------- Scroll loop: one rAF-driven pass for every scroll-linked piece ---------- */

const scrollHandlers = new Set();
let scrollQueued = false;
const runScroll = () => {
  scrollQueued = false;
  masthead?.classList.toggle('is-scrolled', window.scrollY > 8);
  scrollHandlers.forEach((handler) => handler());
};
const requestScroll = () => {
  if (scrollQueued) return;
  scrollQueued = true;
  requestAnimationFrame(runScroll);
};
window.addEventListener('scroll', requestScroll, { passive: true });
window.addEventListener('resize', requestScroll, { passive: true });

// 0 when the element's top meets `start` (fraction of viewport), 1 when it meets `end`.
const viewProgress = (element, start = 0.9, end = 0.2) => {
  const rect = element.getBoundingClientRect();
  const h = window.innerHeight;
  return clamp((h * start - rect.top) / (h * (start - end)));
};

/* ---------- Reveal on view ---------- */

const onceVisible = (elements, callback, options = {}) => {
  if (!('IntersectionObserver' in window)) {
    elements.forEach(callback);
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      observer.unobserve(entry.target);
      callback(entry.target);
    });
  }, { rootMargin: '0px 0px -10% 0px', threshold: 0.15, ...options });
  elements.forEach((element) => observer.observe(element));
};

// Split headings into masked lines once fonts have settled, so line breaks are real.
const splitLines = (element) => {
  const words = element.innerHTML.trim().split(/\s+(?![^<]*>)/);
  element.innerHTML = words.map((word) => `<span class="w">${word}</span>`).join(' ');
  const lines = [];
  let lastTop = null;
  element.querySelectorAll('.w').forEach((word) => {
    const top = Math.round(word.offsetTop);
    if (top !== lastTop) {
      lines.push([]);
      lastTop = top;
    }
    lines[lines.length - 1].push(word.innerHTML);
  });
  element.innerHTML = lines
    .map((line, index) => `<span class="line" style="--line:${index}"><span>${line.join(' ')}</span></span>`)
    .join('');
};

const lineTargets = [...document.querySelectorAll('[data-lines]')];
(document.fonts?.ready ?? Promise.resolve()).then(() => {
  lineTargets.forEach(splitLines);
  onceVisible(lineTargets, (element) => element.classList.add('is-in'));
});
onceVisible([...document.querySelectorAll('[data-reveal]')], (element) => element.classList.add('is-in'));
requestScroll();

/* ==========================================================================
   Flight correction engine
   A sentence is aligned character-by-character with its correction. Runs of
   changed characters become segments; each segment morphs the moment the
   aircraft passes over it. The aircraft can be dragged to scrub both ways.
   ========================================================================== */

// Levenshtein alignment → alternating "keep" runs and changed segments.
const alignSegments = (from, to) => {
  const a = [...from];
  const b = [...to];
  const cost = Array.from({ length: a.length + 1 }, (_, i) => {
    const row = new Array(b.length + 1).fill(0);
    row[0] = i;
    return row;
  });
  for (let j = 0; j <= b.length; j += 1) cost[0][j] = j;
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      cost[i][j] = Math.min(
        cost[i - 1][j] + 1,
        cost[i][j - 1] + 1,
        cost[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
  }
  const ops = [];
  let i = a.length;
  let j = b.length;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1] && cost[i][j] === cost[i - 1][j - 1]) {
      ops.push({ keep: true, from: a[i - 1], to: b[j - 1] }); i -= 1; j -= 1;
    } else if (i > 0 && j > 0 && cost[i][j] === cost[i - 1][j - 1] + 1) {
      ops.push({ keep: false, from: a[i - 1], to: b[j - 1] }); i -= 1; j -= 1;
    } else if (i > 0 && cost[i][j] === cost[i - 1][j] + 1) {
      ops.push({ keep: false, from: a[i - 1], to: '' }); i -= 1;
    } else {
      ops.push({ keep: false, from: '', to: b[j - 1] }); j -= 1;
    }
  }
  ops.reverse();
  const pieces = [];
  ops.forEach((op) => {
    const last = pieces[pieces.length - 1];
    if (last && last.keep === op.keep) {
      last.from += op.from;
      last.to += op.to;
    } else {
      pieces.push({ ...op });
    }
  });
  return pieces;
};

const createFlight = (figure, examples, { loop = true, maxSize = 136, flightSeconds = 3.6 } = {}) => {
  const stage = figure.querySelector('[data-flight-stage]');
  const line = figure.querySelector('[data-flight-line]');
  const plane = figure.querySelector('[data-flight-plane]');
  const trail = figure.querySelector('[data-flight-trail]');
  const statusEl = figure.querySelector('[data-flight-status]');
  const editsEl = figure.querySelector('[data-flight-edits]');
  const numberEl = figure.querySelector('[data-flight-no]');
  const timeEl = figure.querySelector('[data-flight-time]');
  const liveEl = figure.querySelector('[data-flight-live]');
  const pager = figure.querySelector('[data-flight-pager]');

  let index = 0;
  let pieces = [];
  let lineLeft = 0;
  let stageWidth = 0;
  let planeWidth = 0;
  let margin = 0;
  let x = 0;
  let velocity = 0;
  let tilt = 0;
  let mode = 'cruise'; // cruise | drag | landed | swap
  let pointerX = null;
  let landedAt = 0;
  let running = false;
  let lastTime = 0;

  const setStatus = (label) => {
    if (statusEl && statusEl.textContent !== label) statusEl.textContent = label;
  };

  const pagerButtons = examples.map((_, n) => {
    if (!pager) return null;
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = String(n + 1).padStart(2, '0');
    button.setAttribute('aria-label', `Sentence ${n + 1}`);
    button.addEventListener('click', () => swapTo(n));
    pager.append(button);
    return button;
  });

  const measure = (text) => {
    const probe = document.createElement('span');
    probe.className = 'flight-probe';
    probe.textContent = text;
    line.append(probe);
    const width = probe.getBoundingClientRect().width;
    probe.remove();
    return width;
  };

  const build = () => {
    const example = examples[index];
    stageWidth = stage.clientWidth;
    const padding = parseFloat(getComputedStyle(stage).paddingLeft);
    const available = stageWidth - padding * 2;

    line.style.fontSize = '100px';
    line.replaceChildren();
    const widest = Math.max(measure(example.input), measure(example.output));
    const size = Math.min(maxSize, (100 * available) / widest);
    line.style.fontSize = `${size}px`;

    pieces = alignSegments(example.input, example.output).map((piece) => {
      if (piece.keep) {
        const el = document.createElement('span');
        el.className = 'run';
        el.textContent = piece.from;
        return { ...piece, el, fromWidth: measure(piece.from), toWidth: 0, fixed: false };
      }
      const el = document.createElement('span');
      el.className = 'seg';
      const fromEl = document.createElement('span');
      fromEl.className = 'from';
      fromEl.textContent = piece.from;
      const toEl = document.createElement('span');
      toEl.className = 'to';
      toEl.textContent = piece.to;
      el.append(fromEl, toEl);
      const fromWidth = piece.from ? measure(piece.from) : 0;
      const toWidth = piece.to ? measure(piece.to) : 0;
      el.style.width = `${fromWidth}px`;
      return { ...piece, el, fromWidth, toWidth, fixed: false };
    });
    pieces.forEach((piece) => line.append(piece.el));

    lineLeft = line.offsetLeft;
    planeWidth = Math.max(34, size * 0.62);
    margin = planeWidth * 1.4;
    plane.style.width = `${planeWidth}px`;
    const planeTop = line.offsetTop - planeWidth * 0.62;
    plane.style.top = `${planeTop}px`;
    trail.style.top = `${planeTop + planeWidth / 2}px`;

    const total = pieces.filter((piece) => !piece.keep).length;
    if (numberEl && example.ref) numberEl.textContent = example.ref;
    if (timeEl && example.ms) timeEl.textContent = `${example.ms} ms`;
    if (editsEl) editsEl.textContent = `0 / ${total}`;
    pagerButtons.forEach((button, n) => button?.setAttribute('aria-current', String(n === index)));
    figure.setAttribute('aria-description', `Typed: ${example.input} Meant: ${example.output}`);
  };

  // Fix every segment the aircraft's tail has cleared, in order; un-fix when scrubbed back.
  const applyAt = (position) => {
    const reach = position - planeWidth * 0.35;
    let cursor = lineLeft;
    let blocked = false;
    let fixedCount = 0;
    pieces.forEach((piece) => {
      if (piece.keep) {
        cursor += piece.fromWidth;
        if (cursor - piece.fromWidth / 2 > reach) blocked = true;
        return;
      }
      const shouldFix = !blocked && cursor + piece.fromWidth / 2 <= reach;
      if (!shouldFix) blocked = true;
      if (shouldFix !== piece.fixed) {
        piece.fixed = shouldFix;
        piece.el.classList.toggle('is-fixed', shouldFix);
        piece.el.style.width = `${shouldFix ? piece.toWidth : piece.fromWidth}px`;
      }
      if (piece.fixed) fixedCount += 1;
      cursor += piece.fixed ? piece.toWidth : piece.fromWidth;
    });
    if (editsEl) editsEl.textContent = `${fixedCount} / ${pieces.filter((piece) => !piece.keep).length}`;
  };

  const render = () => {
    plane.style.transform = `translate3d(${x - planeWidth / 2}px, 0, 0) rotate(${90 + tilt}deg)`;
    const trailWidth = Math.max(0, x - planeWidth * 0.45);
    trail.style.width = `${trailWidth}px`;
  };

  const land = () => {
    mode = 'landed';
    landedAt = performance.now();
    setStatus('Landed');
    figure.classList.add('is-landed');
    if (liveEl) liveEl.textContent = `Corrected: ${examples[index].output}`;
  };

  function swapTo(next) {
    if (mode === 'swap') return;
    mode = 'swap';
    line.classList.add('is-leaving');
    figure.classList.add('is-landed');
    setTimeout(() => {
      index = (next + examples.length) % examples.length;
      build();
      x = -margin;
      velocity = 0;
      tilt = 0;
      figure.classList.remove('is-landed');
      line.classList.remove('is-leaving');
      line.classList.add('is-entering');
      render();
      requestAnimationFrame(() => line.classList.remove('is-entering'));
      setStatus('Boarding');
      if (reducedMotion.matches) {
        x = stageWidth + margin;
        applyAt(x);
        render();
        land();
      } else {
        setTimeout(() => {
          if (mode === 'swap') mode = 'cruise';
        }, 500);
      }
    }, reducedMotion.matches ? 0 : 650);
  }

  const frame = (now) => {
    if (!running) return;
    const dt = Math.min(0.05, (now - lastTime) / 1000 || 0);
    lastTime = now;
    const previous = x;
    const cruise = (stageWidth + margin * 2) / flightSeconds;

    if (mode === 'drag' && pointerX !== null) {
      x += (pointerX - x) * (1 - Math.exp(-dt * 16));
    } else if (mode === 'cruise') {
      velocity += (cruise - velocity) * (1 - Math.exp(-dt * 1.6));
      x += velocity * dt;
      setStatus(x < lineLeft ? 'Taxi' : 'En route');
      if (x > stageWidth + margin) land();
    } else if (mode === 'landed' && loop && now - landedAt > 2600) {
      swapTo(index + 1);
    }

    const instant = dt > 0 ? (x - previous) / dt : 0;
    const targetTilt = clamp(instant / Math.max(1, cruise) * 4, -9, 9) * (mode === 'drag' ? 1.6 : 0.5);
    tilt += (targetTilt - tilt) * (1 - Math.exp(-dt * 8));

    if (mode !== 'swap') applyAt(x);
    render();
    requestAnimationFrame(frame);
  };

  const start = () => {
    if (running || reducedMotion.matches) return;
    running = true;
    lastTime = performance.now();
    requestAnimationFrame(frame);
  };
  const stop = () => {
    running = false;
  };

  // Dragging: grab anywhere on the stage, the aircraft eases to the pointer.
  const localX = (event) => event.clientX - stage.getBoundingClientRect().left;
  stage.addEventListener('pointerdown', (event) => {
    if (event.button !== 0 || mode === 'swap') return;
    event.preventDefault();
    stage.setPointerCapture(event.pointerId);
    mode = 'drag';
    pointerX = localX(event);
    figure.classList.add('is-dragging');
    figure.classList.remove('is-landed');
    setStatus('Manual');
    if (reducedMotion.matches) {
      running = true;
      lastTime = performance.now();
      requestAnimationFrame(frame);
    }
  });
  stage.addEventListener('pointermove', (event) => {
    if (mode === 'drag') pointerX = clamp(localX(event), -margin, stageWidth + margin);
  });
  const release = () => {
    if (mode !== 'drag') return;
    figure.classList.remove('is-dragging');
    pointerX = null;
    velocity = 0;
    if (reducedMotion.matches) {
      running = false;
      mode = 'landed';
      setStatus(x > stageWidth ? 'Landed' : 'Holding');
      return;
    }
    mode = 'cruise';
  };
  stage.addEventListener('pointerup', release);
  stage.addEventListener('pointercancel', release);

  plane.addEventListener('keydown', (event) => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    event.preventDefault();
    mode = 'landed';
    landedAt = Infinity;
    x = clamp(x + (event.key === 'ArrowRight' ? 1 : -1) * stageWidth * 0.06, -margin, stageWidth + margin);
    setStatus('Manual');
    applyAt(x);
    render();
  });

  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (Math.abs(stage.clientWidth - stageWidth) < 2) return;
      const progress = (x + margin) / (stageWidth + margin * 2);
      build();
      x = progress * (stageWidth + margin * 2) - margin;
      applyAt(x);
      render();
    }, 150);
  });

  build();
  x = -margin;
  render();
  if (reducedMotion.matches) {
    x = stageWidth + margin;
    applyAt(x);
    render();
    land();
  }

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([entry]) => (entry.isIntersecting ? start() : stop()), { threshold: 0.25 }).observe(figure);
  } else {
    start();
  }
};

const heroFlight = document.querySelector('.hero [data-flight]');
if (heroFlight) {
  (document.fonts?.ready ?? Promise.resolve()).then(() => createFlight(heroFlight, [
    { ref: 'EXT-004', ms: 201, input: 'Lest keeP looknig.', output: "Let's keep looking." },
    { ref: 'EXT-006', ms: 168, input: 'Itt sounnds sounnds dangerous.', output: 'It sounds dangerous.' },
    { ref: 'EXT-001', ms: 282, input: "I dno't inda biiT.", output: "I don't mind a bit." },
    { ref: 'EXT-002', ms: 234, input: 'luck turNed inmytaVor.', output: 'Luck turned in my favor.' },
    { ref: 'EXT-005', ms: 357, input: "I'll c to hte arrenemenst for tjhe pa rty.", output: "I'll see to the arrangements for the party." }
  ]));
}

/* ==========================================================================
   Arrivals board — split-flap display of real corrections
   ========================================================================== */

const board = document.querySelector('[data-board]');
if (board) {
  const pages = [
    [
      ['EXT-004', 201, 'Lest keeP looknig.', "Let's keep looking."],
      ['EXT-016', 248, 'co me iw th me, wlil yoU?', 'Come with me, will you?'],
      ['HVY-071', 305, 'I th ougjt Isw a guost.', 'I thought I saw a ghost.'],
      ['EXT-017', 199, 'he 3ill soon Bevac k.', 'He will soon be back.'],
      ['HVY-076', 244, "Thzt aplpe  ins't red.", "That apple isn't red."],
      ['EXT-085', 284, "Rhat's awEsome, rgi ht?", "That's awesome, right?"],
      ['HVY-028', 213, "Lt'sss jiust say yes.", "Let's just say yes."],
      ['EXT-041', 171, 'She li kes oragnes,,', 'She likes oranges.']
    ],
    [
      ['EXT-073', 363, 'das Drawa line frome A 5o B', 'Draw a line from A to B.'],
      ['HVY-067', 247, 'hey,yiu know waht?', 'Hey, you know what?'],
      ['EXT-094', 246, "We sti l n't Know.", "We still don't know."],
      ['HVY-078', 277, 'WWh at he syas i true.', 'What he says is true.'],
      ['HVY-058', 200, "Te don'bt really acre.", "They don't really care."],
      ['EXT-039', 287, 'she isbad at sop sop rts', 'She is bad at sports.'],
      ['HVY-057', 224, "it'ssad, ture.", "It's sad, but true."],
      ['EXT-086', 206, 'He yielded otpre ss ure', 'He yielded to pressure.']
    ]
  ];
  const CELLS = 24;
  const STATUS_CELLS = 6;
  const GLYPHS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const rowsEl = board.querySelector('[data-board-rows]');
  const pageEl = board.querySelector('[data-board-page]');
  const clockEl = board.querySelector('[data-board-clock]');

  const makeFlaps = (count, className) => {
    const wrap = document.createElement('span');
    wrap.className = className;
    wrap.setAttribute('aria-hidden', 'true');
    const cells = Array.from({ length: count }, () => {
      const cell = document.createElement('span');
      cell.className = 'flap';
      cell.textContent = ' ';
      wrap.append(cell);
      return cell;
    });
    return { wrap, cells };
  };

  const rows = pages[0].map(() => {
    const li = document.createElement('li');
    li.className = 'board-row';
    const ref = document.createElement('span');
    ref.className = 'b-ref';
    const from = document.createElement('span');
    from.className = 'b-from';
    const to = makeFlaps(CELLS, 'b-to');
    const time = document.createElement('span');
    time.className = 'b-time';
    const status = makeFlaps(STATUS_CELLS, 'b-status');
    const label = document.createElement('span');
    label.className = 'sr-only';
    li.append(label, ref, from, to.wrap, time, status.wrap);
    rowsEl.append(li);
    return { li, ref, from, to: to.cells, time, status: status.cells, statusWrap: status.wrap, label };
  });

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  // One mechanical flap: a short run of random glyphs, then the target.
  const flipCell = async (cell, target, delay) => {
    await sleep(delay);
    if (cell.textContent === target) return;
    const spins = reducedMotion.matches || target === ' ' ? 0 : 2 + Math.floor(Math.random() * 4);
    for (let n = 0; n <= spins; n += 1) {
      cell.textContent = n === spins ? target : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
      if (!reducedMotion.matches) {
        cell.classList.remove('is-flipping');
        void cell.offsetWidth;
        cell.classList.add('is-flipping');
        await sleep(64);
      }
    }
  };
  const flipText = (cells, text, rowDelay) => {
    const padded = [...text.padEnd(cells.length, ' ')].slice(0, cells.length);
    return Promise.all(cells.map((cell, n) => flipCell(cell, padded[n], rowDelay + n * 18)));
  };

  const showPage = async (pageIndex) => {
    const page = pages[pageIndex];
    if (pageEl) pageEl.textContent = `Page ${pageIndex + 1} / ${pages.length}`;
    // Departures: every row shows what was typed …
    await Promise.all(rows.map((row, n) => {
      const [ref, ms, input, output] = page[n];
      row.li.classList.remove('is-landed');
      row.ref.textContent = ref;
      row.from.textContent = input;
      row.time.textContent = `${ms} ms`;
      row.label.textContent = `${ref}: typed “${input}”, corrected to “${output}” in ${ms} milliseconds.`;
      return Promise.all([flipText(row.to, input, n * 70), flipText(row.status, 'EN RTE', n * 70)]);
    }));
    await sleep(reducedMotion.matches ? 0 : 900);
    // … then lands, row by row, as the corrected sentence.
    await Promise.all(rows.map(async (row, n) => {
      const [, , , output] = page[n];
      await Promise.all([flipText(row.to, output, n * 160), flipText(row.status, 'LANDED', n * 160 + 200)]);
      row.li.classList.add('is-landed');
    }));
  };

  const tickClock = () => {
    if (!clockEl) return;
    const now = new Date();
    clockEl.textContent = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
  };
  tickClock();
  setInterval(tickClock, 15000);

  onceVisible([board], async () => {
    let pageIndex = 0;
    for (;;) {
      await showPage(pageIndex);
      await sleep(6500);
      pageIndex = (pageIndex + 1) % pages.length;
    }
  }, { threshold: 0.3 });
}

/* ---------- Count-up figures (expo-out, no overshoot) ---------- */

const counters = [...document.querySelectorAll('[data-count]')];
const formatCount = (value, decimals) => value.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
if (!reducedMotion.matches) {
  counters.forEach((counter) => { counter.textContent = formatCount(0, Number(counter.dataset.decimals || 0)); });
}
onceVisible(counters, (counter) => {
  const target = Number(counter.dataset.count);
  const decimals = Number(counter.dataset.decimals || 0);
  if (reducedMotion.matches) return;
  const start = performance.now();
  const duration = 2000;
  const tick = (now) => {
    const t = clamp((now - start) / duration);
    counter.textContent = formatCount(target * (1 - Math.pow(2, -10 * t)) / (1 - Math.pow(2, -10)), decimals);
    if (t < 1) requestAnimationFrame(tick);
    else counter.textContent = formatCount(target, decimals);
  };
  requestAnimationFrame(tick);
}, { threshold: 0.5 });

/* ---------- Method: scroll-driven read → context → edit ---------- */

const method = document.querySelector('[data-method]');
if (method) {
  const figure = method.querySelector('[data-method-figure]');
  const sentence = method.querySelector('[data-method-sentence]');
  const arcs = method.querySelector('[data-method-arcs]');
  const steps = [...method.querySelectorAll('[data-step]')];
  const badTokens = [...sentence.querySelectorAll('.tok.is-bad')];
  let currentStep = -1;

  const measureToken = (token) => {
    const from = token.querySelector('.from');
    const to = token.querySelector('.to');
    token.dataset.fromWidth = from.getBoundingClientRect().width;
    token.dataset.toWidth = to.getBoundingClientRect().width;
  };

  const drawArcs = () => {
    const box = figure.getBoundingClientRect();
    arcs.setAttribute('viewBox', `0 0 ${box.width} ${box.height}`);
    const point = (token) => {
      const rect = token.getBoundingClientRect();
      return [rect.left - box.left + rect.width / 2, rect.top - box.top + rect.height * 0.12];
    };
    const source = sentence.querySelector('[data-tok="c"]');
    const targets = ['to', 'arrenemenst', 'pa rty.'].map((key) => [...sentence.querySelectorAll('.tok')].find((token) => (token.dataset.tok || token.textContent) === key));
    const [sx, sy] = point(source);
    arcs.innerHTML = targets.filter(Boolean).map((target, n) => {
      const [tx, ty] = point(target);
      const lift = Math.min(box.height * 0.32, 40 + Math.abs(tx - sx) * 0.28);
      return `<path pathLength="1" style="--n:${n}" d="M${sx} ${sy} C ${sx} ${sy - lift}, ${tx} ${ty - lift}, ${tx} ${ty}"/>`;
    }).join('') + `<circle cx="${sx}" cy="${sy}" r="4"/>`;
  };

  const layout = () => {
    badTokens.forEach((token) => {
      token.style.width = '';
      token.classList.add('is-measuring');
      measureToken(token);
      token.classList.remove('is-measuring');
      token.style.width = `${currentStep >= 2 ? token.dataset.toWidth : token.dataset.fromWidth}px`;
    });
    if (currentStep >= 1 && currentStep < 2) drawArcs();
  };

  const setStep = (step) => {
    if (step === currentStep) return;
    currentStep = step;
    figure.dataset.step = String(step);
    steps.forEach((item, n) => item.classList.toggle('is-active', n === step));
    badTokens.forEach((token) => {
      token.style.width = `${step >= 2 ? token.dataset.toWidth : token.dataset.fromWidth}px`;
    });
    if (step === 1) drawArcs();
  };

  const update = () => {
    const rect = method.getBoundingClientRect();
    const travel = rect.height - window.innerHeight;
    const progress = travel > 0 ? clamp(-rect.top / travel) : 1;
    figure.style.setProperty('--progress', progress.toFixed(4));
    setStep(Math.min(2, Math.floor(progress * 3)));
  };

  (document.fonts?.ready ?? Promise.resolve()).then(() => {
    layout();
    update();
    scrollHandlers.add(update);
    window.addEventListener('resize', () => requestAnimationFrame(layout));
  });
}
