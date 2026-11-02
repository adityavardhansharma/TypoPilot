const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

/* ---------- Header: menu, scrolled state, progress ---------- */

const header = document.querySelector('[data-header]');
const navToggle = document.querySelector('.nav-toggle');
const progressBar = document.querySelector('[data-scroll-progress]');

if (header && navToggle) {
  const setNav = (open) => {
    navToggle.setAttribute('aria-expanded', String(open));
    navToggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    header.classList.toggle('is-open', open);
  };
  navToggle.addEventListener('click', () => setNav(navToggle.getAttribute('aria-expanded') !== 'true'));
  header.querySelector('.site-nav')?.addEventListener('click', (event) => {
    if (event.target.closest('a')) setNav(false);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') setNav(false);
  });
}

let scrollQueued = false;
const onScroll = () => {
  scrollQueued = false;
  const max = document.documentElement.scrollHeight - window.innerHeight;
  header?.classList.toggle('is-scrolled', window.scrollY > 24);
  progressBar?.style.setProperty('--progress', max > 0 ? (window.scrollY / max).toFixed(4) : '0');
};
window.addEventListener('scroll', () => {
  if (!scrollQueued) {
    scrollQueued = true;
    requestAnimationFrame(onScroll);
  }
}, { passive: true });
onScroll();

/* ---------- Scroll reveal ---------- */

const whenVisible = (elements, callback, options = {}) => {
  if (!('IntersectionObserver' in window)) {
    elements.forEach(callback);
    return;
  }
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      callback(entry.target);
      observer.unobserve(entry.target);
    });
  }, { rootMargin: '0px 0px -12% 0px', threshold: 0.12, ...options });
  elements.forEach((element) => observer.observe(element));
};

document.querySelectorAll('[data-stagger]').forEach((group) => {
  [...group.children].forEach((child, index) => child.style.setProperty('--i', index));
});
document.documentElement.classList.add('js');
whenVisible([...document.querySelectorAll('[data-reveal], [data-stagger]')], (element) => {
  element.classList.add('is-visible');
});

/* ---------- Hero: live correction editor ---------- */

// Marks which words in each sentence survive unchanged (longest common subsequence).
const diffWords = (a, b) => {
  const table = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const keptInput = new Set();
  const keptOutput = new Set();
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      keptInput.add(i); keptOutput.add(j); i += 1; j += 1;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      i += 1;
    } else {
      j += 1;
    }
  }
  return { keptInput, keptOutput };
};

const editor = document.querySelector('[data-editor]');
if (editor) {
  const examples = [
    { input: 'Lest keeP looknig.', output: "Let's keep looking." },
    { input: "I dno't inda biiT.", output: "I don't mind a bit." },
    { input: 'Itt sounnds sounnds dangerous.', output: 'It sounds dangerous.' },
    { input: "I'll c to hte arrenemenst for tjhe pa rty.", output: "I'll see to the arrangements for the party." },
    { input: 'luck turNed inmytaVor.', output: 'Luck turned in my favor.' }
  ];
  const inputEl = editor.querySelector('[data-editor-input]');
  const outputEl = editor.querySelector('[data-editor-output]');
  const statusEl = editor.querySelector('[data-editor-status]');
  const editsEl = editor.querySelector('[data-editor-edits]');
  const liveEl = editor.querySelector('[data-editor-live]');
  const pager = editor.querySelector('[data-editor-pager]');

  let run = 0;
  let current = 0;
  const sleep = (ms, token) => new Promise((resolve, reject) => {
    setTimeout(() => (token === run ? resolve() : reject(new Error('cancelled'))), ms);
  });
  const setStatus = (label, state) => {
    statusEl.textContent = label;
    editor.dataset.state = state;
  };
  const wordSpans = (words, kept, className) => words.map((word, index) => {
    const span = document.createElement('span');
    span.className = kept.has(index) ? 'word' : `word ${className}`;
    span.style.setProperty('--w', index);
    span.textContent = word;
    return span;
  });
  const fill = (element, spans) => {
    element.replaceChildren();
    spans.forEach((span, index) => {
      if (index) element.append(' ');
      element.append(span);
    });
  };

  const dots = examples.map((_, index) => {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.setAttribute('aria-label', `Example ${index + 1}`);
    dot.addEventListener('click', () => play(index));
    pager.append(dot);
    return dot;
  });

  const renderFinal = (example) => {
    const inputWords = example.input.split(' ');
    const outputWords = example.output.split(' ');
    const { keptInput, keptOutput } = diffWords(inputWords, outputWords);
    fill(inputEl, wordSpans(inputWords, keptInput, 'is-wrong'));
    fill(outputEl, wordSpans(outputWords, keptOutput, 'is-fixed'));
    editsEl.textContent = String(outputWords.length - keptOutput.size);
    liveEl.textContent = `Input: ${example.input} Corrected: ${example.output}`;
  };

  async function play(index) {
    run += 1;
    const token = run;
    current = index;
    const example = examples[index];
    dots.forEach((dot, dotIndex) => dot.setAttribute('aria-current', String(dotIndex === index)));

    if (reducedMotion.matches) {
      renderFinal(example);
      setStatus('Corrected', 'done');
      return;
    }

    try {
      editor.classList.remove('is-scanning');
      outputEl.replaceChildren();
      inputEl.replaceChildren();
      editsEl.textContent = '0';
      setStatus('Typing…', 'typing');
      const caret = document.createElement('span');
      caret.className = 'caret';
      const typed = document.createTextNode('');
      inputEl.append(typed, caret);
      for (const char of example.input) {
        typed.data += char;
        await sleep(38 + Math.random() * 55, token);
      }
      await sleep(380, token);

      setStatus('Reading context', 'reading');
      renderFinal(example);
      outputEl.replaceChildren();
      editor.classList.add('is-scanning');
      await sleep(950, token);

      setStatus('Corrected', 'done');
      editor.classList.remove('is-scanning');
      renderFinal(example);
      await sleep(3600, token);
      play((current + 1) % examples.length);
    } catch {
      // A newer run took over.
    }
  }

  whenVisible([editor], () => play(0), { threshold: 0.3 });
}

/* ---------- Count-up numbers ---------- */

const counters = [...document.querySelectorAll('[data-count]')];
const formatCount = (value, decimals) => value.toLocaleString('en-US', {
  minimumFractionDigits: decimals,
  maximumFractionDigits: decimals
});
counters.forEach((counter) => {
  counter.textContent = formatCount(0, Number(counter.dataset.decimals || 0));
});
whenVisible(counters, (counter) => {
  const target = Number(counter.dataset.count);
  const decimals = Number(counter.dataset.decimals || 0);
  if (reducedMotion.matches) {
    counter.textContent = formatCount(target, decimals);
    return;
  }
  const duration = 1600;
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 4);
    counter.textContent = formatCount(target * eased, decimals);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}, { threshold: 0.6 });

/* ---------- Pointer spotlight on cards ---------- */

document.querySelectorAll('[data-spotlight]').forEach((card) => {
  card.addEventListener('pointermove', (event) => {
    const rect = card.getBoundingClientRect();
    card.style.setProperty('--mx', `${event.clientX - rect.left}px`);
    card.style.setProperty('--my', `${event.clientY - rect.top}px`);
  });
});
