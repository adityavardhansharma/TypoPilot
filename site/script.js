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
