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
