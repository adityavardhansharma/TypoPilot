const navToggle = document.querySelector('.nav-toggle');
const siteNav = document.querySelector('.site-nav');

if (navToggle && siteNav) {
  const closeNav = () => {
    navToggle.setAttribute('aria-expanded', 'false');
    siteNav.classList.remove('is-open');
  };

  navToggle.addEventListener('click', () => {
    const isOpen = navToggle.getAttribute('aria-expanded') === 'true';
    navToggle.setAttribute('aria-expanded', String(!isOpen));
    siteNav.classList.toggle('is-open', !isOpen);
  });
  siteNav.addEventListener('click', (event) => {
    if (event.target.closest('a')) closeNav();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeNav();
  });
}

const flightBoard = document.querySelector('[data-flight-board]');
if (flightBoard) {
  const examples = [
    { input: 'Lest keeP looknig.', output: "Let's keep looking." },
    { input: "I dno't inda biiT.", output: "I don't mind a bit." },
    { input: 'Itt sounnds sounnds dangerous.', output: 'It sounds dangerous.' },
    { input: 'luck turNed inmytaVor.', output: 'Luck turned in my favor.' }
  ];
  const wrong = flightBoard.querySelector('[data-wrong]');
  const correct = flightBoard.querySelector('[data-correct]');
  const count = flightBoard.querySelector('[data-example-count]');
  const plane = flightBoard.querySelector('.plane');
  const track = flightBoard.querySelector('.plane-track');
  let exampleIndex = 0;

  const sizeFlightPath = () => {
    if (!track) return;
    flightBoard.style.setProperty('--plane-travel', `${Math.max(250, track.clientWidth - 10)}px`);
  };
  const showExample = () => {
    const example = examples[exampleIndex];
    wrong.textContent = example.input;
    correct.textContent = example.output;
    count.textContent = `Example ${exampleIndex + 1} / ${examples.length}`;
    flightBoard.setAttribute('aria-label', `Animated correction example. Input: ${example.input} Output: ${example.output}`);
  };

  plane?.addEventListener('animationiteration', () => {
    exampleIndex = (exampleIndex + 1) % examples.length;
    showExample();
  });
  window.addEventListener('resize', sizeFlightPath, { passive: true });
  sizeFlightPath();
  showExample();
}

const dotMatrix = document.querySelector('[data-dot-matrix]');
if (dotMatrix) {
  const outcomes = [
    ['exact', 629],
    ['improved', 157],
    ['over', 72],
    ['neutral', 42]
  ];
  const fragment = document.createDocumentFragment();
  outcomes.forEach(([className, total]) => {
    for (let index = 0; index < total; index += 1) {
      const dot = document.createElement('i');
      dot.className = className;
      fragment.appendChild(dot);
    }
  });
  dotMatrix.appendChild(fragment);
}

const researchLinks = [...document.querySelectorAll('.research-toc a[href^="#"]')];
if (researchLinks.length && 'IntersectionObserver' in window) {
  const byId = new Map(researchLinks.map((link) => [link.hash.slice(1), link]));
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    researchLinks.forEach((link) => link.removeAttribute('aria-current'));
    byId.get(visible.target.id)?.setAttribute('aria-current', 'location');
  }, { rootMargin: '-18% 0px -68%', threshold: [0, 0.25, 0.6] });
  byId.forEach((_, id) => {
    const section = document.getElementById(id);
    if (section) observer.observe(section);
  });
}
