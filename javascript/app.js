(function () {
  const container = document.getElementById('view-container');
  const cache = {};
  let currentName = null;
  let requestToken = 0;

  const VALID = ['dashboard', 'watchlist', 'markets', 'news', 'settings', 'ai', 'article'];
const NAV_TAB = { ai: 'markets', article: 'news' };
const navTab = NAV_TAB[name] || name;
document.querySelectorAll('.nav__link').forEach(a => {
  a.classList.toggle('is-active', a.dataset.tab === navTab);
});



  // ---- Per-view hooks ----
  const hooks = {
    markets: renderCandles,
    watchlist: bindWatchlistFilters,
  };

  function renderCandles() {
    const svg = document.getElementById('candles');
    if (!svg || svg.dataset.rendered) return;
    svg.dataset.rendered = '1';

    const NS = 'http://www.w3.org/2000/svg';
    const TOP = 2520;
    const N = 100;
    const SPAN = 880;
    const STEP = SPAN / N;
    const BODY = STEP * 0.55;

    let seed = 7;
    const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

    const pts = [[0,2424],[25,2438],[40,2442],[50,2435],[65,2438],[75,2448],[90,2446],[99,2450]];
    const base = i => {
      for (let k = 0; k < pts.length - 1; k++) {
        const [x0, y0] = pts[k], [x1, y1] = pts[k + 1];
        if (i >= x0 && i <= x1) return y0 + (y1 - y0) * (i - x0) / (x1 - x0);
      }
      return pts[pts.length - 1][1];
    };

    let prev = base(0);
    for (let i = 0; i < N; i++) {
      const open = prev;
      const close = base(i) + (rand() - 0.5) * 3;
      const high = Math.max(open, close) + rand() * 2;
      const low  = Math.min(open, close) - rand() * 2;
      prev = close;

      const up = close >= open;
      const color = up ? '#22c55e' : '#ef4444';
      const x = i * STEP + STEP / 2;

      const wick = document.createElementNS(NS, 'line');
      wick.setAttribute('x1', x); wick.setAttribute('x2', x);
      wick.setAttribute('y1', TOP - high); wick.setAttribute('y2', TOP - low);
      wick.setAttribute('stroke', color);
      wick.setAttribute('stroke-width', 1);
      wick.setAttribute('vector-effect', 'non-scaling-stroke');
      svg.appendChild(wick);

      const rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', x - BODY / 2);
      rect.setAttribute('width', BODY);
      rect.setAttribute('y', TOP - Math.max(open, close));
      rect.setAttribute('height', Math.max(Math.abs(close - open), 0.5));
      rect.setAttribute('fill', color);
      svg.appendChild(rect);
    }
  }

  function bindWatchlistFilters() {
    const chips = document.querySelectorAll('.chip[data-filter]');
    const rows  = document.querySelectorAll('.table tbody tr');

    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        chips.forEach(c => c.classList.remove('is-active'));
        chip.classList.add('is-active');
        const f = chip.dataset.filter;
        rows.forEach(r => {
          r.hidden = f !== 'all' && r.dataset.category !== f;
        });
      });
    });
  }

  async function loadView(name) {
    if (!VALID.includes(name)) name = 'dashboard';
    if (name === currentName) return;

    const token = ++requestToken;

    document.querySelectorAll('.nav__link').forEach(a => {
      a.classList.toggle('is-active', a.dataset.tab === name);
    });

    if (cache[name]) {
      container.innerHTML = cache[name];
      currentName = name;
      hooks[name] && hooks[name]();
      return;
    }

    container.innerHTML = '<p class="view-loading">Loading…</p>';

    try {
      const res = await fetch(`/views/${name}.html`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const html = await res.text();

      if (token !== requestToken) return;

      cache[name] = html;
      container.innerHTML = html;
      currentName = name;
      hooks[name] && hooks[name]();
    } catch (err) {
      if (token !== requestToken) return;
      container.innerHTML = `<p class="view-error">Could not load ${name} view.</p>`;
      console.error(err);
    }
  }

  document.addEventListener('click', (e) => {
    const link = e.target.closest('a[href^="#"]');
    if (!link) return;

    const name = link.getAttribute('href').slice(1);
    if (!VALID.includes(name)) return;

    e.preventDefault();
    history.pushState({ view: name }, '', `#${name}`);
    loadView(name);
  });

  window.addEventListener('popstate', () => {
    loadView(location.hash.slice(1) || 'dashboard');
  });

  loadView(location.hash.slice(1) || 'dashboard');


  //page refresh to dashboard on first load
  history.replaceState({ view: 'dashboard' }, '', '#dashboard');
  loadView('dashboard');
})();