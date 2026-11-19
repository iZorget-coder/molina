import { loadNews, bindFilters, loadDashboardNews } from './news.js';
import { bindWatchlistPage, bindDashboardWatchlist, loadDashboardCards } from './watchlist.js';
import { loadForecastHeader } from './dashboard-forecasts.js';
import { bindSettings } from './settings.js';
import { bindNavAvatar } from './profile.js';
import { initMarketPage, selectMarket } from './markets.js';
(function () {
  const container = document.getElementById('view-container');
  const cache = {};
  let currentName = null;
  let requestToken = 0;

  const VALID = ['dashboard', 'watchlist', 'markets', 'news', 'settings', 'ai', 'article'];
  const NAV_TAB = { ai: 'markets', article: 'news' };

  const hooks = {
    dashboard: initDashboard,
    markets: () => { renderCandles(); initMarketPage(); },
    watchlist: bindWatchlistPage,
    news: initNews,
    article: renderArticle,
    settings: bindSettings,
  };

function initDashboard() {
  loadForecastHeader();
  loadDashboardNews();
  bindDashboardWatchlist();
  loadDashboardCards();
}

  function initNews() {
    loadNews();
    bindFilters();
  }

  // Open the Markets page for one market (from a dashboard card, a sidebar item or a watchlist row)
  function openMarket(symbol) {
    selectMarket(symbol);
    history.pushState({ view: 'markets', symbol }, '', '#markets');
    currentName = null; // reload the view even if it's already open
    loadView('markets');
  }

  function renderArticle() {
    const data = history.state?.article
      || (() => {
        try { return JSON.parse(sessionStorage.getItem('currentArticle') || 'null'); }
        catch { return null; }
      })();

    const tagEl = document.getElementById('article-tag');
    const titleEl = document.getElementById('article-title');
    const metaEl = document.getElementById('article-meta');
    const bodyEl = document.getElementById('article-body');
    const sourceEl = document.getElementById('article-source');
    const hero = document.querySelector('.article__hero');
    const moreList = document.getElementById('article-more');
    const moreHeading = document.getElementById('art-more');

    if (!data || !titleEl) {
      if (titleEl) titleEl.textContent = 'Article not found';
      if (bodyEl) bodyEl.innerHTML =
        '<p>This story could not be loaded. Return to the <a href="#news">news feed</a>.</p>';
      if (moreList) moreList.innerHTML = '';
      return;
    }

    const categoryLabel = data.category.charAt(0).toUpperCase() + data.category.slice(1);
    if (tagEl) tagEl.textContent = categoryLabel;
    if (titleEl) titleEl.textContent = data.title;
    if (metaEl) metaEl.innerHTML =
      `<strong>${data.source || 'Molina Wire'}</strong> <i aria-hidden="true"></i> <time>${data.published || ''}</time>`;
    if (bodyEl) bodyEl.innerHTML =
      `<p>${data.description || 'No preview available for this story.'}</p>`;

    if (data.image && hero) {
      const img = new Image();
      img.alt = '';
      img.onload = () => { hero.innerHTML = ''; hero.appendChild(img); };
      img.src = data.image;
    }

    if (data.url && sourceEl) {
      const link = sourceEl.querySelector('a');
      if (link) {
        link.href = data.url;
        sourceEl.hidden = false;
      }
    }

    if (moreHeading) moreHeading.textContent = `More from ${categoryLabel}`;

    if (moreList) {
      const siblings = Array.isArray(data.siblings) ? data.siblings : [];
      const others = siblings.filter(s => s.title && s.title !== data.title).slice(0, 3);

      if (!others.length) {
        moreList.innerHTML = '<li><p class="view-empty">No related stories.</p></li>';
      } else {
        moreList.innerHTML = others.map(sib => {
          const sibPayload = encodeURIComponent(JSON.stringify({
            title: sib.title,
            source: 'Currents',
            published: sib.published || '',
            description: sib.description || '',
            url: sib.url || '',
            image: sib.image || '',
            category: data.category,
            siblings,
          }));
          return `
            <li>
              <a href="#article" class="article-more" data-article="${sibPayload}">
                <strong>${sib.title}</strong>
                <span>Currents · ${sib.published || ''}</span>
              </a>
            </li>
          `;
        }).join('');
      }
    }
  }

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

  async function loadView(name) {
    if (!VALID.includes(name)) name = 'dashboard';
   if (name === currentName) {
      if (name === 'markets') {
        initMarketPage();
      }
      return;
    }

    const token = ++requestToken;

    const navTab = NAV_TAB[name] || name;
    document.querySelectorAll('.nav__link').forEach(a => {
      a.classList.toggle('is-active', a.dataset.tab === navTab);
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
   
    const marketEl = e.target.closest('[data-market]');
    if (marketEl && !e.target.closest('button, a')) {
      openMarket(marketEl.dataset.market);
      return;
    }

    const story = e.target.closest('a[data-article]');
    if (story) {
      e.preventDefault();
      let payload = null;
      try {
        payload = JSON.parse(decodeURIComponent(story.dataset.article));
      } catch (err) {
        console.error('Bad article payload', err);
      }
      if (payload) {
        sessionStorage.setItem('currentArticle', JSON.stringify(payload));
        history.pushState({ view: 'article', article: payload }, '', '#article');
        currentName = null;
        loadView('article');
      }
      return;
    }

    const link = e.target.closest('a[href^="#"]');
    if (!link) return;

    const name = link.getAttribute('href').slice(1);
    if (!VALID.includes(name)) return;

    e.preventDefault();
    history.pushState({ view: name }, '', `#${name}`);
    loadView(name);
  });
 
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    const marketEl = e.target.closest?.('[data-market]');
    if (marketEl && e.target === marketEl) openMarket(marketEl.dataset.market);
  });

  window.addEventListener('popstate', () => {
    const name = location.hash.slice(1) || 'dashboard';
    loadView(name);
  });

  bindNavAvatar(); 

  const initial = location.hash.slice(1);
  if (initial && VALID.includes(initial)) {
    loadView(initial);
  } else {
    history.replaceState({ view: 'dashboard' }, '', '#dashboard');
    loadView('dashboard');
  }
})();