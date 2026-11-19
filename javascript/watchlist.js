import { doc, getDoc, setDoc } from 'firebase/firestore';
import { db, auth, onAuthStateChanged } from './firebase.js';
import { MARKETS, fetchLatestPredictions } from './predictions.js';

const DEFAULT_WATCHLIST = ['SPX'];

let cachedSymbols = null;
let authReady = null;

function whenAuthReady() {
  if (authReady) return authReady;
  authReady = new Promise(resolve => {
    const unsub = onAuthStateChanged(auth, user => { unsub(); resolve(user); });
  });
  return authReady;
}

function userRef(uid) {
  return doc(db, 'users', uid);
}

async function fetchWatchlist() {
  if (cachedSymbols) return cachedSymbols;

  const user = await whenAuthReady();
  if (!user) {
    cachedSymbols = [...DEFAULT_WATCHLIST];
    return cachedSymbols;
  }

  const ref = userRef(user.uid);
  const snap = await getDoc(ref);

  if (snap.exists() && Array.isArray(snap.data().watchlist)) {
    cachedSymbols = snap.data().watchlist;
  } else {
    cachedSymbols = [...DEFAULT_WATCHLIST];
    await setDoc(ref, {
      watchlist: cachedSymbols,
      updatedAt: new Date().toISOString(),
    }, { merge: true });
  }
  return cachedSymbols;
}

async function saveWatchlist(symbols) {
  const user = await whenAuthReady();
  cachedSymbols = symbols;
  if (!user) return;
  await setDoc(userRef(user.uid), {
    watchlist: symbols,
    updatedAt: new Date().toISOString(),
  }, { merge: true });
}

/* ---------- Live forecasts (Firestore: daily_predictions) ---------- */

let live = null;  

 
async function loadLive() {
  live = await fetchLatestPredictions();
  return live;
}

// One market, ready to render: { symbol, name, sub, dir, label, confidence, price, change, changeText }
function getMarket(symbol) {
  const meta = MARKETS.find(m => m.symbol === symbol);
  if (!meta) return null;

  return live?.bySymbol[symbol] || {
    ...meta,
    sub: `${meta.symbol} · ${meta.type}`,
    hasForecast: false,
    dir: 'none',
    label: 'No forecast',
    confidence: null,
    price: '—',
    change: null,
    changeText: '—',
  };
}

// Colour for the day-on-day change; moves under 0.1% count as flat
function changeClass(market) {
  if (market.change == null || Math.abs(market.change) < 0.1) return 'flat';
  return market.change > 0 ? 'up' : 'down';
}

/* ---------- Watchlist page rendering ---------- */

function watchRowMarkup(market) {
  return `
    <tr data-category="${market.dir}">
      <td class="col-market">
        <div class="market__name">${market.name}</div>
        <div class="market__type">${market.sub}</div>
      </td>
      <td class="col-close num">${market.price}</td>
      <td class="col-pred">
        <span class="tag tag--${market.dir}">${market.label}</span>
      </td>
      <td class="col-change num ${changeClass(market)}">${market.changeText}</td>
      <td class="col-actions num">
        <button class="btn-remove" data-remove="${market.symbol}" aria-label="Remove ${market.name} from watchlist">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        </button>
      </td>
    </tr>
  `;
}

export async function loadWatchlistPage() {
  const tbody = document.querySelector('#watchlist-table tbody');
  if (!tbody) return;

  tbody.innerHTML = '<tr><td colspan="5" class="view-loading">Loading watchlist…</td></tr>';

  try {
    const [symbols] = await Promise.all([fetchWatchlist(), loadLive()]);
    const markets = symbols.map(getMarket).filter(Boolean);

    if (!markets.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="view-empty">Your watchlist is empty. Add a market to get started.</td></tr>';
    } else {
      tbody.innerHTML = markets.map(watchRowMarkup).join('');
    }

    const countEl = document.getElementById('watchlist-count');
    if (countEl) countEl.textContent = `${markets.length} ${markets.length === 1 ? 'market' : 'markets'}`;
  } catch (err) {
    console.error(err);
    tbody.innerHTML = '<tr><td colspan="5" class="view-error">Could not load watchlist.</td></tr>';
  }
}

/* ---------- Dashboard sidebar rendering ---------- */

function dashboardItemMarkup(market) {
  return `
    <li class="watch__item" data-market="${market.symbol}">
      <div class="watch__row">
        <div>
          <div class="watch__name">${market.name}</div>
          <div class="watch__sub">${market.sub}</div>
        </div>
        <div class="watch__price">${market.price}</div>
      </div>
      <div class="watch__row">
        <span class="badge badge--${market.dir}">${market.label}</span>
        <span class="change ${changeClass(market)}">${market.changeText}</span>
      </div>
    </li>
  `;
}

export async function loadDashboardWatchlist() {
  const list = document.querySelector('.watch');
  if (!list) return;

  try {
    const [symbols] = await Promise.all([fetchWatchlist(), loadLive()]);
    const markets = symbols.map(getMarket).filter(Boolean);

    list.innerHTML = markets.length
      ? markets.map(dashboardItemMarkup).join('')
      : '<li class="watch__item"><p class="view-empty">No markets yet.</p></li>';

    const countEl = document.querySelector('.sidebar__head .muted');
    if (countEl) countEl.textContent = `${markets.length} ${markets.length === 1 ? 'market' : 'markets'}`;
  } catch (err) {
    console.error(err);
    list.innerHTML = '<li class="watch__item"><p class="view-error">Could not load watchlist.</p></li>';
  }
}

/* ---------- Dashboard forecast cards ---------- */


function sparkPolyline(market) {
  const seedBase = market.symbol.split('').reduce((a, c) => a + c.charCodeAt(0), 7);
  let seed = seedBase;
  const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

  const points = [];
  let y = market.dir === 'up' ? 46 : market.dir === 'down' ? 6 : 40;

  for (let i = 0; i < 17; i++) {
    const x = 2 + (i * 196) / 16;
    if (market.dir === 'up')        y -= rand() * 2.5;
    else if (market.dir === 'down') y += rand() * 2.5;
    else                            y  = 40 + Math.sin(i * 1.7 + seedBase) * 10;
    const clamped = Math.max(2, Math.min(54, y));
    points.push(`${x.toFixed(0)},${clamped.toFixed(0)}`);
  }
  return points.join(' ');
}

function cardMarkup(market) {
  const pillLabel = {
    up: '▲ UP',
    down: '▼ DOWN',
    flat: '◆ CONSOLIDATION'
  }[market.dir] || 'NO FORECAST';

  const confidence = market.confidence ?? 0;
  const confidenceText = market.confidence === null
    ? '—'
    : `${market.confidence}%`;

  const sparkDir = market.dir === 'none' ? 'flat' : market.dir;

  return `
    <article
      class="card"
      data-market="${market.symbol}"
      tabindex="0"
      role="link"
      aria-label="View ${market.name} market details"
    >
      <div class="card__id">
        <div class="card__title">${market.name} <span class="ticker">${market.symbol}</span></div>
        <span class="pill pill--${market.dir}">${pillLabel}</span>
      </div>
      <div class="card__conf">
        <div class="conf__row"><span class="conf__label">Confidence</span><span class="conf__value">${confidenceText}</span></div>
        <div class="progress"><div class="progress__bar" style="width:${confidence}%"></div></div>
        <div class="conf__row conf__row--small">
          <span class="muted-light">Last close <strong>${market.price}</strong></span>
          <span class="${changeClass(market)}">${market.changeText}</span>
        </div>
      </div>
      <div class="card__chart">
        <svg viewBox="0 0 200 56" preserveAspectRatio="none" class="spark spark--${sparkDir}">
          <polyline points="${sparkPolyline(market)}"/>
        </svg>
        <span class="chart__label">Last 24 sessions</span>
      </div>
    </article>
  `;
}

export async function loadDashboardCards() {
  const container = document.getElementById('dashboard-cards');
  if (!container) return;

  try {
    const [symbols] = await Promise.all([fetchWatchlist(), loadLive()]);
    const markets = symbols.map(getMarket).filter(Boolean);

    if (!markets.length) {
      container.innerHTML = '<p class="view-empty">Your watchlist is empty. Add a market to see forecasts.</p>';
      return;
    }

    container.innerHTML = markets.map(cardMarkup).join('');
  } catch (err) {
    console.error(err);
    container.innerHTML = '<p class="view-error">Could not load forecasts.</p>';
  }
}

/* ---------- Add-market modal ---------- */

function renderModalRows(dialog, currentSymbols) {
  const rows = dialog.querySelectorAll('.add-modal__row');
  rows.forEach(row => {
    const symbol = row.dataset.symbol;
    const input = row.querySelector('input');
    input.disabled = currentSymbols.includes(symbol);
    input.checked = false;

 
    const market = getMarket(symbol);
    const priceEl = row.querySelector('.add-modal__price');
    const tagEl = row.querySelector('.add-modal__tag');
    if (market && priceEl) priceEl.textContent = market.price;
    if (market && tagEl) {
      tagEl.textContent = market.label;
      tagEl.className = `add-modal__tag add-modal__tag--${market.dir}`;
    }
  });

  const empty = dialog.querySelector('.add-modal__empty');
  const allAdded = rows.length > 0 &&
    Array.from(rows).every(row => currentSymbols.includes(row.dataset.symbol));
  if (empty) empty.hidden = !allAdded;

  const countEl = dialog.querySelector('[data-count]');
  if (countEl) countEl.textContent = '0';

  const addBtn = dialog.querySelector('[data-add-selected]');
  if (addBtn) addBtn.disabled = true;
}

function updateSelectionCount(dialog) {
  const checked = dialog.querySelectorAll('.add-modal__row input:checked:not(:disabled)');
  const countEl = dialog.querySelector('[data-count]');
  if (countEl) countEl.textContent = String(checked.length);
  const addBtn = dialog.querySelector('[data-add-selected]');
  if (addBtn) addBtn.disabled = checked.length === 0;
}

export async function openAddMarketModal() {
  const dialog = document.getElementById('add-market-modal');
  if (!dialog) return;

 
  const [currentSymbols] = await Promise.all([
    fetchWatchlist(),
    loadLive().catch(err => console.error('Could not load forecasts for the modal', err)),
  ]);
  renderModalRows(dialog, currentSymbols);

  dialog.showModal();

  if (!dialog.dataset.bound) {
    dialog.dataset.bound = '1';

    dialog.addEventListener('change', e => {
      if (e.target.matches('input[name="market"]')) updateSelectionCount(dialog);
    });

    dialog.addEventListener('click', e => {
      if (e.target.closest('[data-close]')) {
        dialog.close();
        return;
      }

      if (e.target.closest('[data-add-selected]')) {
        const selected = Array.from(
          dialog.querySelectorAll('input[name="market"]:checked:not(:disabled)')
        ).map(input => input.value);

        if (!selected.length) return;

     
        const merged = Array.from(new Set([...(cachedSymbols || []), ...selected]));
        saveWatchlist(merged)
          .then(() => {
            dialog.close();
            loadWatchlistPage();
            loadDashboardWatchlist();
            loadDashboardCards();
          })
          .catch(err => console.error('Could not save watchlist', err));
      }
    });
  }
}

/* ---------- Page + dashboard binding ---------- */

export function bindWatchlistPage() {
  loadWatchlistPage();

  const tbody = document.querySelector('#watchlist-table tbody');
  if (tbody && !tbody.dataset.bound) {
    tbody.dataset.bound = '1';
    tbody.addEventListener('click', e => {
      const btn = e.target.closest('[data-remove]');
      if (!btn) return;
      const symbol = btn.dataset.remove;
      const next = (cachedSymbols || []).filter(s => s !== symbol);
      saveWatchlist(next)
        .then(() => {
          loadWatchlistPage();
          loadDashboardWatchlist();
          loadDashboardCards();
        })
        .catch(err => console.error('Could not remove market', err));
    });
  }

  const addBtn = document.querySelector('[data-open="add-market-modal"]');
  if (addBtn && !addBtn.dataset.bound) {
    addBtn.dataset.bound = '1';
    addBtn.addEventListener('click', openAddMarketModal);
  }
}

export function bindDashboardWatchlist() {
  loadDashboardWatchlist();
}