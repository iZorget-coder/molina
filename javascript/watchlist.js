import { doc, getDoc, setDoc } from 'firebase/firestore';
import { db, auth, onAuthStateChanged } from './firebase.js';

const ALL_MARKETS = [
  { symbol: 'SPX',    name: 'S&P 500',         sub: 'SPX · Index',        price: '5,892.40',  dir: 'up',   label: '▲ UP 62%',            change: '+0.42%', confidence: 62 },
  { symbol: 'NDX',    name: 'NASDAQ',          sub: 'NDX · Index',        price: '20,431.18', dir: 'down', label: '▼ DOWN 58%',          change: '-0.71%', confidence: 58 },
  { symbol: 'DJI',    name: 'DOW JONES',       sub: 'DJI · Index',        price: '43,128.65', dir: 'flat', label: '◆ CONSOLIDATION 51%', change: '+0.05%', confidence: 51 },
  { symbol: 'XAUUSD', name: 'Gold',            sub: 'XAUUSD · Commodity', price: '2,447.85',  dir: 'up',   label: '▲ UP 62%',            change: '+0.88%', confidence: 62 },
  { symbol: 'DXY',    name: 'US Dollar Index', sub: 'DXY · FX',           price: '104.27',    dir: 'down', label: '▼ DOWN 55%',          change: '-0.31%', confidence: 55 },
];

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
  return doc(db, 'watchlists', uid);
}

async function fetchWatchlist() {
  if (cachedSymbols) return cachedSymbols;

  const user = await whenAuthReady();
  if (!user) {
    cachedSymbols = ['SPX'];
    return cachedSymbols;
  }

  const ref = userRef(user.uid);
  const snap = await getDoc(ref);

  if (snap.exists()) {
    cachedSymbols = snap.data().symbols || ['SPX'];
  } else {
    cachedSymbols = ['SPX'];
    await setDoc(ref, { symbols: cachedSymbols, updatedAt: new Date().toISOString() });
  }
  return cachedSymbols;
}

async function saveWatchlist(symbols) {
  const user = await whenAuthReady();
  cachedSymbols = symbols;
  if (!user) return;
  await setDoc(userRef(user.uid), {
    symbols,
    updatedAt: new Date().toISOString(),
  }, { merge: true });
}

function getMarket(symbol) {
  return ALL_MARKETS.find(m => m.symbol === symbol);
}

/* ---------- Watchlist page rendering ---------- */

function watchRowMarkup(market) {
  const dirClass = market.dir === 'up' ? 'up' : market.dir === 'down' ? 'down' : 'flat';
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
      <td class="col-change num ${dirClass}">${market.change}</td>
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
    const symbols = await fetchWatchlist();
    const markets = symbols.map(getMarket).filter(Boolean);

    if (!markets.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="view-empty">Your watchlist is empty. Add a market to get started.</td></tr>';
    } else {
      tbody.innerHTML = markets.map(watchRowMarkup).join('');
    }

    const countEl = document.getElementById('watchlist-count');
    if (countEl) countEl.textContent = `${symbols.length} markets`;
  } catch (err) {
    console.error(err);
    tbody.innerHTML = '<tr><td colspan="5" class="view-error">Could not load watchlist.</td></tr>';
  }
}

/* ---------- Dashboard sidebar rendering ---------- */

function dashboardItemMarkup(market) {
  const dirClass = market.dir === 'up' ? 'up' : market.dir === 'down' ? 'down' : 'flat';
  return `
    <li class="watch__item">
      <div class="watch__row">
        <div>
          <div class="watch__name">${market.name}</div>
          <div class="watch__sub">${market.sub}</div>
        </div>
        <div class="watch__price">${market.price}</div>
      </div>
      <div class="watch__row">
        <span class="badge badge--${market.dir}">${market.label}</span>
        <span class="change ${dirClass}">${market.change}</span>
      </div>
    </li>
  `;
}

export async function loadDashboardWatchlist() {
  const list = document.querySelector('.watch');
  if (!list) return;

  try {
    const symbols = await fetchWatchlist();
    const markets = symbols.map(getMarket).filter(Boolean);

    list.innerHTML = markets.length
      ? markets.map(dashboardItemMarkup).join('')
      : '<li class="watch__item"><p class="view-empty">No markets yet.</p></li>';

    const countEl = document.querySelector('.sidebar__head .muted');
    if (countEl) countEl.textContent = `${symbols.length} markets`;
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
    if (market.dir === 'up')       y -= rand() * 2.5;
    else if (market.dir === 'down') y += rand() * 2.5;
    else                            y  = 40 + Math.sin(i * 1.7 + seedBase) * 10;
    const clamped = Math.max(2, Math.min(54, y));
    points.push(`${x.toFixed(0)},${clamped.toFixed(0)}`);
  }
  return points.join(' ');
}

function cardMarkup(market) {
  const pillClass = market.dir === 'up' ? 'pill--up' : market.dir === 'down' ? 'pill--down' : 'pill--flat';
  const pillLabel = market.dir === 'up' ? '▲ UP' : market.dir === 'down' ? '▼ DOWN' : '◆ CONSOLIDATION';
  const ticker = market.sub.split(' · ')[0];

  const changeClass = market.change.startsWith('-') ? 'down'
                    : market.change === '+0.00%' ? 'flat'
                    : 'up';

  return `
    <article class="card">
      <div class="card__id">
        <div class="card__title">${market.name} <span class="ticker">${ticker}</span></div>
        <span class="pill ${pillClass}">${pillLabel}</span>
      </div>
      <div class="card__conf">
        <div class="conf__row"><span class="conf__label">Confidence</span><span class="conf__value">${market.confidence}%</span></div>
        <div class="progress"><div class="progress__bar" style="width:${market.confidence}%"></div></div>
        <div class="conf__row conf__row--small">
          <span class="muted-light">Last close <strong>${market.price}</strong></span>
          <span class="${changeClass}">${market.change}</span>
        </div>
      </div>
      <div class="card__chart">
        <svg viewBox="0 0 200 56" preserveAspectRatio="none" class="spark spark--${market.dir}">
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
    const symbols = await fetchWatchlist();
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
    const inList = currentSymbols.includes(symbol);
    input.disabled = inList;
    input.checked = false;
  });

  const empty = dialog.querySelector('.add-modal__empty');
  if (empty) empty.hidden = currentSymbols.length < ALL_MARKETS.length;

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

  const currentSymbols = await fetchWatchlist();
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

        const merged = Array.from(new Set([...currentSymbols, ...selected]));
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