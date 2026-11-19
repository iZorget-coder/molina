import { collection, query, orderBy, limit, getDocs } from 'firebase/firestore';
import { auth, onAuthStateChanged, db } from './firebase.js';

/* ---------- The five markets the app supports ---------- */
export const MARKETS = [
  { symbol: 'SPX',    name: 'S&P 500',         type: 'Index' },
  { symbol: 'NDX',    name: 'NASDAQ',          type: 'Index' },
  { symbol: 'DJI',    name: 'DOW JONES',       type: 'Index' },
  { symbol: 'XAUUSD', name: 'Gold',            type: 'Commodity' },
  { symbol: 'DXY',    name: 'US Dollar Index', type: 'FX' },
];

/* ---------- Direction helpers ---------- */
const DIRECTIONS = {
  UP:            { key: 'up',   label: 'UP',            arrow: '▲' },
  DOWN:          { key: 'down', label: 'DOWN',          arrow: '▼' },
  CONSOLIDATION: { key: 'flat', label: 'CONSOLIDATION', arrow: '◆' },
};
const NO_FORECAST = { key: 'none', label: 'NO FORECAST', arrow: '' };

export const directionOf = (bias) =>
  DIRECTIONS[String(bias || '').toUpperCase()] || NO_FORECAST;

export function countDirections(data) {
  const counts = { up: 0, down: 0, flat: 0 };
  if (!data) return counts;
  for (const p of data.list) {
    if (p.dir in counts) counts[p.dir] += 1;
  }
  return counts;
}

/* ---------- Formatting ---------- */
const formatPrice = (n) =>
  typeof n === 'number'
    ? n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : '—';

export function formatChange(pct) {
  if (typeof pct !== 'number' || !Number.isFinite(pct)) return '—';
  const v = Math.round(pct * 100) / 100;
  return `${v < 0 ? '-' : '+'}${Math.abs(v).toFixed(2)}%`;
}

// dates are plain YYYY-MM-DD strings
const utc = (iso) => {
  const [y, m, d] = String(iso).split('-').map(Number);
  return new Date(Date.UTC(y, m - 1, d));
};

export const formatDay = (iso) =>
  iso
    ? utc(iso)
        .toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' })
        .replace(',', '')
    : '';

export const formatLong = (iso) =>
  iso
    ? utc(iso)
        .toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
        .replace(',', '')
    : '';

export const weekdayLong = (iso) =>
  iso ? utc(iso).toLocaleDateString('en-GB', { weekday: 'long', timeZone: 'UTC' }) : '';

// "18 Nov 2026"
export const formatDate = (iso) =>
  iso
    ? utc(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' })
    : '';


export function nextTradingDay(iso) {
  const dt = utc(iso);
  do {
    dt.setUTCDate(dt.getUTCDate() + 1);
  } while (dt.getUTCDay() === 0 || dt.getUTCDay() === 6);
  return dt.toISOString().slice(0, 10);
}

/* ---------- Shape the Firestore documents for the UI ---------- */

function previousClose(olderPredictions, symbol, closeDate) {
  for (const preds of olderPredictions) {
    const p = preds[symbol];
    if (p && typeof p.close === 'number' && p.date && p.date !== closeDate) return p.close;
  }
  return null;
}

// The "trade" map the pipeline stores next to each prediction (entry, stop, target, risk/reward)
function normaliseTrade(t) {
  if (!t || typeof t !== 'object') return null;
  const num = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  return {
    action: t.action ? String(t.action).toUpperCase() : null,    
    conviction: t.conviction ?? null,                           
    entry: num(t.entry),
    stopLoss: num(t.stop_loss),
    takeProfit: num(t.take_profit),
    risk: num(t.risk),                                          
    reward: num(t.reward),                                      
    riskReward: num(t.risk_reward),
    entryType: t.entry_type ?? null,
    stopSource: t.stop_source ?? null,
    targetSource: t.target_source ?? null,
  };
}


function normalise(docId, data, olderPredictions) {
  const raw = data.predictions || {};
  const bySymbol = {};

  for (const m of MARKETS) {
    const p = raw[m.symbol];
    const direction = directionOf(p?.bias);
    const hasForecast = Boolean(p) && direction.key !== 'none';

    const confidence = typeof p?.confidence === 'number'
      ? Math.round(p.confidence * 100)
      : null;

    // Individual probabilities for all three market directions
    const probabilities = {
      up: typeof p?.probabilities?.UP === 'number'
        ? Math.round(p.probabilities.UP * 100)
        : null,

      down: typeof p?.probabilities?.DOWN === 'number'
        ? Math.round(p.probabilities.DOWN * 100)
        : null,

      consolidation: typeof p?.probabilities?.CONSOLIDATION === 'number'
        ? Math.round(p.probabilities.CONSOLIDATION * 100)
        : null,
    };

    const close = typeof p?.close === 'number' ? p.close : null;
    const closeDate = p?.date ?? null;

    let change = null;
    if (close !== null && closeDate) {
      const prev = previousClose(olderPredictions, m.symbol, closeDate);
      if (prev) change = (close / prev - 1) * 100;
    }

    bySymbol[m.symbol] = {
      ...m,
      sub: `${m.symbol} · ${m.type}`,
      hasForecast,
      bias: p?.bias ?? null,
      direction,
      dir: direction.key,
      confidence,
      probabilities,
      close,
      closeDate,
      trade: normaliseTrade(p?.trade),
      change,
      price: formatPrice(close),
      changeText: formatChange(change),
      label: hasForecast
        ? `${direction.arrow} ${direction.label}${confidence !== null ? ` ${confidence}%` : ''}`
        : 'No forecast',
    };
  }

  const closeDates = Object.values(raw)
    .map(p => p?.date)
    .filter(Boolean)
    .sort();

  const closeDate = closeDates.length
    ? closeDates[closeDates.length - 1]
    : null;

  return {
    forecastDate: docId,
    forecastDay: closeDate ? nextTradingDay(closeDate) : docId,
    closeDate,
    generatedAt: data.generated_at?.toDate
      ? data.generated_at.toDate()
      : null,
    bySymbol,
    list: MARKETS.map(m => bySymbol[m.symbol]),
  };
}


/* ---------- Fetch the newest forecast documents ---------- */
// Firestore only answers signed-in users, so wait for Firebase to restore the session first.
const authReady = new Promise((resolve) => {
  const stop = onAuthStateChanged(auth, (user) => {
    stop();
    resolve(user);
  });
});

const CACHE_MS = 10 * 60 * 1000;
let cached = null;
let cachedAt = 0;
let inflight = null;

export function fetchLatestPredictions({ force = false } = {}) {
  if (cached && !force && Date.now() - cachedAt < CACHE_MS) return Promise.resolve(cached);
  if (inflight) return inflight;

  inflight = (async () => {
    await authReady;
    
    const q = query(collection(db, 'daily_predictions'), orderBy('generated_at', 'desc'), limit(5));
    const snap = await getDocs(q);
    if (snap.empty) return null;

    const [latest, ...older] = snap.docs;
    cached = normalise(
      latest.id,
      latest.data(),
      older.map((d) => d.data().predictions || {})
    );
    cachedAt = Date.now();
    return cached;
  })().finally(() => {
    inflight = null;
  });

  return inflight;
}