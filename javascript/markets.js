import { MARKETS, fetchLatestPredictions, formatLong } from './predictions.js';

const STORAGE_KEY = 'molina:market';
const DEFAULT_SYMBOL = 'XAUUSD';

const DESCRIPTIONS = {
  SPX: 'S&P 500 index',
  NDX: 'NASDAQ index',
  DJI: 'Dow Jones Industrial Average',
  XAUUSD: 'Gold spot / US dollar',
  DXY: 'US Dollar Index',
};

// Only gold is priced in dollars; the indices are plain points
const LEVEL_PREFIX = { XAUUSD: '$' };

const isSymbol = (s) => MARKETS.some((m) => m.symbol === s);

/* ---------- Which market is open ---------- */

export function selectMarket(symbol) {
  if (isSymbol(symbol)) sessionStorage.setItem(STORAGE_KEY, symbol);
}

// Back/forward restore the market through history.state; otherwise use the last one opened
export function selectedSymbol() {
  const fromState = history.state?.symbol;
  if (isSymbol(fromState)) return fromState;
  const stored = sessionStorage.getItem(STORAGE_KEY);
  return isSymbol(stored) ? stored : DEFAULT_SYMBOL;
}

/* ---------- Small helpers ---------- */

const $ = (id) => document.getElementById(id);

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

const fmt = (n) => n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const level = (symbol, n) => (n === null ? '—' : `${LEVEL_PREFIX[symbol] || ''}${fmt(n)}`);

const distance = (symbol, n) =>
  n === null ? '—' : LEVEL_PREFIX[symbol] ? `${LEVEL_PREFIX[symbol]}${fmt(n)}` : `${fmt(n)} pts`;

/* ---------- Header: symbol, price, day change ---------- */

function paintHeader(symbol, m) {
  setText('mk-symbol', symbol);
  setText('mk-sub', `${DESCRIPTIONS[symbol]} · Daily`);
  setText('mk-price', m ? m.price : '—');

  const delta = $('mk-delta');
  if (!delta) return;
  if (m && m.change !== null) {
    const up = m.change >= 0;
    const tone = Math.abs(m.change) < 0.1 ? 'flat' : up ? 'up' : 'down';
    delta.className = `delta ${tone}`;
    delta.textContent = `${up ? '▲' : '▼'} ${m.changeText}`;
  } else {
    delta.className = 'delta';
    delta.textContent = '';
  }
}

/* ---------- Prediction card ---------- */

const TAG_CLASS = { up: 'tag--up', down: 'tag--down', flat: 'tag--flat', none: 'tag--neutral' };
const ICON = { up: '▲', down: '▼', flat: '◆', none: '–' };


function paintPrediction(symbol, m, live) {
  const dir = m?.dir ?? 'none';
  const has = Boolean(m?.hasForecast);
  const confidence = has ? m.confidence : null;

  const icon = $('pred-icon');
  if (icon) {
    icon.textContent = ICON[dir];
    icon.className = `prediction__icon prediction__icon--${dir}`;
  }

  setText(
    'pred-title',
    has
      ? `${symbol} — Predicted ${m.direction.label} with ${confidence ?? '—'}% confidence`
      : `${symbol} — No forecast yet`
  );

  const conviction = m?.trade?.conviction;
  setText(
    'pred-sub',
    has
      ? `Forecast for ${formatLong(live.forecastDay)} · daily model${conviction ? ` · ${conviction} conviction` : ''}`
      : 'Waiting for the next daily run'
  );

  const tag = $('pred-tag');
  if (tag) {
    tag.className = `tag ${TAG_CLASS[dir]}`;
    tag.textContent = has
      ? `${m.direction.arrow} ${m.direction.label}`
      : 'NO FORECAST';
  }

  setText('pred-conf', confidence === null ? '—' : `${confidence}%`);

  const bar = $('pred-conf-bar');
  if (bar) bar.style.width = `${confidence ?? 0}%`;

  // Display all three model probabilities
  const probs = $('pred-probs');

  if (probs) {
    const probabilities = m?.probabilities;

    if (has && probabilities) {
      const outcomes = [
        {
          key: 'up',
          label: 'UP',
          arrow: '▲',
          value: probabilities.up,
          className: 'up',
        },
        {
          key: 'down',
          label: 'DOWN',
          arrow: '▼',
          value: probabilities.down,
          className: 'down',
        },
        {
          key: 'consolidation',
          label: 'CONSOLIDATION',
          arrow: '◆',
          value: probabilities.consolidation,
          className: 'flat',
        },
      ];

      probs.innerHTML = outcomes.map(outcome => {
        const value = outcome.value;
        const percentage = value ?? 0;

        return `
          <li>
            <span class="probs__label ${outcome.className}">
              ${outcome.arrow} ${outcome.label}
            </span>

            <div class="bar">
              <div
                class="bar__fill bar__fill--${outcome.className}"
                style="width:${percentage}%">
              </div>
            </div>

            <span class="probs__val">
              ${value === null ? '—' : `${value}%`}
            </span>
          </li>
        `;
      }).join('');
    } else {
      probs.innerHTML = '';
    }
  }
}


/* ---------- Trade setup card ---------- */

function paintTrade(symbol, m) {
  const t = m?.trade ?? null;

  const tag = $('setup-tag');
  if (tag) {
    const action = t?.action;
    const cls = action === 'LONG' ? 'tag--up' : action === 'SHORT' ? 'tag--down' : 'tag--neutral';
    tag.className = `tag ${cls} tag--sm`;
    tag.textContent = action === 'LONG' ? '▲ LONG' : action === 'SHORT' ? '▼ SHORT' : action || 'NO SETUP';
  }

  setText('setup-entry', level(symbol, t?.entry ?? null));
  setText('setup-stop', level(symbol, t?.stopLoss ?? null));
  setText('setup-target', level(symbol, t?.takeProfit ?? null));

 
  const rr =
    t?.riskReward ??
    (t && t.risk && t.reward !== null ? t.reward / t.risk : null);
  setText('setup-rr', rr === null ? '—' : `1 : ${Number.isInteger(rr) ? rr : rr.toFixed(2)}`);

  const riskBar = $('setup-rr-risk');
  const rewardBar = $('setup-rr-reward');
  if (riskBar) riskBar.style.flex = '1';
  if (rewardBar) rewardBar.style.flex = rr === null ? '1' : String(rr);

  setText('setup-risk', `Risk ${distance(symbol, t?.risk ?? null)}`);
  setText('setup-reward', `Reward ${distance(symbol, t?.reward ?? null)}`);

  // Where each level came from
  const notes = $('setup-notes');
  if (notes) {
    const rows = [
      ['note-entry', t?.entryType],
      ['note-stop', t?.stopSource],
      ['note-target', t?.targetSource],
    ];
    rows.forEach(([id, text]) => setText(id, text || '—'));
    notes.hidden = !t || rows.every(([, text]) => !text);
  }
}

/* ---------- Page entry point ---------- */

export async function initMarketPage() {
  const symbol = selectedSymbol();

  // show the right name straight away, then fill in the live data
  setText('mk-symbol', symbol);
  setText('mk-sub', `${DESCRIPTIONS[symbol]} · Daily`);

  try {
    const live = await fetchLatestPredictions();
    if (!$('pred-title') || selectedSymbol() !== symbol) return; 

    const m = live?.bySymbol[symbol] ?? null;
    paintHeader(symbol, m);
    paintPrediction(symbol, m, live);
    paintTrade(symbol, m);
  } catch (err) {
    console.error('Could not load the market', err);
    if (!$('pred-title')) return;
    setText('pred-title', 'Forecast unavailable');
    setText('pred-sub', 'Could not load the latest forecast.');
  }
}