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

const SIGNAL_META = {
  'BOS Bull':       { tone: 'up',      arrow: '▲', description: 'Bullish break of structure' },
  'BOS Bear':       { tone: 'down',    arrow: '▼', description: 'Bearish break of structure' },
  'Bull FVG':       { tone: 'up',      arrow: '▲', description: 'Bullish fair value gap' },
  'Bear FVG':       { tone: 'down',    arrow: '▼', description: 'Bearish fair value gap' },
  'Bull IFVG':      { tone: 'up',      arrow: '▲', description: 'Bullish inverse fair value gap' },
  'Bear IFVG':      { tone: 'down',    arrow: '▼', description: 'Bearish inverse fair value gap' },
  'Bull Breaker':   { tone: 'up',      arrow: '▲', description: 'Bullish breaker block' },
  'Bear Breaker':   { tone: 'down',    arrow: '▼', description: 'Bearish breaker block' },
  'Bull OB':        { tone: 'up',      arrow: '▲', description: 'Bullish order block' },
  'Bear OB':        { tone: 'down',    arrow: '▼', description: 'Bearish order block' },
  'Liquidity Sweep':{ tone: 'neutral', arrow: '◆', description: 'Liquidity taken at recent extreme' },
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

function paintChart(symbol, m) {
  const svg = document.getElementById('price-chart');
  const lastDateEl = document.getElementById('chart-last-date');
  if (!svg) return;

  const candles = Array.isArray(m?.candles) ? m.candles : [];
  if (!candles.length) {
    svg.innerHTML = '<text x="500" y="200" text-anchor="middle" fill="#8b95ad" font-size="14">No candle data</text>';
    return;
  }

  if (lastDateEl) lastDateEl.textContent = `Last session: ${candles[candles.length - 1].date}`;

  const NS = 'http://www.w3.org/2000/svg';
  const W = 1000, H = 400;
  const PAD = { left: 70, right: 20, top: 20, bottom: 40 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  // Price range: candles + trade levels
  const t = m?.trade;
  const closes = candles.map(c => c.close);
  const highs = candles.map(c => c.high);
  const lows = candles.map(c => c.low);
  const extras = [t?.entry, t?.stopLoss, t?.takeProfit].filter(v => typeof v === 'number' && !isNaN(v));
  const allPrices = [...highs, ...lows, ...extras];
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices);
  const pad = (maxP - minP) * 0.05 || 1;
  const lo = minP - pad;
  const hi = maxP + pad;

  const y = p => PAD.top + plotH - ((p - lo) / (hi - lo)) * plotH;
  const barW = plotW / candles.length;
  const bodyW = barW * 0.55;

  let html = '';

  // --- Gridlines + y-axis labels (5 levels) ---
  for (let i = 0; i <= 5; i++) {
    const price = lo + (hi - lo) * (i / 5);
    const yPix = y(price);
    html += `<line x1="${PAD.left}" y1="${yPix}" x2="${W - PAD.right}" y2="${yPix}" stroke="#1b2540" stroke-width="1" vector-effect="non-scaling-stroke"/>`;
    html += `<text x="${PAD.left - 8}" y="${yPix + 4}" text-anchor="end" fill="#8b95ad" font-size="11" font-family="'Space Mono', monospace">${price.toLocaleString('en-US', { maximumFractionDigits: 2 })}</text>`;
  }

  // --- Candles ---
  candles.forEach((c, i) => {
    const x = PAD.left + i * barW + barW / 2;
    const up = c.close >= c.open;
    const color = up ? '#22c55e' : '#ef4444';

    // wick
    html += `<line x1="${x}" y1="${y(c.high)}" x2="${x}" y2="${y(c.low)}" stroke="${color}" stroke-width="1" vector-effect="non-scaling-stroke"/>`;

    // body
    const bodyTop = y(Math.max(c.open, c.close));
    const bodyBot = y(Math.min(c.open, c.close));
    const bodyH = Math.max(bodyBot - bodyTop, 1);
    html += `<rect x="${x - bodyW / 2}" y="${bodyTop}" width="${bodyW}" height="${bodyH}" fill="${color}"/>`;
  });

  // --- Trade level lines ---
  const levels = [
    { price: t?.takeProfit, color: '#22c55e', label: 'TP' },
    { price: t?.entry,      color: '#818cf8', label: 'Entry' },
    { price: t?.stopLoss,   color: '#ef4444', label: 'SL' },
  ].filter(l => typeof l.price === 'number' && !isNaN(l.price));

  levels.forEach(l => {
    const yPix = y(l.price);
    html += `<line x1="${PAD.left}" y1="${yPix}" x2="${W - PAD.right}" y2="${yPix}" stroke="${l.color}" stroke-width="1" stroke-dasharray="6 4" vector-effect="non-scaling-stroke"/>`;
    html += `<text x="${W - PAD.right - 4}" y="${yPix - 4}" text-anchor="end" fill="${l.color}" font-size="11" font-weight="600" font-family="'Space Mono', monospace">${l.label} ${l.price.toLocaleString('en-US', { maximumFractionDigits: 2 })}</text>`;
  });

  // --- X-axis: 5 date labels, evenly spaced ---
  for (let i = 0; i <= 4; i++) {
    const idx = Math.floor((candles.length - 1) * (i / 4));
    const c = candles[idx];
    const x = PAD.left + idx * barW + barW / 2;
    html += `<text x="${x}" y="${H - PAD.bottom + 18}" text-anchor="middle" fill="#8b95ad" font-size="11" font-family="'Space Mono', monospace">${c.date}</text>`;
  }

  svg.innerHTML = html;
}

function paintSmc(symbol, m) {
  const grid = $('smc-grid');
  const wide = $('smc-wide');
  if (!grid || !wide) return;

  const signals = Array.isArray(m?.signals) ? m.signals : [];

  if (!signals.length) {
    grid.innerHTML = '<p class="muted-light">No active SMC signals for this market.</p>';
    wide.innerHTML = '';
    return;
  }

 
  const gridItems = [];
  const wideItems = [];

  signals.forEach(raw => {
    const meta = SIGNAL_META[raw] || { tone: 'neutral', arrow: '◆', description: raw };
    const item = {
      label: raw,
      tone: meta.tone,
      arrow: meta.arrow,
      description: meta.description,
    };
    if (raw.includes('Liquidity') || raw.includes('Sweep')) {
      wideItems.push(item);
    } else {
      gridItems.push(item);
    }
  });

  grid.innerHTML = gridItems.map(item => `
    <div class="signal">
      <span class="tag tag--${item.tone} tag--sm">${item.arrow} ${item.label}</span>
      <p>${item.description}</p>
    </div>
  `).join('');

  wide.innerHTML = wideItems.map(item => `
    <div class="signal signal--wide">
      <span class="tag tag--${item.tone} tag--sm">${item.arrow} ${item.label}</span>
      <p>${item.description}</p>
    </div>
  `).join('');
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
    paintSmc(symbol, m);
    paintChart(symbol, m);
  } catch (err) {
    console.error('Could not load the market', err);
    if (!$('pred-title')) return;
    setText('pred-title', 'Forecast unavailable');
    setText('pred-sub', 'Could not load the latest forecast.');
  }
}