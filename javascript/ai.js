import { doc, getDoc, setDoc, collection, query, orderBy, getDocs, serverTimestamp } from 'firebase/firestore';
import { db, auth, onAuthStateChanged } from './firebase.js';
import { fetchLatestPredictions, formatLong } from './predictions.js';
import { selectedSymbol } from './markets.js';

const OPENROUTER_KEY = import.meta.env.VITE_OPENROUTER_API_KEY;
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
const OPENROUTER_MODELS_URL = 'https://openrouter.ai/api/v1/models';
const FALLBACK_MODEL = 'openrouter/free';
const MAX_DAILY_MESSAGES = 5;
const MAX_HISTORY_MESSAGES = 10;
const STORAGE_KEY = 'molina:ai:chatId';

let currentUser = null;
let currentChatId = null;
let authReady = null;

function whenAuthReady() {
    if (authReady) return authReady;
    authReady = new Promise(resolve => {
        const unsub = onAuthStateChanged(auth, user => { unsub(); resolve(user); });
    });
    return authReady;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatMessage(text) {
    return escapeHtml(text)
        .replace(/\*\*(.+?)\*\*/gs, '<strong>$1</strong>')
        .split(/\n{2,}/)
        .map(p => `<p>${p.trim().replace(/\n/g, '<br>')}</p>`)
        .join('');
}

// --- Usage Limits ---

async function getUsage() {
    const user = await whenAuthReady();
    if (!user) return { count: 0, date: '' };

    const ref = doc(db, 'users', user.uid);
    const snap = await getDoc(ref);
    const data = snap.exists() ? snap.data() : {};
    const today = new Date().toISOString().slice(0, 10);

    if (!data.aiUsage || data.aiUsage.date !== today) {
        // Reset for a new day
        const newUsage = { date: today, count: 0 };
        await setDoc(ref, { aiUsage: newUsage }, { merge: true });
        return newUsage;
    }
    return data.aiUsage;
}

async function incrementUsage() {
    const user = await whenAuthReady();
    if (!user) return;
    const ref = doc(db, 'users', user.uid);
    const usage = await getUsage();
    await setDoc(ref, { aiUsage: { ...usage, count: usage.count + 1 } }, { merge: true });
}

function renderUsage(usage) {
    const left = Math.max(0, MAX_DAILY_MESSAGES - usage.count);
    const pct = (usage.count / MAX_DAILY_MESSAGES) * 100;

    document.getElementById('ai-usage-count').textContent = `${usage.count} / ${MAX_DAILY_MESSAGES}`;
    document.getElementById('ai-usage-bar').style.width = `${pct}%`;
    document.getElementById('ai-usage-text').textContent = `${left} message${left === 1 ? '' : 's'} left today`;
    document.getElementById('ai-input-hint').textContent = `${left} left today`;

    const input = document.getElementById('ai-input');
    const send = document.getElementById('ai-send');
    if (left <= 0) {
        input.disabled = true;
        send.disabled = true;
        input.placeholder = 'Daily limit reached. Try again tomorrow.';
    } else {
        input.disabled = false;
        send.disabled = false;
        input.placeholder = `Ask about ${selectedSymbol()}…`;
    }
}

// --- Chat Persistence ---

function generateChatId() {
    return 'chat_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
}

async function loadChat(chatId) {
    const user = await whenAuthReady();
    if (!user || !chatId) return [];

    const chatRef = doc(db, 'users', user.uid, 'chats', chatId);
    const snap = await getDoc(chatRef);
    if (!snap.exists()) return [];

    const data = snap.data();
    return data.messages || [];
}

async function saveMessage(chatId, role, content, extra = {}) {
    const user = await whenAuthReady();
    if (!user) return;

    const chatRef = doc(db, 'users', user.uid, 'chats', chatId);
    const snap = await getDoc(chatRef);
    const existing = snap.exists() ? snap.data() : { messages: [] };
    const messages = [...(existing.messages || []), { role, content, timestamp: Date.now(), ...extra }];

    await setDoc(chatRef, {
        title: messages.find(m => m.role === 'user')?.content.slice(0, 40) || 'New chat',
        messages,
        updatedAt: serverTimestamp(),
    }, { merge: true });
}

async function renderRecentChats() {
    const user = await whenAuthReady();
    if (!user) return;

    const list = document.getElementById('ai-recent-list');
    const chatsRef = collection(db, 'users', user.uid, 'chats');
    const q = query(chatsRef, orderBy('updatedAt', 'desc'));
    const snap = await getDocs(q);

    if (snap.empty) {
        list.innerHTML = '<li><p class="muted-light" style="padding: 10px;">No recent chats.</p></li>';
        return;
    }

    list.innerHTML = snap.docs.map(d => {
        const data = d.data();
        const title = escapeHtml(data.title || 'Chat');
        const isActive = d.id === currentChatId ? 'is-active' : '';
        return `
            <li>
                <button type="button" class="ai-recent__item ${isActive}" data-chat-id="${escapeHtml(d.id)}">
                    <span class="ai-recent__title">${title}</span>
                    <span class="ai-recent__date">Today</span>
                </button>
            </li>
        `;
    }).join('');

    list.querySelectorAll('.ai-recent__item').forEach(btn => {
        btn.addEventListener('click', () => {
            currentChatId = btn.dataset.chatId;
            sessionStorage.setItem(STORAGE_KEY, currentChatId);
            initChatSession(currentChatId);
        });
    });
}

// --- UI Rendering ---

const AI_AVATAR = `
    <span class="ai-msg__avatar" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/></svg>
    </span>`;

function renderMessages(messages, { pending = false } = {}) {
    const container = document.getElementById('ai-messages');
    if (!messages.length && !pending) {
        container.innerHTML = '<p class="muted-light" style="text-align: center; padding: 40px;">Ask a question to start a new chat.</p>';
        return;
    }

    const html = messages.map(m => {
        if (m.role === 'user') {
            return `
                <div class="ai-msg ai-msg--user">
                    ${formatMessage(m.content)}
                </div>
            `;
        }
        return `
            <div class="ai-msg ai-msg--ai">
                ${AI_AVATAR}
                <div class="ai-msg__body">${formatMessage(m.content)}</div>
            </div>
        `;
    }).join('');

    // While waiting for the model, show an animated "thinking" bubble
    const typing = pending ? `
        <div class="ai-msg ai-msg--ai ai-msg--typing" role="status" aria-label="Molina AI is thinking">
            ${AI_AVATAR}
            <div class="ai-msg__body">
                <span class="ai-typing" aria-hidden="true"><i></i><i></i><i></i></span>
                <span class="ai-typing__text">Thinking…</span>
            </div>
        </div>
    ` : '';

    container.innerHTML = html + typing;

    // Scroll to bottom
    const scroll = document.getElementById('ai-chat-scroll');
    if (scroll) scroll.scrollTop = scroll.scrollHeight;
}

// --- Market context (which market the user came from on the Markets page) ---

async function renderContext() {
    const symbol = selectedSymbol();
    const ctx = document.getElementById('ai-chat-ctx');
    if (ctx) ctx.textContent = `Context: ${symbol}`;

    const input = document.getElementById('ai-input');
    if (input && !input.disabled) input.placeholder = `Ask about ${symbol}…`;

    const card = document.getElementById('ai-ctx-card');
    if (!card) return;

    const live = await fetchLatestPredictions().catch(() => null);
    if (selectedSymbol() !== symbol) return; // user moved on while loading

    const m = live?.bySymbol?.[symbol];
    card.innerHTML = m?.hasForecast
        ? `
            <h3>${escapeHtml(symbol)}</h3>
            <p>${escapeHtml(m.direction.label)} · ${m.confidence}% confidence</p>
            <p class="muted-light">${m.signals?.length ? escapeHtml(m.signals.join(' · ')) : 'No active SMC signals'}</p>
        `
        : `<h3>${escapeHtml(symbol)}</h3><p class="muted-light">No forecast yet</p>`;
}

function describeMarket(symbol, live) {
    const m = live?.bySymbol?.[symbol];
    if (!m || !m.hasForecast) return `Market: ${symbol}. No forecast is available for it yet.`;

    const t = m.trade;
    const p = m.probabilities;
    const lines = [
        `Market: ${symbol}${m.price ? ` (last price ${m.price})` : ''}`,
        `Forecast for: ${formatLong(live.forecastDay)} (daily model)`,
        `Predicted direction: ${m.direction.label} with ${m.confidence}% confidence`,
    ];

    if (p) {
        lines.push(`Probabilities: up ${p.up}%, down ${p.down}%, consolidation ${p.consolidation}%`);
    }

    if (t) {
        lines.push(
            `Trade setup: ${t.action || 'none'}, entry ${t.entry ?? 'n/a'}, stop loss ${t.stopLoss ?? 'n/a'}, take profit ${t.takeProfit ?? 'n/a'}, R:R ${t.riskReward ?? 'n/a'}`,
            `Entry type: ${t.entryType || 'n/a'}; stop source: ${t.stopSource || 'n/a'}; target source: ${t.targetSource || 'n/a'}`
        );
        if (t.conviction) lines.push(`Conviction: ${t.conviction}`);
    }

    lines.push(`Active SMC signals: ${m.signals?.length ? m.signals.join(', ') : 'none'}`);
    return lines.join('\n');
}

function buildSystemPrompt(symbol, live) {
    return [
        'You are Molina AI, a market analysis assistant inside the Molina app.',
        `The user is asking about ${symbol}. Below is the model output for it.`,
        'Explain the forecast using ICT / Smart Money Concepts (break of structure, fair value gaps, inverse FVGs, order blocks, breaker blocks, liquidity sweeps) and tie each point to the signals and levels listed.',
        'Use only the data provided for numbers and levels. If something is missing, say so rather than inventing it.',
        'Forecasts are probabilistic. Do not give personal financial advice. Be concise and use plain language.',
        '',
        '--- MODEL DATA ---',
        describeMarket(symbol, live),
    ].join('\n');
}

// --- Free model list ---

async function fetchFreeModels() {
    try {
        const res = await fetch(OPENROUTER_MODELS_URL);
        if (!res.ok) return [];
        const { data = [] } = await res.json();
        return data
            .filter(m => {
                const p = m.pricing || {};
                return parseFloat(p.prompt || '0') === 0 && parseFloat(p.completion || '0') === 0;
            })
            .filter(m => !m.id.startsWith('openrouter/')) // meta-routers are added separately
            .filter(m => (m.architecture?.output_modalities || ['text']).includes('text'))
            .map(m => ({ id: m.id, name: m.name || m.id }))
            .sort((a, b) => a.name.localeCompare(b.name));
    } catch {
        return [];
    }
}

async function populateModelSelect() {
    const select = document.getElementById('ai-model');
    if (!select || select.dataset.bound) return;
    select.dataset.bound = '1';

    // Show the fallback immediately so the control is never empty while loading
    select.innerHTML = `<option value="${FALLBACK_MODEL}">Auto (Free Router)</option>`;
    select.value = FALLBACK_MODEL;

    const models = await fetchFreeModels();
    const options = [{ id: FALLBACK_MODEL, name: 'Auto (Free Router)' }, ...models];
    select.innerHTML = options
        .map(m => `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)}</option>`)
        .join('');
    select.value = FALLBACK_MODEL;
}

// --- OpenRouter Call ---

async function callOpenRouter(history, symbol) {
    const live = await fetchLatestPredictions().catch(() => null);
    const model = document.getElementById('ai-model')?.value || FALLBACK_MODEL;

    const body = {
        model,
        messages: [
            { role: 'system', content: buildSystemPrompt(symbol, live) },
            ...history
                .filter(m => !m.isError)
                .slice(-MAX_HISTORY_MESSAGES)
                .map(({ role, content }) => ({ role, content })),
        ],
    };

    const res = await fetch(OPENROUTER_URL, {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${OPENROUTER_KEY}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': window.location.origin,
            'X-OpenRouter-Title': 'Molina',
        },
        body: JSON.stringify(body),
    });

    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        const error = new Error(err.error?.message || `API error ${res.status}`);
        error.status = res.status;
        throw error;
    }

    const data = await res.json();
    return data.choices?.[0]?.message?.content || 'No response received.';
}

// --- Session Initialization ---

async function initChatSession(chatId) {
    currentChatId = chatId || sessionStorage.getItem(STORAGE_KEY) || generateChatId();
    sessionStorage.setItem(STORAGE_KEY, currentChatId);

    const messages = await loadChat(currentChatId);
    renderMessages(messages);
    await renderRecentChats();

    const firstUserMsg = messages.find(m => m.role === 'user');
    document.getElementById('ai-chat-title').textContent = firstUserMsg ? firstUserMsg.content.slice(0, 50) : 'Molina AI';

    renderContext();
}

export async function initAiPage() {
    currentUser = await whenAuthReady();
    const usage = await getUsage();
    renderUsage(usage);

    populateModelSelect();

    const newBtn = document.getElementById('ai-new-chat');
    if (newBtn && !newBtn.dataset.bound) {
        newBtn.dataset.bound = '1';
        newBtn.addEventListener('click', () => {
            const newId = generateChatId();
            sessionStorage.setItem(STORAGE_KEY, newId);
            initChatSession(newId);
        });
    }

    // Send button
    const sendBtn = document.getElementById('ai-send');
    const input = document.getElementById('ai-input');
    if (sendBtn && !sendBtn.dataset.bound) {
        sendBtn.dataset.bound = '1';

        const send = async () => {
            if (input.disabled) return;
            const text = input.value.trim();
            if (!text) return;

            const currentUsage = await getUsage();
            if (currentUsage.count >= MAX_DAILY_MESSAGES) {
                alert(`You have reached your daily limit of ${MAX_DAILY_MESSAGES} messages. Please try again tomorrow.`);
                return;
            }

            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;

            const symbol = selectedSymbol();
            const messages = await loadChat(currentChatId);
            messages.push({ role: 'user', content: text, timestamp: Date.now() });
            input.placeholder = 'Molina AI is thinking…';
            renderMessages(messages, { pending: true });

            await saveMessage(currentChatId, 'user', text);

            try {
                const reply = await callOpenRouter(messages, symbol);
                await saveMessage(currentChatId, 'assistant', reply);
                // Only count the message once the model actually answered
                await incrementUsage();
            } catch (err) {
                console.error(err);
                const errorMsg = err.status === 429
                    ? 'Free tier exhausted for today across all users. Please try again tomorrow.'
                    : 'Something went wrong. Please try again.';
                await saveMessage(currentChatId, 'assistant', errorMsg, { isError: true });
            } finally {
                renderMessages(await loadChat(currentChatId));
                renderRecentChats();
                renderUsage(await getUsage());
                input.focus();
            }
        };

        sendBtn.addEventListener('click', send);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') send();
        });
    }

    // Load existing chat or start new one
    initChatSession(sessionStorage.getItem(STORAGE_KEY));
}