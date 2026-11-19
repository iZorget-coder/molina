import { doc, getDoc, setDoc, collection, addDoc, query, orderBy, getDocs, deleteDoc, serverTimestamp } from 'firebase/firestore';
import { db, auth, onAuthStateChanged } from './firebase.js';

const OPENROUTER_KEY = import.meta.env.VITE_OPENROUTER_API_KEY;
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
const MAX_DAILY_MESSAGES = 5;
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
        input.placeholder = "Daily limit reached. Try again tomorrow.";
    } else {
        input.disabled = false;
        send.disabled = false;
        input.placeholder = "Ask about your watchlist…";
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

async function saveMessage(chatId, role, content) {
    const user = await whenAuthReady();
    if (!user) return;

    const chatRef = doc(db, 'users', user.uid, 'chats', chatId);
    const snap = await getDoc(chatRef);
    const existing = snap.exists() ? snap.data() : { messages: [] };
    const messages = [...(existing.messages || []), { role, content, timestamp: Date.now() }];

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

    list.innerHTML = snap.docs.map(doc => {
        const data = doc.data();
        const title = data.title || 'Chat';
        const isActive = doc.id === currentChatId ? 'is-active' : '';
        return `
            <li>
                <button type="button" class="ai-recent__item ${isActive}" data-chat-id="${doc.id}">
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

function renderMessages(messages) {
    const container = document.getElementById('ai-messages');
    if (!messages.length) {
        container.innerHTML = '<p class="muted-light" style="text-align: center; padding: 40px;">Ask a question to start a new chat.</p>';
        return;
    }

    container.innerHTML = messages.map(m => {
        if (m.role === 'user') {
            return `
                <div class="ai-msg ai-msg--user">
                    <p>${escapeHtml(m.content)}</p>
                </div>
            `;
        } else {
            return `
                <div class="ai-msg ai-msg--ai">
                    <span class="ai-msg__avatar" aria-hidden="true">
                        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/></svg>
                    </span>
                    <div class="ai-msg__body"><p>${escapeHtml(m.content)}</p></div>
                </div>
            `;
        }
    }).join('');

    // Scroll to bottom
    const scroll = document.getElementById('ai-chat-scroll');
    if (scroll) scroll.scrollTop = scroll.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// --- OpenRouter Call ---

async function callOpenRouter(userMessage) {
    
    const body = {
        model: 'openrouter/free',  
        messages: [
            { role: 'system', content: 'You are Molina AI, a helpful assistant for market analysis. Be concise and use plain language.' },
            { role: 'user', content: userMessage }
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
        throw new Error(err.error?.message || `API error ${res.status}`);
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
    document.getElementById('ai-chat-ctx').textContent = 'Context: Watchlist';
}

export async function initAiPage() {
    currentUser = await whenAuthReady();
    const usage = await getUsage();
    renderUsage(usage);
 
    const select = document.getElementById('ai-model');
    if (select && !select.dataset.bound) {
        select.dataset.bound = '1';
        select.innerHTML = '<option value="openrouter/free">Auto (Free Router)</option>';
        select.value = 'openrouter/free';
    }

 
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
            const text = input.value.trim();
            if (!text) return;

            const currentUsage = await getUsage();
            if (currentUsage.count >= MAX_DAILY_MESSAGES) {
                alert('You have reached your daily limit of 5 messages. Please try again tomorrow.');
                return;
            }

            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;

             
            const messages = await loadChat(currentChatId);
            messages.push({ role: 'user', content: text, timestamp: Date.now() });
            renderMessages(messages);

             
            await saveMessage(currentChatId, 'user', text);
            await incrementUsage();
            const newUsage = await getUsage();
            renderUsage(newUsage);

            try {
                const reply = await callOpenRouter(text);
                await saveMessage(currentChatId, 'assistant', reply);
                const updatedMessages = await loadChat(currentChatId);
                renderMessages(updatedMessages);
            } catch (err) {
                console.error(err);
                const errorMsg = err.message.includes('429')
                    ? 'Free tier exhausted for today across all users. Please try again tomorrow.'
                    : 'Something went wrong. Please try again.';
             
                await saveMessage(currentChatId, 'assistant', errorMsg);
                const updatedMessages = await loadChat(currentChatId);
                renderMessages(updatedMessages);
            } finally {
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
                 
                const finalUsage = await getUsage();
                renderUsage(finalUsage);
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