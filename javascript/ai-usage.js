
import { doc, getDoc } from 'firebase/firestore';
import { db, auth, onAuthStateChanged } from './firebase.js';

export const MAX_DAILY_MESSAGES = 5;

function getLocalDate() {
    return new Intl.DateTimeFormat('en-CA', {
        timeZone: 'Africa/Johannesburg',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
    }).format(new Date());
}

function whenAuthReady() {
    if (auth.currentUser) return Promise.resolve(auth.currentUser);

    return new Promise(resolve => {
        const unsubscribe = onAuthStateChanged(auth, user => {
            unsubscribe();
            resolve(user);
        });
    });
}

export async function getAiUsage() {
    const user = await whenAuthReady();

    if (!user) {
        throw new Error('User is not authenticated.');
    }

    const ref = doc(db, 'users', user.uid);
    const snap = await getDoc(ref);

    if (!snap.exists()) {
        return { date: getLocalDate(), count: 0 };
    }

    const data = snap.data();
    const today = getLocalDate();

    if (!data.aiUsage || data.aiUsage.date !== today) {
        return { date: today, count: 0 };
    }

    return {
        date: today,
        count: Math.min(
            MAX_DAILY_MESSAGES,
            Math.max(0, Number(data.aiUsage.count) || 0)
        )
    };
}

export async function renderDashboardAiUsage() {
    const countEl = document.getElementById('dashboard-ai-usage-count');
    const barEl = document.getElementById('dashboard-ai-usage-bar');

    if (!countEl || !barEl) return;

    try {
        const usage = await getAiUsage();
        const count = usage.count;
        const remaining = MAX_DAILY_MESSAGES - count;
        const percentage = (count / MAX_DAILY_MESSAGES) * 100;

        countEl.textContent = `${count} of ${MAX_DAILY_MESSAGES} used today`;
        barEl.style.width = `${percentage}%`;

        barEl.parentElement?.setAttribute('role', 'progressbar');
        barEl.parentElement?.setAttribute('aria-valuemin', '0');
        barEl.parentElement?.setAttribute('aria-valuemax', String(MAX_DAILY_MESSAGES));
        barEl.parentElement?.setAttribute('aria-valuenow', String(count));

    } catch (error) {
        console.error('Could not load dashboard AI usage:', error);
        countEl.textContent = 'Usage unavailable';
        barEl.style.width = '0%';
    }
}

export async function renderSettingsAiUsage() {
    const countEl = document.getElementById('settings-ai-usage-count');
    const leftEl = document.getElementById('settings-ai-usage-left');
    const barEl = document.getElementById('settings-ai-usage-bar');
    const progressEl = document.getElementById('settings-ai-usage-progress');

    if (!countEl || !leftEl || !barEl || !progressEl) return;

    try {
        const usage = await getAiUsage();
        const count = usage.count;
        const remaining = MAX_DAILY_MESSAGES - count;
        const percentage = (count / MAX_DAILY_MESSAGES) * 100;

        countEl.textContent = `${count} of ${MAX_DAILY_MESSAGES} messages used today`;
        leftEl.textContent = `${remaining} left`;
        barEl.style.width = `${percentage}%`;
        progressEl.setAttribute('aria-valuenow', String(count));

    } catch (error) {
        console.error('Could not load settings AI usage:', error);
        countEl.textContent = 'Usage unavailable';
        leftEl.textContent = '—';
        barEl.style.width = '0%';
    }
}
