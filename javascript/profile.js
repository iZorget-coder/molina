import { doc, getDoc, setDoc } from 'firebase/firestore';
import { db, auth, onAuthStateChanged, updateProfile } from './firebase.js';

/* ---------- Auth-flow guard ----------
   Signing up/in fires onAuthStateChanged immediately, and the "already signed in
   -> go to the app" redirects would navigate away before the profile write finishes.
   Pages call beginAuthFlow() before signing in and redirect themselves afterwards. */
let flowActive = false;
export function beginAuthFlow() { flowActive = true; }
export function endAuthFlow() { flowActive = false; }
export function isAuthFlowActive() { return flowActive; }

/* ---------- Helpers ---------- */
export function splitName(full = '') {
  const parts = String(full).trim().split(/\s+/).filter(Boolean);
  return { firstName: parts[0] || '', lastName: parts.slice(1).join(' ') };
}


export function largePhoto(url) {
  return url ? url.replace(/=s\d+-c$/, '=s256-c') : null;
}

export function initialsFor({ firstName = '', lastName = '', email = '' } = {}) {
  const a = firstName.trim().charAt(0);
  const b = lastName.trim().charAt(0);
  const out = (a + b) || email.trim().charAt(0);
  return out ? out.toUpperCase() : '?';
}

function currentUser() {
  return new Promise(resolve => {
    let unsub;
    unsub = onAuthStateChanged(auth, user => { if (unsub) unsub(); resolve(user); });
  });
}

function buildProfile(user, data = {}) {
  const fromAuth = splitName(user.displayName || '');
  const firstName = data.firstName || fromAuth.firstName || '';
  const lastName = data.lastName || fromAuth.lastName || '';
  const email = data.email || user.email || '';
  const providers = (user.providerData || []).map(p => p.providerId);
  const isGoogle = providers.includes('google.com');
  const hasPassword = providers.includes('password');

  return {
    uid: user.uid,
    firstName,
    lastName,
    fullName: [firstName, lastName].filter(Boolean).join(' ') || email,
    email,
    photoURL: data.photoURL || largePhoto(user.photoURL) || null,
    providers,
    isGoogle,
    hasPassword,
    providerText: isGoogle && hasPassword ? 'Signed in with Google or email'
                : isGoogle ? 'Signed in with Google'
                : 'Signed in with email',
    initials: initialsFor({ firstName, lastName, email }),
  };
}

/* ---------- Create / complete the profile (call right after sign-up or Google sign-in) ----------
   Never overwrites a name or avatar the user already has. */
export async function ensureUserProfile(user, names = {}) {
  const ref = doc(db, 'users', user.uid);
  const snap = await getDoc(ref);
  const existing = snap.exists() ? snap.data() : {};
  const fromAuth = splitName(user.displayName || '');   // Google gives "First Last"

  const firstName = existing.firstName || names.firstName || fromAuth.firstName || '';
  const lastName  = existing.lastName  || names.lastName  || fromAuth.lastName  || '';
  const displayName = [firstName, lastName].filter(Boolean).join(' ');

  if (displayName && user.displayName !== displayName) {
    try { await updateProfile(user, { displayName }); }
    catch (err) { console.warn('Could not update auth profile', err); }
  }

  const now = new Date().toISOString();
  const data = {
    firstName,
    lastName,
    displayName,
    email: existing.email || user.email || '',
    photoURL: existing.photoURL || largePhoto(user.photoURL) || null,
    providers: (user.providerData || []).map(p => p.providerId),
    updatedAt: now,
  };
  if (!existing.createdAt) data.createdAt = now;

  await setDoc(ref, data, { merge: true });
  cache = null;
  return buildProfile(user, { ...existing, ...data });
}

/* ---------- Read the signed-in user's profile (cached) ---------- */
let cache = null;

export async function getUserProfile({ force = false } = {}) {
  const user = await currentUser();
  if (!user) { cache = null; return null; }
  if (!force && cache && cache.uid === user.uid) return cache;

  let data = {};
  try {
    const snap = await getDoc(doc(db, 'users', user.uid));
    if (snap.exists()) {
      data = snap.data();
    } else {

      ensureUserProfile(user).catch(err => console.warn('Profile backfill failed', err));
    }
  } catch (err) {
    console.error('Could not read profile', err);
  }

  cache = buildProfile(user, data);
  return cache;
}

/* ---------- Avatar rendering ---------- */


export function renderAvatar(el, profile) {
  if (!el || !profile) return;
  el.textContent = '';

  const showInitials = () => {
    el.textContent = '';
    const span = document.createElement('span');
    span.className = 'me-initials';
    span.textContent = profile.initials;
    el.appendChild(span);
  };

  if (!profile.photoURL) { showInitials(); return; }

  const img = new Image();
  img.className = 'me-img';
  img.alt = '';
  img.referrerPolicy = 'no-referrer';  
  img.onerror = showInitials;
  img.src = profile.photoURL;
  el.appendChild(img);
}

let navObserver = null;

export function bindNavAvatar() {
  if (navObserver) return;

  const apply = () => {
    const el = document.querySelector('.avatar:not([data-profile-bound])');
    if (!el) return;
    el.dataset.profileBound = '1';

    getUserProfile().then(profile => {
      if (!profile) return;
      el.classList.add('has-profile');
      el.title = profile.fullName;
      el.setAttribute('aria-label', `Account: ${profile.fullName}`);
      renderAvatar(el, profile);
    }).catch(err => console.error(err));
  };

  navObserver = new MutationObserver(apply);
  navObserver.observe(document.body, { childList: true, subtree: true });
  apply();
}