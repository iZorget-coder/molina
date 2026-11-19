import { auth, signOut } from './firebase.js';
import { getUserProfile, renderAvatar } from './profile.js';

// Runs every time the settings view is injected.
export async function bindSettings() {
  const logoutBtn = document.querySelector('[data-logout]');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      logoutBtn.disabled = true;
      try {
        await signOut(auth);
      } catch (err) {
        console.error('Could not log out', err);
        logoutBtn.disabled = false;
        return;
      }
      window.location.replace('/index.html');
    });
  }

  const profile = await getUserProfile().catch(() => null);
  if (!profile) return;

  document.querySelectorAll('[data-user]').forEach(el => {
    el.textContent = profile[el.dataset.user] || '—';
  });

  renderAvatar(document.querySelector('[data-user-avatar]'), profile);

  // Google-only accounts have no password to change.
  if (!profile.hasPassword) {
    const value = document.querySelector('[data-password-value]');
    const btn = document.querySelector('[data-password-change]');
    if (value) {
      value.textContent = 'Managed by Google';
      value.classList.remove('settings-row__value--dots');
      value.classList.add('settings-row__value--muted');
      value.removeAttribute('aria-label');
    }
    if (btn) btn.hidden = true;
  }
}