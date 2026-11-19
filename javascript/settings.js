import { auth, signOut } from './firebase.js';

import {
  EmailAuthProvider,
  GoogleAuthProvider,
  reauthenticateWithCredential,
  reauthenticateWithPopup,
  deleteUser
} from 'firebase/auth';

import {
  doc,
  deleteDoc
} from 'firebase/firestore';

import { db } from './firebase.js';

import {
  getUserProfile,
  renderAvatar
} from './profile.js';

import { renderSettingsAiUsage } from './ai-usage.js';


const SILENT_DELETE_ERRORS = new Set([
  'auth/popup-closed-by-user',
  'auth/cancelled-popup-request',
  'auth/user-cancelled'
]);

 
export async function bindSettings() {

  renderSettingsAiUsage();



  const logoutBtn = document.querySelector('[data-logout]');

  if (logoutBtn) {

    logoutBtn.addEventListener('click', async () => {

      const user = auth.currentUser;

      if (!user) {
        window.location.replace(import.meta.env.BASE_URL + 'index.html');
        return;
      }

      logoutBtn.disabled = true;

      try {
        await signOut(auth);
        window.location.replace(import.meta.env.BASE_URL + 'index.html');
      } catch (err) {
        console.error('Could not log out:', err);
        alert('Could not log out. Please try again.');
        logoutBtn.disabled = false;
      }

    });

  }

 
  const deleteBtn = document.querySelector('[data-delete-account]');

  if (deleteBtn) {

    deleteBtn.addEventListener('click', async () => {

      const user = auth.currentUser;

      if (!user) {
        window.location.replace(import.meta.env.BASE_URL + 'index.html');
        return;
      }

      const originalText = deleteBtn.textContent;

      const resetButton = () => {
        deleteBtn.disabled = false;
        deleteBtn.textContent = originalText;
      };

      deleteBtn.disabled = true;

      try {

        const providers = (user.providerData || [])
          .map(provider => provider.providerId);

        if (providers.includes('google.com')) {
 
          await reauthenticateWithPopup(user, new GoogleAuthProvider());

        } else if (providers.includes('password')) {

          const password = window.prompt(
            'Enter your current password to delete your account:'
          );

          
          if (password === null) {
            resetButton();
            return;
          }

          if (!password.trim()) {
            throw new Error('A password is required.');
          }

          const credential = EmailAuthProvider.credential(user.email, password);

          await reauthenticateWithCredential(user, credential);

        } else {

          throw new Error(
            'Your sign-in provider does not currently support account deletion.'
          );

        }

        
        deleteBtn.textContent = 'Deleting...';

 
        await deleteDoc(doc(db, 'users', user.uid));
        await deleteUser(user);

        
        window.location.replace(import.meta.env.BASE_URL + 'index.html');

      } catch (err) {

        console.error('Could not delete account:', err);

        resetButton();

       
        if (SILENT_DELETE_ERRORS.has(err.code)) return;

        let message = 'Could not delete your account. Please try again.';

        if (err.code === 'auth/wrong-password' ||
            err.code === 'auth/invalid-credential') {

          message = 'Incorrect password. Please try again.';

        } else if (err.code === 'auth/popup-blocked') {

          message = 'Your browser blocked the Google sign-in window. Allow pop-ups for this site and try again.';

        } else if (err.code === 'auth/requires-recent-login') {

          message = 'Your session has expired. Please sign in again and retry.';

        } else if (err.code === 'auth/network-request-failed') {

          message = 'Network error. Check your internet connection.';

        } else if (err.code === 'auth/too-many-requests') {

          message = 'Too many attempts. Please try again later.';

        } else if (err.code === 'permission-denied') {

          message = 'Could not delete your profile data. Please try again later.';

        } else if (err.message) {

          message = err.message;

        }

        alert(message);

      }

    });

  }

  // --------------------------------------------------
  // LOAD USER PROFILE
  // --------------------------------------------------

  const profile = await getUserProfile().catch(err => {
    console.error('Could not load user profile:', err);
    return null;
  });

  if (!profile) return;

  
  document.querySelectorAll('[data-user]').forEach(el => {
    el.textContent = profile[el.dataset.user] || '—';
  });

  
  renderAvatar(
    document.querySelector('[data-user-avatar]'),
    profile
  );

   
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