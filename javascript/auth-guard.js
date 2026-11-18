import { auth, onAuthStateChanged } from './firebase.js';

const DESTINATION = '/pages/app.html';

onAuthStateChanged(auth, (user) => {
  if (user && !window.location.pathname.endsWith('/pages/app.html')) {
    window.location.replace(DESTINATION);
  }
});