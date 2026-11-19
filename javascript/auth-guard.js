import { auth, onAuthStateChanged } from './firebase.js';

const DESTINATION = import.meta.env.BASE_URL + 'pages/app.html';

onAuthStateChanged(auth, (user) => {
  if (user && !window.location.pathname.endsWith('/pages/app.html')) {
    window.location.replace(DESTINATION);
  }
});