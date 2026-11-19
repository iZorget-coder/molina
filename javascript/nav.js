(function () {
  const container = document.getElementById('nav-container');
  if (!container) return;

 
  const links = [
    { label: 'Dashboard', href: '/pages/app.html#dashboard', key: 'dashboard' },
    { label: 'Watchlist', href: '/pages/app.html#watchlist', key: 'watchlist' },
    { label: 'Markets',   href: '/pages/app.html#markets',   key: 'markets'   },
    { label: 'News',      href: '/pages/app.html#news',      key: 'news'      },
    { label: 'Settings',  href: '/pages/app.html#settings',  key: 'settings'  },
  ];


  const bottomLinks = [
    { label: 'Home',      href: '/pages/app.html#dashboard', key: 'dashboard',
      icon: '<path d="M3 10.5 12 3l9 7.5V20a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>' },
    { label: 'Watchlist', href: '/pages/app.html#watchlist', key: 'watchlist',
      icon: '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>' },
    { label: 'News',      href: '/pages/app.html#news',      key: 'news',
      icon: '<path d="M4 4h13a1 1 0 0 1 1 1v14a2 2 0 0 0 2 2H6a2 2 0 0 1-2-2z"/><path d="M18 8h2a1 1 0 0 1 1 1v10a2 2 0 0 1-2 2"/><line x1="8" y1="8" x2="14" y2="8"/><line x1="8" y1="12" x2="14" y2="12"/><line x1="8" y1="16" x2="12" y2="16"/>' },
    { label: 'Settings',  href: '/pages/app.html#settings',  key: 'settings',
      icon: '<line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/><line x1="15" y1="3" x2="15" y2="9"/><line x1="8" y1="9" x2="8" y2="15"/><line x1="16" y1="15" x2="16" y2="21"/>' },
  ];

  const navLinks = links
    .map(l => `<a href="${l.href}" class="nav__link" data-tab="${l.key}">${l.label}</a>`)
    .join('');

  const bottomNav = bottomLinks
    .map(l => `
      <a href="${l.href}" class="bottom-nav__link" data-tab="${l.key}">
        <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${l.icon}</svg>
        <span>${l.label}</span>
      </a>`)
    .join('');

  if (!document.getElementById('bottom-nav-styles')) {
    const style = document.createElement('style');
    style.id = 'bottom-nav-styles';
    style.textContent = `
      .bottom-nav { display: none; }
      @media (max-width: 800px) {
        .bottom-nav {
          position: fixed;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 50;
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          padding: 10px 8px calc(10px + env(safe-area-inset-bottom));
          background: #0b1020;
          border-top: 1px solid #1b2540;
        }
        .bottom-nav__link {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 6px;
          padding: 6px 0;
          color: #aab3c8;
          font-family: "Space Mono", monospace;
          font-size: 13px;
          font-weight: 500;
          text-decoration: none;
          -webkit-tap-highlight-color: transparent;
          transition: color 0.15s;
        }
        .bottom-nav__link.is-active { color: #818cf8; }
      }
    `;
    document.head.appendChild(style);
  }

  container.innerHTML = `
    <header class="topbar">
      <div class="topbar__left">
        <a href="index.html" class="brand">
          <span class="brand__logo">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 12 7 12 10 4 14 20 17 12 21 12"/></svg>
          </span>
          <span class="brand__name">Molina</span>
        </a>
        <nav class="nav">${navLinks}</nav>
      </div>

      <div class="topbar__right">
        <label class="search">
          <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.5" y2="16.5"/></svg>
          <input type="search" placeholder="Search markets">
        </label>
        <button class="avatar" aria-label="Account">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 4-6 8-6s8 2 8 6"/></svg>
        </button>
      </div>
    </header>

    <nav class="bottom-nav" aria-label="Primary">${bottomNav}</nav>
  `;


  function currentKey() {
    return (location.hash || '#dashboard').slice(1).split(/[/?]/)[0] || 'dashboard';
  }
  function sync() {
    const key = currentKey();
    document.body.dataset.view = key;
    container.querySelectorAll('[data-tab]').forEach(a => {
      a.classList.toggle('is-active', a.dataset.tab === key);
    });
  }
  window.addEventListener('hashchange', sync);
  sync();


  const mobileMq = window.matchMedia('(max-width: 800px)');
  document.addEventListener('click', (e) => {
    const head = e.target.closest('.setup .card__head--bordered');
    if (!head || !mobileMq.matches) return;
    head.closest('.setup').classList.toggle('is-open');
  });
})();