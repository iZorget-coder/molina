(function () {
  const container = document.getElementById('nav-container');
  if (!container) return;

  const links = [
    { label: 'Dashboard', href: '#dashboard', key: 'dashboard' },
    { label: 'Watchlist', href: '#watchlist', key: 'watchlist' },
    { label: 'Markets',   href: '#markets',   key: 'markets'   },
    { label: 'News',      href: '#news',      key: 'news'      },
    { label: 'Settings',  href: '#settings',  key: 'settings'  },
  ];

  const navLinks = links
    .map(l => `<a href="${l.href}" class="nav__link" data-tab="${l.key}">${l.label}</a>`)
    .join('');

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
  `;
})();