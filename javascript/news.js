import { doc, getDoc, setDoc } from 'firebase/firestore';
import { db, auth, onAuthStateChanged } from './firebase.js';

const API_KEY = import.meta.env.VITE_CURRENTS_API_KEY;
const ENDPOINT = 'https://api.currentsapi.services/v1/latest-news?language=en';

let cachedArticles = null;
let authReady = null;

function whenAuthReady() {
  if (authReady) return authReady;
  authReady = new Promise(resolve => {
    const unsub = onAuthStateChanged(auth, user => {
      unsub();
      resolve(user);
    });
  });
  return authReady;
}

function todayKey() {
  return new Date().toISOString().slice(0, 10);
}

function mapCategory(categories) {
  const flat = (categories || []).join(' ').toLowerCase();
  if (flat.includes('technology') || flat.includes('science')) return 'tech';
  if (flat.includes('business') || flat.includes('finance') || flat.includes('economy')) return 'finance';
  return 'markets';
}

function timeAgo(dateString) {
  if (!dateString) return '';
  const diff = Date.now() - new Date(dateString).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} h ago`;
  return `${Math.floor(hrs / 24)} d ago`;
}

function storyMarkup(article, siblings) {
  const category = mapCategory(article.category);
  const published = timeAgo(article.published);
  const payload = encodeURIComponent(JSON.stringify({
    title: article.title || '',
    source: 'Currents',
    published,
    description: article.description || '',
    url: article.url || '',
    image: article.image || '',
    category,
    siblings: (siblings || []).map(s => ({
      title: s.title || '',
      description: s.description || '',
      url: s.url || '',
      image: s.image || '',
      published: timeAgo(s.published),
    })),
  }));

  const fallbackSvg = `<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/></svg>`;
  const thumb = article.image
    ? `<img src="${article.image}" alt="" loading="lazy"
           onerror="this.style.display='none';this.nextElementSibling.style.display='';" />
       <span style="display:none">${fallbackSvg}</span>`
    : fallbackSvg;

  return `
    <li>
      <a class="story" href="#article" data-article="${payload}" data-category="${category}">
        <span class="story__thumb" aria-hidden="true">${thumb}</span>
        <span class="story__body">
          <span class="story__tag">${category.charAt(0).toUpperCase() + category.slice(1)}</span>
          <span class="story__title">${article.title || 'Untitled'}</span>
          <span class="story__meta">Currents <i></i> <time>${published}</time></span>
        </span>
      </a>
    </li>
  `;
}

function renderArticles(articles) {
  const finance = articles.filter(a => mapCategory(a.category) === 'finance').slice(0, 5);
  const tech = articles.filter(a => mapCategory(a.category) === 'tech').slice(0, 5);

  const pool = articles.filter(a => !finance.includes(a) && !tech.includes(a));
  while (finance.length < 5 && pool.length) finance.push(pool.shift());
  while (tech.length < 5 && pool.length) tech.push(pool.shift());

  const financeList = document.getElementById('news-list-finance');
  const techList = document.getElementById('news-list-tech');

  if (financeList) {
    financeList.innerHTML = finance.length
      ? finance.map(a => storyMarkup(a, finance)).join('')
      : '<li><p class="view-empty">No finance stories available.</p></li>';
  }
  if (techList) {
    techList.innerHTML = tech.length
      ? tech.map(a => storyMarkup(a, tech)).join('')
      : '<li><p class="view-empty">No tech stories available.</p></li>';
  }

  const financeCount = document.getElementById('count-finance');
  const techCount = document.getElementById('count-tech');
  if (financeCount) financeCount.textContent = `${finance.length} stories`;
  if (techCount) techCount.textContent = `${tech.length} stories`;
}

function setLoading() {
  ['finance', 'tech'].forEach(key => {
    const list = document.getElementById(`news-list-${key}`);
    if (list) list.innerHTML = '<li><p class="view-loading">Loading stories…</p></li>';
    const count = document.getElementById(`count-${key}`);
    if (count) count.textContent = '…';
  });
}

function setError() {
  ['finance', 'tech'].forEach(key => {
    const list = document.getElementById(`news-list-${key}`);
    if (list) list.innerHTML = '<li><p class="view-error">News unavailable. Try again later.</p></li>';
    const count = document.getElementById(`count-${key}`);
    if (count) count.textContent = '0 stories';
  });
}

async function fetchFromCurrents() {
  const res = await fetch(ENDPOINT, { headers: { Authorization: API_KEY } });
  if (!res.ok) throw new Error(`Currents error ${res.status}`);
  const data = await res.json();
  return (data.news || []).map(a => ({
    title: a.title || '',
    description: a.description || '',
    url: a.url || '',
    image: a.image || '',
    published: a.published || '',
    category: a.category || [],
  }));
}

export async function loadNews() {
  if (cachedArticles) {
    renderArticles(cachedArticles);
    return;
  }

  const user = await whenAuthReady();

  if (!user) {
    setLoading();
    try {
      const articles = await fetchFromCurrents();
      cachedArticles = articles;
      renderArticles(articles);
    } catch (err) {
      console.error(err);
      setError();
    }
    return;
  }

  setLoading();

  try {
    const ref = doc(db, 'newsCache', user.uid, 'days', todayKey());
    const snap = await getDoc(ref);

    if (snap.exists()) {
      cachedArticles = snap.data().articles || [];
      renderArticles(cachedArticles);
      return;
    }

    const articles = await fetchFromCurrents();
    await setDoc(ref, {
      date: todayKey(),
      articles,
      fetchedAt: new Date().toISOString(),
    });
    cachedArticles = articles;
    renderArticles(articles);
  } catch (err) {
    console.error(err);
    setError();
  }
}

function dashboardItemMarkup(article, siblings) {
  const published = timeAgo(article.published);
  const title = article.title || 'Untitled';
  const source = 'Currents';
  const payload = encodeURIComponent(JSON.stringify({
    title: article.title || '',
    source,
    published,
    description: article.description || '',
    url: article.url || '',
    image: article.image || '',
    category: mapCategory(article.category),
    siblings: (siblings || []).map(s => ({
      title: s.title || '',
      description: s.description || '',
      url: s.url || '',
      image: s.image || '',
      published: timeAgo(s.published),
    })),
  }));

  return `
    <li class="news__item">
      <a href="#article" data-article="${payload}">
        <h3>${title}</h3>
        <div class="news__meta"><span>${source}</span><span>${published}</span></div>
      </a>
    </li>
  `;
}

export async function loadDashboardNews() {
  const list = document.getElementById('dashboard-news-list');
  if (!list) return;

  if (cachedArticles) {
    list.innerHTML = cachedArticles.slice(0, 5)
      .map(a => dashboardItemMarkup(a, cachedArticles))
      .join('');
    return;
  }

  const user = await whenAuthReady();

  list.innerHTML = '<li class="news__item"><h3>Loading…</h3></li>';

  try {
    let articles;

    if (user) {
      const ref = doc(db, 'newsCache', user.uid, 'days', todayKey());
      const snap = await getDoc(ref);
      if (snap.exists()) {
        articles = snap.data().articles || [];
      } else {
        articles = await fetchFromCurrents();
        await setDoc(ref, {
          date: todayKey(),
          articles,
          fetchedAt: new Date().toISOString(),
        });
      }
    } else {
      articles = await fetchFromCurrents();
    }

    cachedArticles = articles;
    list.innerHTML = articles.slice(0, 5)
      .map(a => dashboardItemMarkup(a, articles))
      .join('');
  } catch (err) {
    console.error(err);
    list.innerHTML = '<li class="news__item"><h3>News unavailable.</h3></li>';
  }
}

let filtersBound = false;

export function bindFilters() {
  if (filtersBound) return;
  filtersBound = true;

  const buttons = document.querySelectorAll('.news-filter');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => {
        b.classList.remove('is-active');
        b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('is-active');
      btn.setAttribute('aria-selected', 'true');

      const filter = btn.dataset.filter;
      document.querySelectorAll('.story').forEach(story => {
        story.closest('li').hidden = filter !== 'all' && story.dataset.category !== filter;
      });
    });
  });
}