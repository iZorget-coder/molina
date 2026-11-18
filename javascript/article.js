const params = new URLSearchParams(window.location.search);
const title = params.get('title');
const source = params.get('source');
const published = params.get('published');
const description = params.get('description');
const url = params.get('url');
const image = params.get('image');
const category = params.get('category') || 'news';

if (title) {
  document.getElementById('article-tag').textContent =
    category.charAt(0).toUpperCase() + category.slice(1);
  document.getElementById('article-title').textContent = title;
  document.getElementById('article-meta').innerHTML =
    `<strong>${source || 'Molina Wire'}</strong> <i aria-hidden="true"></i> <time>${published || ''}</time>`;

  const body = document.getElementById('article-body');
  body.innerHTML = `<p>${description || 'No preview available for this story.'}</p>`;

  if (image) {
    const hero = document.querySelector('.article__hero');
    if (hero) {
      const img = new Image();
      img.alt = '';
      img.loading = 'lazy';
      img.style.cssText = 'width:100%;height:100%;object-fit:cover;border-radius:12px;';
      img.onload = () => {
        hero.innerHTML = '';
        hero.appendChild(img);
      };
      img.src = image;
    }
  }

  if (url) {
    const sourceP = document.getElementById('article-source');
    const link = sourceP.querySelector('a');
    link.href = url;
    sourceP.hidden = false;
  }
} else {
  document.getElementById('article-title').textContent = 'Article not found';
  document.getElementById('article-body').innerHTML =
    '<p>This story could not be loaded. Return to the <a href="/pages/app.html#news">news feed</a>.</p>';
}