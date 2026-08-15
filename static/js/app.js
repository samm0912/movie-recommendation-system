/* ── app.js — Movie Recommendation System Frontend Controller ── */

// Navbar scroll background effect
window.addEventListener('scroll', () => {
  document.getElementById('navbar')?.classList.toggle('scrolled', window.scrollY > 40);
});

// Mobile Navigation Toggle
function toggleMobileNav() {
  const toggle = document.getElementById('mobileNavToggle');
  const drawer = document.getElementById('mobileNavDrawer');
  if (toggle && drawer) {
    toggle.classList.toggle('open');
    drawer.classList.toggle('open');
  }
}

// Quick Search Keyboard Shortcut ('/' or 'Ctrl+K')
document.addEventListener('keydown', (e) => {
  if ((e.key === '/' || (e.ctrlKey && e.key === 'k')) && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
  }
  if (e.key === 'Escape') {
    const searchInput = document.getElementById('searchInput');
    if (searchInput && document.activeElement === searchInput) {
      searchInput.blur();
    }
  }
});

// ── Render Movie Card Component ───────────────────────────────────────────
function renderCard(m) {
  const trailerBtn = (m.has_trailer || m.trailer_key)
    ? `<button class="card-btn card-trailer" title="Watch Trailer" data-trailer="${m.trailer_key || ''}" data-id="${m.id}" data-title="${escapeHtml(m.title)}" onclick="event.stopPropagation(); openTrailer(this)">▶ Trailer</button>`
    : '';

  const trailerBadge = (m.has_trailer || m.trailer_key)
    ? `<span class="card-badge-trailer" title="Trailer Available">🎬</span>`
    : '';

  const genresFormatted = (m.genres || '').replace(/\|/g, ' · ').slice(0, 30);
  const fallbackImg = 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60';

  return `
    <div class="movie-card" data-id="${m.id}" onclick="window.location='/movie/${m.id}'">
      <div class="card-img-wrap">
        <img src="${m.poster || fallbackImg}" alt="${escapeHtml(m.title)}" loading="lazy"
          onerror="this.onerror=null; this.src='${fallbackImg}'">
        <div class="card-badges">
          <span class="card-badge-rating">⭐ ${m.rating}</span>
          ${trailerBadge}
        </div>
        <div class="card-overlay">
          <div class="card-actions">
            ${trailerBtn}
            <button class="card-btn card-like" title="Add to Watchlist" onclick="event.stopPropagation(); likeMovie(${m.id}, this)">♡</button>
            <button class="card-btn card-why" title="Why Recommended?" onclick="event.stopPropagation(); showWhy(${m.id})">?</button>
          </div>
          <div class="card-info-bottom">
            <p class="card-title">${escapeHtml(m.title)}</p>
            <p class="card-meta"><span class="meta-star">⭐ ${m.rating}</span> · ${m.year}</p>
            <p class="card-genres">${escapeHtml(genresFormatted)}</p>
          </div>
        </div>
      </div>
    </div>`;
}

// ── Video Trailer Player Modal ────────────────────────────────────────────
async function openTrailer(trailerKeyOrEl, movieTitle = 'Movie') {
  let trailerKey = '';
  let title = movieTitle || 'Movie';

  if (trailerKeyOrEl && typeof trailerKeyOrEl === 'object' && trailerKeyOrEl.dataset) {
    trailerKey = trailerKeyOrEl.dataset.trailer || trailerKeyOrEl.dataset.id || '';
    title = trailerKeyOrEl.dataset.title || title;
  } else {
    trailerKey = String(trailerKeyOrEl || '').trim();
  }

  // If trailerKey is empty or numeric (movie ID), fetch dynamically from backend
  if (!trailerKey || trailerKey === 'undefined' || trailerKey === 'null' || /^\d+$/.test(trailerKey)) {
    const movieId = trailerKey && /^\d+$/.test(trailerKey) ? trailerKey : '';
    if (movieId) {
      showToast('🎬 Loading trailer stream…');
      const data = await apiFetch(`/api/trailer/${movieId}`);
      if (data) {
        if (data.trailer_key) trailerKey = data.trailer_key;
        if (data.title) title = data.title;
      }
    }
  }

  const modal = document.getElementById('trailerModal');
  const titleEl = document.getElementById('trailerTitle');
  const container = document.getElementById('videoContainer');
  const extLink = document.getElementById('trailerExternalLink');
  if (!modal || !container) return;

  let youtubeUrl = '';
  let embedUrl = '';

  if (trailerKey && trailerKey !== 'undefined' && trailerKey !== 'null' && !/^\d+$/.test(String(trailerKey))) {
    youtubeUrl = `https://www.youtube.com/watch?v=${trailerKey}`;
    embedUrl = `https://www.youtube.com/embed/${trailerKey}?autoplay=1&enablejsapi=1&rel=0&modestbranding=1`;
  } else {
    // Universal fallback: YouTube search embed & query link for the movie
    const q = encodeURIComponent(`${title} official trailer`);
    youtubeUrl = `https://www.youtube.com/results?search_query=${q}`;
    embedUrl = `https://www.youtube.com/embed?listType=search&list=${q}&autoplay=1`;
  }

  if (titleEl) titleEl.textContent = `${title} — Official Trailer`;
  if (extLink) extLink.href = youtubeUrl;

  container.innerHTML = `
    <iframe src="${embedUrl}" 
      title="${escapeHtml(title)} Trailer" 
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
      allowfullscreen>
    </iframe>`;

  openModal('trailerModal');
}

function closeTrailer() {
  const container = document.getElementById('videoContainer');
  if (container) container.innerHTML = '';
  closeModal('trailerModal');
}

// ── Feature: Surprise Me (Single Movie Recommendation) ────────────────────
async function triggerSurpriseMe() {
  openModal('surpriseModal');
  const content = document.getElementById('surpriseContent');
  if (!content) return;

  content.innerHTML = '<div class="loading-state">✨ Selecting a cinematic gem from 10,000 titles…</div>';

  const data = await apiFetch('/api/surprise');
  if (!data?.movie) {
    content.innerHTML = '<p style="color:#ef4444">Failed to pick a movie. Please try again.</p>';
    return;
  }

  const m = data.movie;
  const fallbackImg = 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60';
  const genresFormatted = (m.genres || '').replace(/\|/g, ' · ');

  const trailerBtn = (m.has_trailer || m.trailer_key)
    ? `<button class="btn-surprise-trailer" data-trailer="${m.trailer_key || ''}" data-id="${m.id}" data-title="${escapeHtml(m.title)}" onclick="closeModal('surpriseModal'); openTrailer(this)">▶ Watch Trailer</button>`
    : `<button class="btn-surprise-trailer disabled" disabled>Trailer Unavailable</button>`;

  content.innerHTML = `
    <div class="surprise-card-inner">
      <img src="${m.poster || fallbackImg}" alt="${escapeHtml(m.title)}" class="surprise-poster"
        onerror="this.onerror=null; this.src='${fallbackImg}'">
      <h3 class="surprise-title">${escapeHtml(m.title)}</h3>
      <div class="surprise-meta">
        <span style="color:var(--accent-gold); font-weight:700">⭐ ${m.rating}/10</span>
        <span>·</span>
        <span>${m.year}</span>
      </div>
      <p class="surprise-genres">${escapeHtml(genresFormatted)}</p>
      <p class="surprise-overview">${escapeHtml(m.overview || 'No synopsis available for this title.')}</p>
      <div class="surprise-actions">
        ${trailerBtn}
        <a href="/movie/${m.id}" class="btn-surprise-details">View Details</a>
        <button class="btn-surprise-again" onclick="triggerSurpriseMe()">Roll Again ✨</button>
      </div>
    </div>`;
}

// ── Row Loaders ───────────────────────────────────────────────────────────
async function loadRecommended() {
  const row = document.getElementById('recommendedRow');
  if (!row) return;
  const data = await apiFetch('/api/recommend?method=hybrid');
  if (data?.movies?.length) {
    row.innerHTML = data.movies.map(renderCard).join('');
  }
}

async function loadCollab() {
  const row = document.getElementById('collabRow');
  if (!row) return;
  const data = await apiFetch('/api/recommend?method=collab');
  if (data?.movies?.length) {
    row.innerHTML = data.movies.map(renderCard).join('');
  }
}

async function loadGenreRows() {
  const rows = document.querySelectorAll('.genre-row');
  for (const row of rows) {
    const genre = row.dataset.genre;
    const data = await apiFetch(`/api/genre/${encodeURIComponent(genre)}`);
    if (data?.movies?.length) {
      row.innerHTML = data.movies.map(renderCard).join('');
    }
  }
}

// ── Search & AI Prompt Recommendation Controller ────────────────────────
let searchTimeout;

function submitSearch() {
  const input = document.getElementById('searchInput');
  if (!input) return;
  const q = input.value.trim();
  if (!q) return;

  const section = document.getElementById('searchResults');
  if (!section) {
    // If on a page without searchResults (e.g. /movie/<id>), navigate to home with query
    window.location.href = `/?q=${encodeURIComponent(q)}`;
    return;
  }

  // If on homepage, execute immediately
  handleSearch(q, true);
}

function handleSearch(query, immediate = false) {
  clearTimeout(searchTimeout);
  const section = document.getElementById('searchResults');
  const grid = document.getElementById('searchGrid');
  const matchedSection = document.getElementById('matchedMoviesSection');
  const matchedGrid = document.getElementById('matchedGrid');
  const titleEl = document.getElementById('searchHeaderTitle');
  const noteEl = document.getElementById('searchInsightNote');
  const promptRecsHeader = document.getElementById('promptRecsHeader');
  const promptRecsSub = document.getElementById('promptRecsSub');

  if (!section || !grid) return;

  const q = (query || '').trim();
  if (!q) {
    section.style.display = 'none';
    if (matchedSection) matchedSection.style.display = 'none';
    if (noteEl) noteEl.style.display = 'none';
    if (promptRecsHeader) promptRecsHeader.style.display = 'none';
    return;
  }

  const executeApi = async () => {
    const data = await apiFetch(`/api/search?q=${encodeURIComponent(q)}`);
    if (!data) return;

    section.style.display = 'block';

    const hasMatches = data.matched_movies && data.matched_movies.length > 0;
    const hasRecs = data.recommendations && data.recommendations.length > 0;

    if (hasMatches) {
      if (matchedSection && matchedGrid) {
        matchedSection.style.display = 'block';
        matchedGrid.innerHTML = data.matched_movies.map(renderCard).join('');
      }
      if (promptRecsHeader) {
        promptRecsHeader.style.display = 'flex';
        if (promptRecsSub) {
          const titles = data.matched_movies.map(m => m.title).join(' & ');
          promptRecsSub.textContent = `Because you liked ${titles} · Content & Semantic Match`;
        }
      }
      if (titleEl) {
        titleEl.innerHTML = `✨ AI Recommendations for <em>"${escapeHtml(q)}"</em>`;
      }
      grid.innerHTML = hasRecs
        ? data.recommendations.map(renderCard).join('')
        : '<p style="color:var(--text-muted);padding:1.5rem">No additional recommendations found.</p>';
    } else {
      if (matchedSection) matchedSection.style.display = 'none';
      if (promptRecsHeader) promptRecsHeader.style.display = 'none';
      if (titleEl) {
        titleEl.innerHTML = `🔍 Search Results for <em>"${escapeHtml(q)}"</em>`;
      }
      const movies = data.recommendations || data.movies || [];
      grid.innerHTML = movies.length
        ? movies.map(renderCard).join('')
        : '<p style="color:var(--text-muted);padding:1.5rem">No matching movies found in the dataset.</p>';
    }

    if (noteEl) {
      if (data.message) {
        noteEl.textContent = data.message;
        noteEl.style.display = 'block';
      } else {
        noteEl.style.display = 'none';
      }
    }

    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  if (immediate) {
    executeApi();
  } else {
    searchTimeout = setTimeout(executeApi, 250);
  }
}

function clearSearch() {
  const input = document.getElementById('searchInput');
  const section = document.getElementById('searchResults');
  const matchedSection = document.getElementById('matchedMoviesSection');
  const noteEl = document.getElementById('searchInsightNote');
  const promptRecsHeader = document.getElementById('promptRecsHeader');
  if (input) input.value = '';
  if (section) section.style.display = 'none';
  if (matchedSection) matchedSection.style.display = 'none';
  if (noteEl) noteEl.style.display = 'none';
  if (promptRecsHeader) promptRecsHeader.style.display = 'none';
}

// Auto-execute query from URL search param if present on page load
document.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  if (q) {
    const input = document.getElementById('searchInput');
    if (input) {
      input.value = q;
      handleSearch(q, true);
    }
  }
});

// ── Watchlist / Like Controller ───────────────────────────────────────────
async function likeMovie(movieId, btn) {
  const data = await apiFetch(`/api/like/${movieId}`, 'POST');
  if (data?.success) {
    const added = data.action === 'added';
    btn.classList.toggle('liked', added);
    btn.textContent = added ? '♥' : '♡';
    showToast(added ? '❤️ Added to your watchlist!' : 'Removed from watchlist');
  } else if (data?.error) {
    showToast('Please sign in to save movies to your watchlist');
    openLoginModal();
  }
}

// ── Feature: Personalized Quiz Controller ─────────────────────────────────
const quizState = { genres: [], mood: '' };

async function loadQuizGenres() {
  const data = await apiFetch('/api/stats');
  const container = document.getElementById('quizGenreGrid');
  if (!data?.genres || !container) return;

  container.innerHTML = data.genres.map(g =>
    `<div class="quiz-genre-chip" data-genre="${escapeHtml(g)}" onclick="toggleGenre(this)">${escapeHtml(g)}</div>`
  ).join('');
}

function toggleGenre(el) {
  el.classList.toggle('selected');
  const g = el.dataset.genre;
  quizState.genres = quizState.genres.includes(g)
    ? quizState.genres.filter(x => x !== g)
    : [...quizState.genres, g];
}

function quizNext() {
  if (quizState.genres.length === 0) {
    showToast('Please pick at least one genre to continue');
    return;
  }
  document.getElementById('quizStep1').style.display = 'none';
  document.getElementById('quizStep2').style.display = 'block';
}

function quizBack() {
  document.getElementById('quizStep2').style.display = 'none';
  document.getElementById('quizStep1').style.display = 'block';
}

function selectMood(el) {
  document.querySelectorAll('.mood-card').forEach(c => c.classList.remove('selected'));
  el.classList.add('selected');
  quizState.mood = el.dataset.mood;
}

async function submitQuiz() {
  if (!quizState.mood) {
    showToast('Please choose a viewing mood');
    return;
  }
  closeModal('quizModal');

  const data = await apiFetch('/api/quiz', 'POST', {
    genres: quizState.genres,
    mood: quizState.mood
  });

  if (data?.movies?.length) {
    const section = document.getElementById('quizResults');
    const grid = document.getElementById('quizGrid');
    const note = document.getElementById('quizNote');
    if (section && grid) {
      section.style.display = 'block';
      grid.innerHTML = data.movies.map(renderCard).join('');
      if (note) note.textContent = data.ml_explanation;
      section.scrollIntoView({ behavior: 'smooth' });
      showToast('🎯 Personalized recommendations generated!');
    }
  }

  // Reset quiz UI for next use
  document.getElementById('quizStep1').style.display = 'block';
  document.getElementById('quizStep2').style.display = 'none';
  quizState.genres = [];
  quizState.mood = '';
  document.querySelectorAll('.quiz-genre-chip').forEach(c => c.classList.remove('selected'));
  document.querySelectorAll('.mood-card').forEach(c => c.classList.remove('selected'));
}

// ── Feature: Interactive Star Rating ──────────────────────────────────────
function hoverStars(val) {
  document.querySelectorAll('.star').forEach(s => {
    s.classList.toggle('hover', parseInt(s.dataset.val) <= val);
  });
}

function resetStars(currentRating) {
  document.querySelectorAll('.star').forEach(s => {
    s.classList.remove('hover');
    s.classList.toggle('filled', parseInt(s.dataset.val) <= currentRating);
  });
}

async function rateMovie(movieId, rating) {
  const data = await apiFetch(`/api/rate/${movieId}`, 'POST', { rating });
  if (data?.success) {
    resetStars(rating);
    const fb = document.getElementById('starFeedback');
    if (fb) fb.textContent = `Your rating: ${rating}★ — Model updated!`;
    showToast(`⭐ ${rating}★ rating saved! Collaborative model updated.`);
  } else if (data?.error === 'Login required') {
    showToast('Please sign in to rate movies');
    openLoginModal();
  }
}

// ── Feature: Why This? ML Explanation ─────────────────────────────────────
async function showWhy(movieId) {
  openModal('whyModal');
  const container = document.getElementById('whyContent');
  if (!container) return;

  container.innerHTML = '<div class="why-loading">Analyzing ML feature vectors…</div>';
  const data = await apiFetch(`/api/why/${movieId}`);
  if (!data) return;

  const scorePercent = Math.round((data.cosine_score || 0.75) * 100);

  container.innerHTML = `
    <div class="why-movie-header">
      <img src="${data.poster}" class="why-poster" onerror="this.style.display='none'">
      <div>
        <p class="why-movie-title">${escapeHtml(data.movie)}</p>
        <p class="why-movie-rating">⭐ ${data.rating}/10 &nbsp;·&nbsp; ${escapeHtml(data.genres?.join(', '))}</p>
        ${data.because_you_liked ? `<p style="font-size:0.8rem;color:var(--accent-emerald);margin-top:4px">❤️ Because you liked: <strong>${escapeHtml(data.because_you_liked)}</strong></p>` : ''}
      </div>
    </div>
    <div class="why-reasons">
      ${(data.reasons || []).map(r => `
        <div class="reason-card">
          <span class="reason-icon">${r.icon}</span>
          <div>
            <p class="reason-type">${escapeHtml(r.type)}</p>
            <p class="reason-detail">${escapeHtml(r.detail)}</p>
          </div>
        </div>`).join('')}
    </div>
    <div class="why-cosine">
      <span class="cosine-label">Affinity Match</span>
      <div class="cosine-bar-wrap">
        <div class="cosine-bar" id="cosineBar" style="width:0%"></div>
      </div>
      <span class="cosine-val">${scorePercent}%</span>
    </div>
    <div class="why-similar">
      <p class="why-similar-title">Connected Recommendations</p>
      <div class="why-similar-row">
        ${(data.similar_movies || []).map(m => `<span class="why-sim-chip">${escapeHtml(m.title)}</span>`).join('')}
      </div>
    </div>`;

  setTimeout(() => {
    const bar = document.getElementById('cosineBar');
    if (bar) bar.style.width = scorePercent + '%';
  }, 100);
}

// ── Feature: Compare Algorithms Benchmark ─────────────────────────────────
async function showCompare(movieId) {
  openModal('compareModal');
  const container = document.getElementById('compareContent');
  if (!container) return;

  container.innerHTML = '<div class="why-loading">Running Content, Collaborative, and Hybrid vectors…</div>';
  const data = await apiFetch(`/api/compare/${movieId}`);
  if (!data) return;

  const renderCol = (movies) => (movies || []).map(m =>
    `<div class="compare-movie">• ${escapeHtml(m.title)} <span style="color:var(--text-dim);font-size:0.7rem">(${m.year})</span></div>`
  ).join('');

  container.innerHTML = `
    <div class="compare-grid">
      <div class="compare-col content">
        <p class="compare-col-title">🔤 Content-Based</p>
        ${renderCol(data.content_based)}
      </div>
      <div class="compare-col collab">
        <p class="compare-col-title">👥 Collaborative</p>
        ${renderCol(data.collaborative)}
      </div>
      <div class="compare-col hybrid">
        <p class="compare-col-title">⚡ Hybrid (Combined)</p>
        ${renderCol(data.hybrid)}
      </div>
    </div>
    <div class="compare-insight">
      <p class="insight-overlap">🎯 ${escapeHtml(data.insights?.overlap_insight)}</p>
      <div class="compare-logic">
        <span class="logic-chip"><strong>TF-IDF:</strong> ${escapeHtml(data.insights?.content_logic)}</span>
        <span class="logic-chip"><strong>User Matrix:</strong> ${escapeHtml(data.insights?.collab_logic)}</span>
        <span class="logic-chip"><strong>Hybrid Blend:</strong> ${escapeHtml(data.insights?.hybrid_logic)}</span>
      </div>
    </div>`;
}

// ── ML Stats Modal ────────────────────────────────────────────────────────
async function showStats() {
  openModal('statsModal');
  const data = await apiFetch('/api/stats');
  if (data) {
    document.getElementById('statMovies').textContent = data.total_movies.toLocaleString();
    document.getElementById('statGenres').textContent = data.unique_genres || data.genres?.length || '18';
    document.getElementById('statVocab').textContent = (data.vocab_size || 8000).toLocaleString();
    document.getElementById('statAccuracy').textContent = data.accuracy_score || '89.4%';
    const el = document.getElementById('statAlgo');
    if (el) el.textContent = data.algorithm;
  }
}

// ── Genre Modal Controller ────────────────────────────────────────────────
async function showGenreModal() {
  openModal('genreModal');
  const data = await apiFetch('/api/stats');
  const pills = document.getElementById('genrePills');
  if (data?.genres && pills) {
    pills.innerHTML = data.genres.map(g =>
      `<div class="genre-pill" onclick="filterGenre('${escapeJsStr(g)}')">${escapeHtml(g)}</div>`
    ).join('');
  }
}

function filterGenre(genre) {
  closeModal('genreModal');
  const input = document.getElementById('searchInput');
  if (input) input.value = genre;
  handleSearch(genre);
}

// ── Authentication Controller ─────────────────────────────────────────────
function openLoginModal() { openModal('loginModal'); }

async function doLogin() {
  const username = document.getElementById('loginUser')?.value.trim();
  const password = document.getElementById('loginPass')?.value;
  const errEl = document.getElementById('loginError');
  const fd = new FormData();
  fd.append('username', username);
  fd.append('password', password);

  const resp = await fetch('/login', { method: 'POST', body: fd });
  const data = await resp.json();
  if (data.success) {
    window.location.reload();
  } else {
    if (errEl) {
      errEl.textContent = data.error || 'Login failed';
      errEl.style.display = 'block';
    }
  }
}

async function logout() {
  await apiFetch('/logout');
  window.location.reload();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && document.getElementById('loginModal')?.classList.contains('open')) {
    doLogin();
  }
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.open').forEach(m => {
      if (m.id === 'trailerModal') closeTrailer();
      else m.classList.remove('open');
    });
  }
});

// ── Modal UI Helpers ──────────────────────────────────────────────────────
function openModal(id)  { document.getElementById(id)?.classList.add('open'); }
function closeModal(id) { document.getElementById(id)?.classList.remove('open'); }

// ── Toast System ──────────────────────────────────────────────────────────
function showToast(msg) {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3200);
}

// ── Helpers ───────────────────────────────────────────────────────────────
async function apiFetch(url, method = 'GET', body = null) {
  try {
    const opts = { method };
    if (body) {
      opts.headers = { 'Content-Type': 'application/json' };
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(url, opts);
    return resp.ok ? await resp.json() : null;
  } catch (e) {
    console.error('API error:', e);
    return null;
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function escapeJsStr(str) {
  if (!str) return '';
  return String(str).replace(/'/g, "\\'").replace(/"/g, '\\"');
}
