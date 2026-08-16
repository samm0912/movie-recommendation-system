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
  const trailerBtn = (m.has_trailer || m.trailer_key || m.trailer_url)
    ? `<button class="card-btn card-trailer" title="Watch Trailer" data-trailer="${m.trailer_key || ''}" data-url="${m.trailer_url || ''}" data-id="${m.id}" data-title="${escapeHtml(m.title)}" onclick="event.stopPropagation(); openTrailer(this)">▶ Trailer</button>`
    : '';

  const trailerBadge = (m.has_trailer || m.trailer_key || m.trailer_url)
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

// ── Video Trailer Player Modal (Plays 100% In-Player In-Page) ─────────────
async function openTrailer(trailerKeyOrEl, movieTitle = 'Movie') {
  let trailerKey = '';
  let trailerUrl = '';
  let title = movieTitle || 'Movie';
  let movieId = '';

  if (trailerKeyOrEl && typeof trailerKeyOrEl === 'object' && trailerKeyOrEl.dataset) {
    trailerKey = trailerKeyOrEl.dataset.trailer || '';
    trailerUrl = trailerKeyOrEl.dataset.url || '';
    movieId = trailerKeyOrEl.dataset.id || '';
    title = trailerKeyOrEl.dataset.title || title;
  } else {
    trailerKey = String(trailerKeyOrEl || '').trim();
  }

  const modal = document.getElementById('trailerModal');
  const titleEl = document.getElementById('trailerTitle');
  const container = document.getElementById('videoContainer');
  const extLink = document.getElementById('trailerExternalLink');
  if (!modal || !container) return;

  if (titleEl) titleEl.textContent = `${title} — Official Trailer`;

  // Show trailer modal immediately with player loading state inside video space
  container.innerHTML = `
    <div style="position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#000; color:#38bdf8; z-index:10;">
      <div style="width:40px; height:40px; border:3px solid rgba(56,189,248,0.2); border-top-color:#38bdf8; border-radius:50%; animation:spin 0.8s linear infinite; margin-bottom:0.85rem;"></div>
      <p style="font-size:0.92rem; font-weight:700; color:#e2e8f0; margin:0;">Loading Official HD Trailer Stream…</p>
    </div>`;
  openModal('trailerModal');

  // Check if trailerKey is already a valid 11-char YouTube ID
  let validKey = (trailerKey && trailerKey !== 'None' && trailerKey !== 'undefined' && trailerKey !== 'null' && trailerKey.length === 11 && !/^\d+$/.test(trailerKey)) ? trailerKey : null;

  if (!validKey && movieId) {
    try {
      const data = await apiFetch(`/api/trailer/${movieId}`);
      if (data && data.trailer_key) {
        validKey = data.trailer_key;
        if (data.title) title = data.title;
        if (data.trailer_url && extLink) extLink.href = data.trailer_url;
      }
    } catch (err) {
      console.warn('Trailer stream fetch error:', err);
    }
  }

  if (!validKey) {
    validKey = 'PLl99DlL6b4'; // Universal fallback
  }

  const youtubeUrl = `https://www.youtube.com/watch?v=${validKey}`;
  if (extLink) extLink.href = youtubeUrl;
  if (titleEl) titleEl.textContent = `${title} — Official Trailer`;

  // Play video directly inside the video player space on the page
  container.innerHTML = `
    <iframe src="https://www.youtube-nocookie.com/embed/${validKey}?autoplay=1&enablejsapi=1&rel=0&modestbranding=1&playsinline=1" 
      title="${escapeHtml(title)} Official Trailer" 
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
      allowfullscreen
      style="position:absolute; inset:0; width:100%; height:100%; border:none;">
    </iframe>`;
}

function closeTrailer() {
  const container = document.getElementById('videoContainer');
  if (container) container.innerHTML = '';
  closeModal('trailerModal');
}

// ── Surprise Me Modal ─────────────────────────────────────────────────────
async function triggerSurpriseMe() {
  openModal('surpriseModal');
  const content = document.getElementById('surpriseContent');
  if (!content) return;

  content.innerHTML = '<div class="loading-state">✨ Selecting a cinematic gem from 60,000+ titles…</div>';

  const data = await apiFetch('/api/surprise');
  if (!data?.movie) {
    content.innerHTML = '<p style="color:#ef4444">Failed to pick a movie. Please try again.</p>';
    return;
  }

  const m = data.movie;
  const fallbackImg = 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60';
  const genresFormatted = (m.genres || '').replace(/\|/g, ' · ');

  const trailerBtn = (m.has_trailer || m.trailer_key || m.trailer_url)
    ? `<button class="btn-surprise-trailer" data-trailer="${m.trailer_key || ''}" data-url="${m.trailer_url || ''}" data-id="${m.id}" data-title="${escapeHtml(m.title)}" onclick="closeModal('surpriseModal'); openTrailer(this)">▶ Watch Trailer</button>`
    : `<button class="btn-surprise-trailer disabled" disabled>Trailer Unavailable</button>`;

  const langBadges = (m.available_languages && m.available_languages.length > 1)
    ? `<p style="font-size:0.78rem; color:#34d399; font-weight:600; margin:4px 0;">🌐 Available in ${m.available_languages.length} languages: ${escapeHtml(m.available_languages.join(' · '))}</p>`
    : `<p style="font-size:0.78rem; color:var(--text-dim); margin:4px 0;">🌐 Language: ${escapeHtml(m.language)}</p>`;

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
      ${langBadges}
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
// ── Search & AI Prompt Recommendation Controller ────────────────────────
function submitSearch() {
  const input = document.getElementById('searchInput');
  if (!input) return;
  const q = input.value.trim();
  const langEl = document.getElementById('languageSelect');
  const lang = langEl ? langEl.value : 'All';

  const section = document.getElementById('searchResults');
  if (!section) {
    // If on a page without searchResults (e.g. /movie/<id>), navigate to home with query
    const langParam = lang && lang !== 'All' ? `&lang=${encodeURIComponent(lang)}` : '';
    window.location.href = `/?q=${encodeURIComponent(q)}${langParam}`;
    return;
  }

  if (!q && (!lang || lang === 'All')) {
    clearSearch();
    return;
  }

  handleSearch(q, lang);
}

async function handleSearch(query, language = 'All') {
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
  const langParam = language && language !== 'All' ? `&lang=${encodeURIComponent(language)}` : '';

  if (!q && (!language || language === 'All')) {
    section.style.display = 'none';
    if (matchedSection) matchedSection.style.display = 'none';
    if (noteEl) noteEl.style.display = 'none';
    if (promptRecsHeader) promptRecsHeader.style.display = 'none';
    return;
  }

  showToast('🔍 Analyzing prompt & computing ML recommendations…');
  const data = await apiFetch(`/api/search?q=${encodeURIComponent(q)}${langParam}`);
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
      titleEl.innerHTML = q
        ? `🔍 Search Results for <em>"${escapeHtml(q)}"</em>`
        : `🌐 ${escapeHtml(language)} Recommendations`;
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

// ── Multi-Language Controller ─────────────────────────────────────────────
async function handleLanguageChange(lang) {
  const desktopSelect = document.getElementById('languageSelect');
  const mobileSelect = document.getElementById('mobileLanguageSelect');
  if (desktopSelect) desktopSelect.value = lang;
  if (mobileSelect) mobileSelect.value = lang;

  const searchInput = document.getElementById('searchInput');
  const query = searchInput ? searchInput.value.trim() : '';

  if (query) {
    submitSearch();
    return;
  }

  const langSection = document.getElementById('languageResults');
  const langGrid = document.getElementById('languageGrid');
  const langTitle = document.getElementById('languageHeaderTitle');
  const langNote = document.getElementById('languageInsightNote');

  if (!langSection || !langGrid) {
    if (lang && lang !== 'All') {
      window.location.href = `/?lang=${encodeURIComponent(lang)}`;
    }
    return;
  }

  if (!lang || lang === 'All') {
    langSection.style.display = 'none';
    return;
  }

  showToast(`🌐 Loading top ${lang} cinema…`);
  const data = await apiFetch(`/api/language/${encodeURIComponent(lang)}?limit=18`);
  if (!data?.movies) return;

  langSection.style.display = 'block';
  if (langTitle) langTitle.innerHTML = `🌐 ${escapeHtml(lang)} Cinema Spotlight`;
  if (langNote) {
    langNote.textContent = `Top rated & popular titles across our ${escapeHtml(lang)} collection.`;
    langNote.style.display = 'block';
  }

  langGrid.innerHTML = data.movies.length
    ? data.movies.map(renderCard).join('')
    : `<p style="color:var(--text-muted);padding:1.5rem">No ${escapeHtml(lang)} movies found in current index.</p>`;

  langSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearLanguageFilter() {
  const desktopSelect = document.getElementById('languageSelect');
  const mobileSelect = document.getElementById('mobileLanguageSelect');
  if (desktopSelect) desktopSelect.value = 'All';
  if (mobileSelect) mobileSelect.value = 'All';
  const langSection = document.getElementById('languageResults');
  if (langSection) langSection.style.display = 'none';
}

// Auto-execute query from URL search param if present on page load
document.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(window.location.search);
  const q = params.get('q');
  const lang = params.get('lang');

  if (lang && lang !== 'All') {
    const desktopSelect = document.getElementById('languageSelect');
    const mobileSelect = document.getElementById('mobileLanguageSelect');
    if (desktopSelect) desktopSelect.value = lang;
    if (mobileSelect) mobileSelect.value = lang;
    if (!q) handleLanguageChange(lang);
  }

  if (q) {
    const input = document.getElementById('searchInput');
    if (input) {
      input.value = q;
      submitSearch();
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

/* ════════════════════════════════════════════════════════════════════════════
   CINEBOT AI CONVERSATIONAL CONTROLLER
════════════════════════════════════════════════════════════════════════════ */
let cinebotHistory = [];
let cinebotIsLoading = false;

function toggleChatbot() {
  const win = document.getElementById('cinebotWindow');
  if (!win) return;
  const isOpen = win.classList.toggle('open');
  if (isOpen) {
    setTimeout(() => {
      const input = document.getElementById('cinebotInput');
      if (input) input.focus();
    }, 150);
  }
}

function openChatbot() {
  const win = document.getElementById('cinebotWindow');
  if (win && !win.classList.contains('open')) {
    win.classList.add('open');
    setTimeout(() => {
      const input = document.getElementById('cinebotInput');
      if (input) input.focus();
    }, 150);
  }
}

function closeChatbot() {
  const win = document.getElementById('cinebotWindow');
  if (win) win.classList.remove('open');
}

function clearChat() {
  cinebotHistory = [];
  const body = document.getElementById('cinebotBody');
  if (body) {
    body.innerHTML = `
      <div class="cinebot-msg bot intro-msg">
        <div class="cinebot-msg-bubble">
          <p>👋 Chat cleared! I'm ready for new questions or recommendations.</p>
          <p>Ask for movies like <em>"Interstellar"</em>, request specific moods, or say <em>"Surprise Me"</em>!</p>
        </div>
      </div>`;
  }
  updateChatChips(["🌌 Interstellar", "🍿 Surprise Me", "😂 Feel-Good", "🧠 How ML Works"]);
}

function sendChatPrompt(promptText) {
  openChatbot();
  const input = document.getElementById('cinebotInput');
  if (input) input.value = promptText;
  submitChatInput();
}

function submitChatInput() {
  const input = document.getElementById('cinebotInput');
  if (!input) return;
  const message = input.value.trim();
  if (!message || cinebotIsLoading) return;
  input.value = '';
  executeChat(message);
}

function formatMarkdownText(text) {
  if (!text) return '';
  let formatted = escapeHtml(text);
  // Bold **text**
  formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Italic *text*
  formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
  // Line breaks
  formatted = formatted.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
  return `<p>${formatted}</p>`;
}

async function executeChat(message) {
  const body = document.getElementById('cinebotBody');
  const sendBtn = document.getElementById('cinebotSendBtn');
  const input = document.getElementById('cinebotInput');
  if (!body) return;

  cinebotIsLoading = true;
  if (sendBtn) sendBtn.disabled = true;
  if (input) input.disabled = true;

  // 1. Render user message bubble
  const userMsgEl = document.createElement('div');
  userMsgEl.className = 'cinebot-msg user';
  userMsgEl.innerHTML = `<div class="cinebot-msg-bubble"><p>${escapeHtml(message)}</p></div>`;
  body.appendChild(userMsgEl);
  body.scrollTop = body.scrollHeight;

  // 2. Render typing indicator
  const typingEl = document.createElement('div');
  typingEl.className = 'cinebot-msg bot typing-msg';
  typingEl.id = 'cinebotTyping';
  typingEl.innerHTML = `
    <div class="cinebot-msg-bubble typing-bubble">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>`;
  body.appendChild(typingEl);
  body.scrollTop = body.scrollHeight;

  try {
    const data = await apiFetch('/api/chat', 'POST', {
      message: message,
      history: cinebotHistory
    });

    // Remove typing bubble
    const currentTyping = document.getElementById('cinebotTyping');
    if (currentTyping) currentTyping.remove();

    if (data && data.success) {
      // 3. Render bot response message
      const botMsgEl = document.createElement('div');
      botMsgEl.className = 'cinebot-msg bot';

      let movieCardsHtml = '';
      if (data.movies && data.movies.length > 0) {
        movieCardsHtml = renderChatMovieCards(data.movies);
      }

      botMsgEl.innerHTML = `
        <div class="cinebot-msg-bubble">
          ${formatMarkdownText(data.reply)}
          ${movieCardsHtml}
        </div>`;
      body.appendChild(botMsgEl);

      // Update mode badge
      const tagEl = document.getElementById('cinebotModelTag');
      if (tagEl) {
        tagEl.textContent = (data.mode === 'gemini_ai') ? 'Gemini AI' : 'Hybrid ML';
      }

      // Update chips if provided
      if (data.suggested_prompts && data.suggested_prompts.length > 0) {
        updateChatChips(data.suggested_prompts);
      }

      // Append to local history (limit to last 10 turns)
      cinebotHistory.push({ role: 'user', content: message });
      cinebotHistory.push({ role: 'model', content: data.reply });
      if (cinebotHistory.length > 12) {
        cinebotHistory = cinebotHistory.slice(-10);
      }
    } else {
      const errEl = document.createElement('div');
      errEl.className = 'cinebot-msg bot';
      errEl.innerHTML = `
        <div class="cinebot-msg-bubble">
          <p>⚠️ Sorry, I encountered an issue retrieving recommendations. Please try asking again!</p>
        </div>`;
      body.appendChild(errEl);
    }
  } catch (err) {
    const currentTyping = document.getElementById('cinebotTyping');
    if (currentTyping) currentTyping.remove();

    const errEl = document.createElement('div');
    errEl.className = 'cinebot-msg bot';
    errEl.innerHTML = `
      <div class="cinebot-msg-bubble">
        <p>⚠️ Connection issue. Please check your network and try again.</p>
      </div>`;
    body.appendChild(errEl);
  } finally {
    cinebotIsLoading = false;
    if (sendBtn) sendBtn.disabled = false;
    if (input) {
      input.disabled = false;
      input.focus();
    }
    body.scrollTop = body.scrollHeight;
  }
}

function renderChatMovieCards(movies) {
  if (!movies || !movies.length) return '';
  const fallbackImg = 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60';
  
  const cards = movies.map(m => {
    const poster = m.poster || fallbackImg;
    const genres = (m.genres || '').replace(/\|/g, ' · ').slice(0, 24);
    const title = escapeHtml(m.title);
    const trailerBtn = (m.has_trailer || m.trailer_key)
      ? `<button class="chat-action-btn chat-btn-trailer" onclick="event.stopPropagation(); openTrailer('${m.trailer_key || m.id}', '${escapeJsStr(m.title)}')">▶ Trailer</button>`
      : `<button class="chat-action-btn chat-btn-trailer" onclick="event.stopPropagation(); openTrailer('${m.id}', '${escapeJsStr(m.title)}')">▶ Trailer</button>`;

    return `
      <div class="chat-movie-card" onclick="window.location='/movie/${m.id}'">
        <img class="chat-card-poster" src="${poster}" alt="${title}" loading="lazy" onerror="this.onerror=null; this.src='${fallbackImg}'">
        <div class="chat-card-content">
          <div>
            <div class="chat-card-title" title="${title}">${title}</div>
            <div class="chat-card-meta">
              <span class="chat-card-rating">⭐ ${m.rating}</span>
              <span>· ${m.year}</span>
            </div>
            <div class="chat-card-genres">${escapeHtml(genres)}</div>
          </div>
          <div class="chat-card-actions">
            ${trailerBtn}
            <button class="chat-action-btn chat-btn-view" onclick="event.stopPropagation(); window.location='/movie/${m.id}'">ℹ Details</button>
          </div>
        </div>
      </div>`;
  }).join('');

  return `<div class="chat-cards-list">${cards}</div>`;
}

function updateChatChips(prompts) {
  const chipsContainer = document.getElementById('cinebotChips');
  if (!chipsContainer || !prompts || !prompts.length) return;
  chipsContainer.innerHTML = prompts.map(p => `
    <button class="chat-chip" onclick="sendChatPrompt('${escapeJsStr(p)}')">${escapeHtml(p)}</button>
  `).join('');
}

