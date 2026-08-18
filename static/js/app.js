/* ── app.js — Movie Recommendation System Frontend Controller ── */

// Navbar scroll background effect
window.addEventListener('scroll', () => {
  document.getElementById('navbar')?.classList.toggle('scrolled', window.scrollY > 40);
});

// Mobile Navigation Toggle
function toggleMobileNav(forceState = null) {
  const toggle = document.getElementById('mobileNavToggle');
  const drawer = document.getElementById('mobileNavDrawer');
  if (toggle && drawer) {
    const shouldOpen = forceState !== null ? forceState : !drawer.classList.contains('open');
    toggle.classList.toggle('open', shouldOpen);
    drawer.classList.toggle('open', shouldOpen);
  }
}

// Close mobile drawer when resizing back to desktop or clicking outside
window.addEventListener('resize', () => {
  if (window.innerWidth > 1024) {
    toggleMobileNav(false);
  }
});

document.addEventListener('click', (e) => {
  const drawer = document.getElementById('mobileNavDrawer');
  const toggle = document.getElementById('mobileNavToggle');
  if (drawer && drawer.classList.contains('open')) {
    if (!drawer.contains(e.target) && toggle && !toggle.contains(e.target)) {
      toggleMobileNav(false);
    }
  }
});

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
          <span class="card-badge-rating">⭐ ${m.rating}/10</span>
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
            <p class="card-meta"><span class="meta-star">⭐ ${m.rating}/10</span> · ${m.year}</p>
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
  } else if (trailerKeyOrEl && !isNaN(trailerKeyOrEl) && Number(trailerKeyOrEl) > 100) {
    movieId = String(trailerKeyOrEl);
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
  let validKey = (trailerKey && trailerKey !== 'None' && trailerKey !== 'undefined' && trailerKey !== 'null' && trailerKey.length === 11 && !/^\d+$/.test(trailerKey) && trailerKey !== 'PLl99DlL6b4') ? trailerKey : null;
  let embedUrl = '';
  let youtubeUrl = trailerUrl || `https://www.youtube.com/results?search_query=${encodeURIComponent(title + ' official trailer')}`;

  if ((!validKey || !trailerUrl) && movieId) {
    try {
      const data = await apiFetch(`/api/trailer/${movieId}`);
      if (data && data.success) {
        if (data.trailer_key) validKey = data.trailer_key;
        if (data.title) title = data.title;
        if (data.trailer_url) youtubeUrl = data.trailer_url;
        if (data.embed_url) embedUrl = data.embed_url;
      }
    } catch (err) {
      console.warn('Trailer stream fetch error:', err);
    }
  }

  if (validKey) {
    embedUrl = `https://www.youtube-nocookie.com/embed/${validKey}?autoplay=1&enablejsapi=1&rel=0&modestbranding=1&playsinline=1`;
    youtubeUrl = `https://www.youtube.com/watch?v=${validKey}`;
  } else if (!embedUrl) {
    const searchParam = encodeURIComponent(title + ' official trailer');
    embedUrl = `https://www.youtube-nocookie.com/embed?listType=search&list=${searchParam}&autoplay=1`;
    youtubeUrl = `https://www.youtube.com/results?search_query=${searchParam}`;
  }

  if (extLink) extLink.href = youtubeUrl;
  if (titleEl) titleEl.textContent = `${title} — Official Trailer`;

  // Play video directly inside the video player space on the page
  container.innerHTML = `
    <iframe src="${embedUrl}" 
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

  // Initialize Discover by IMDb Rating section if present
  if (document.getElementById('ratingResultsGrid')) {
    updateRatingVisuals(7.5);
    loadRatingMovies(7.5);
  }
});

// ── Feature: Discover by IMDb Rating Filter Controller ────────────────────
let ratingDebounceTimer = null;
let currentRatingFilter = 7.5;
let currentRatingMax = null;

function getTierLabel(minR, maxR) {
  if (minR >= 8.5) return '8.5 – 10.0 (Masterpieces)';
  if (minR >= 8.0) return '8.0 – 8.4 (All-Time Classics)';
  if (minR >= 7.5) return '7.5 – 7.9 (Highly Acclaimed)';
  if (minR >= 7.0) return '7.0 – 7.4 (Popular Quality)';
  if (minR >= 6.0) return '6.0 – 6.9 (Solid Watchlist)';
  if (minR >= 5.0) return '5.0 – 5.9 (Average Gems)';
  if (minR <= 0.0) return 'All Ratings (0.0 – 10.0)';
  return `${minR.toFixed(1)} – ${(minR + 0.8).toFixed(1)}`;
}

function updateRatingVisuals(val) {
  const num = parseFloat(val);
  const slider = document.getElementById('ratingRangeSlider');
  const display = document.getElementById('ratingDisplayVal');
  const btnText = document.getElementById('btnRatingBrowseText');

  if (slider && parseFloat(slider.value) !== num) {
    slider.value = num;
  }

  // Real-time track fill
  if (slider) {
    const pct = Math.max(0, Math.min(100, (num / 10) * 100));
    slider.style.background = `linear-gradient(90deg, #f59e0b 0%, #f59e0b ${pct}%, #374151 ${pct}%, #4b5563 100%)`;
  }

  // Update badge and button
  if (display) display.textContent = num.toFixed(1);
  if (btnText) btnText.textContent = `Browse ${num > 0 ? num.toFixed(1) + ' Tier' : 'All Movies'}`;

  // Update active label highlight
  let closestLabel = null;
  let minDiff = Infinity;
  document.querySelectorAll('.rating-slider-labels span').forEach(span => {
    const spanVal = parseFloat(span.getAttribute('data-val'));
    const diff = Math.abs(spanVal - num);
    if (diff < minDiff && diff <= 0.35) {
      minDiff = diff;
      closestLabel = span;
    }
    span.classList.remove('active-label');
  });
  if (closestLabel) {
    closestLabel.classList.add('active-label');
  }

  // Update quick preset button states
  document.querySelectorAll('.rating-preset-btn').forEach(btn => {
    btn.classList.remove('active');
  });
  if (Math.abs(num - 8.5) < 0.1) {
    document.querySelector(".rating-preset-btn[data-preset='8.5']")?.classList.add('active');
  } else if (Math.abs(num - 7.5) < 0.1) {
    document.getElementById('preset75')?.classList.add('active');
    document.querySelector(".rating-preset-btn[data-preset='7.5']")?.classList.add('active');
  } else if (Math.abs(num - 7.0) < 0.1) {
    document.querySelector(".rating-preset-btn[data-preset='7.0']")?.classList.add('active');
  } else if (Math.abs(num - 6.0) < 0.1) {
    document.querySelector(".rating-preset-btn[data-preset='6.0']")?.classList.add('active');
  } else if (num <= 0.1) {
    document.querySelector(".rating-preset-btn[data-preset='0']")?.classList.add('active');
  }
}

function handleRatingSliderChange(val) {
  const num = parseFloat(val);
  currentRatingFilter = num;
  currentRatingMax = null;

  updateRatingVisuals(num);

  clearTimeout(ratingDebounceTimer);
  ratingDebounceTimer = setTimeout(() => {
    loadRatingMovies(num);
  }, 120);
}

function setRatingFilter(minVal, maxVal = null) {
  const minNum = parseFloat(minVal);
  currentRatingFilter = minNum;
  currentRatingMax = maxVal !== null ? parseFloat(maxVal) : null;

  updateRatingVisuals(minNum);
  loadRatingMovies(minNum, currentRatingMax);
}

function executeRatingSearch() {
  const slider = document.getElementById('ratingRangeSlider');
  const num = slider ? parseFloat(slider.value) : currentRatingFilter;
  currentRatingFilter = num;

  const btn = document.getElementById('btnRatingBrowse');
  if (btn) {
    btn.classList.add('loading');
    const btnText = document.getElementById('btnRatingBrowseText');
    if (btnText) btnText.textContent = 'Searching…';
  }

  loadRatingMovies(num, currentRatingMax).finally(() => {
    if (btn) {
      btn.classList.remove('loading');
      const btnText = document.getElementById('btnRatingBrowseText');
      if (btnText) btnText.textContent = `Browse ${num > 0 ? num.toFixed(1) + ' Tier' : 'All Movies'}`;
    }
  });

  const section = document.getElementById('ratingDiscoverSection');
  if (section) {
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

async function loadRatingMovies(minRating, maxRating = null) {
  const grid = document.getElementById('ratingResultsGrid');
  if (!grid) return;

  const minNum = parseFloat(minRating !== undefined ? minRating : currentRatingFilter);
  const currentLang = document.getElementById('languageSelect')?.value;
  const currentGenre = document.getElementById('ratingGenreSelect')?.value;

  const langParam = (currentLang && currentLang !== 'All') ? `&lang=${encodeURIComponent(currentLang)}` : '';
  const genreParam = (currentGenre && currentGenre !== 'All') ? `&genre=${encodeURIComponent(currentGenre)}` : '';
  const maxParam = (maxRating !== null && maxRating !== undefined) ? `&max_rating=${maxRating}` : '';

  // Synchronize visuals
  updateRatingVisuals(minNum);

  const statusText = document.getElementById('ratingStatusText');
  const statusCount = document.getElementById('ratingStatusCount');

  // Loading skeleton
  grid.innerHTML = '<div class="loading-cards"><div class="skeleton-card"></div><div class="skeleton-card"></div><div class="skeleton-card"></div><div class="skeleton-card"></div><div class="skeleton-card"></div><div class="skeleton-card"></div></div>';

  try {
    const data = await apiFetch(`/api/discover/rating?min_rating=${minNum}&limit=18${maxParam}${langParam}${genreParam}`);
    if (data?.movies?.length) {
      grid.innerHTML = data.movies.map(renderCard).join('');
      const genreStr = currentGenre && currentGenre !== 'All' ? ` ${currentGenre}` : '';
      const langStr = currentLang && currentLang !== 'All' ? ` [${currentLang}]` : '';
      const tierDesc = getTierLabel(minNum, maxRating);
      if (statusText) {
        statusText.innerHTML = minNum > 0
          ? `Showing top <strong>${genreStr}${langStr}</strong> movies in rating tier <strong>${tierDesc}</strong>`
          : `Showing top <strong>${genreStr}${langStr}</strong> movies across all IMDb ratings`;
      }
      if (statusCount) statusCount.textContent = `${data.movies.length} titles`;
    } else {
      grid.innerHTML = `<div style="padding:2rem;text-align:center;color:var(--text-muted);width:100%">
        <p style="font-weight:700;font-size:1.05rem;color:#fff;margin-bottom:0.4rem">No matching titles found</p>
        <p style="font-size:0.85rem">No movies found matching IMDb rating in ${minNum.toFixed(1)} tier${currentGenre && currentGenre !== 'All' ? ' in ' + currentGenre : ''}. Try adjusting the rating slider or resetting the genre filter.</p>
      </div>`;
      if (statusCount) statusCount.textContent = '0 titles';
    }
  } catch (err) {
    grid.innerHTML = `<p style="color:var(--text-muted);padding:1.5rem">Error loading movies. Please try again.</p>`;
  }
}

// ── Watchlist / Like Controller ───────────────────────────────────────────
async function likeMovie(movieId, btn) {
  const data = await apiFetch(`/api/like/${movieId}`, 'POST');
  if (data?.success) {
    const added = data.action === 'added';
    if (btn) {
      btn.classList.toggle('liked', added);
      btn.textContent = added ? '♥' : '♡';
    }
    showToast(added ? '❤️ Added to your liked watchlist!' : 'Removed from watchlist');
    
    // Update profile liked counter if present on page
    const likedCountEl = document.getElementById('profileLikedCount');
    if (likedCountEl && data.total_liked !== undefined) {
      likedCountEl.textContent = data.total_liked;
    }
  } else {
    showToast('Please sign in to save movies to your profile watchlist');
    openLoginModal();
  }
}

// ── Feature: User Profile, Liked & Rated Controller ────────────────────────
let currentProfileTab = 'liked';
let cachedProfileData = null;

function switchProfileTab(tabName) {
  currentProfileTab = tabName;

  // Tab buttons
  const btnLiked = document.getElementById('tabBtnLiked');
  const btnRated = document.getElementById('tabBtnRated');
  if (btnLiked) btnLiked.classList.toggle('active', tabName === 'liked');
  if (btnRated) btnRated.classList.toggle('active', tabName === 'rated');

  // Stats bar items
  const statLiked = document.getElementById('statLikedTab');
  const statRated = document.getElementById('statRatedTab');
  if (statLiked) statLiked.classList.toggle('active', tabName === 'liked');
  if (statRated) statRated.classList.toggle('active', tabName === 'rated');

  // Content areas
  const contentLiked = document.getElementById('profileLikedTabContent');
  const contentRated = document.getElementById('profileRatedTabContent');
  if (contentLiked) contentLiked.style.display = (tabName === 'liked') ? 'block' : 'none';
  if (contentRated) contentRated.style.display = (tabName === 'rated') ? 'block' : 'none';
}

async function openProfileModal(initialTab = null) {
  openModal('profileModal');
  if (initialTab) {
    switchProfileTab(initialTab);
  } else {
    switchProfileTab(currentProfileTab || 'liked');
  }

  const likedContainer = document.getElementById('profileLikedContainer');
  const ratedContainer = document.getElementById('profileRatedContainer');
  if (likedContainer) likedContainer.innerHTML = '<div class="loading-state">Loading your liked movies…</div>';
  if (ratedContainer) ratedContainer.innerHTML = '<div class="loading-state">Loading your rated movies…</div>';

  const data = await apiFetch('/api/user/profile');
  if (!data?.success || data?.require_login) {
    closeModal('profileModal');
    showToast('Please sign in to view your profile');
    openLoginModal();
    return;
  }

  cachedProfileData = data;
  renderProfileModalData(data);
}

function renderProfileModalData(data) {
  const user = data.user || {};
  const likedMovies = data.liked_movies || [];
  const ratedMovies = data.rated_movies || [];

  const nameEl = document.getElementById('profileDisplayName');
  const userEl = document.getElementById('profileUsername');
  const avatarEl = document.getElementById('profileAvatarChar');
  const likedCountEl = document.getElementById('profileLikedCount');
  const ratingsCountEl = document.getElementById('profileRatingsCount');
  const tabCountLiked = document.getElementById('tabCountLiked');
  const tabCountRated = document.getElementById('tabCountRated');

  if (nameEl) nameEl.textContent = user.name || 'User Profile';
  if (userEl) userEl.textContent = '@' + (user.username || 'user');
  if (avatarEl) avatarEl.textContent = (user.name || 'U')[0].toUpperCase();
  if (likedCountEl) likedCountEl.textContent = likedMovies.length;
  if (ratingsCountEl) ratingsCountEl.textContent = ratedMovies.length;
  if (tabCountLiked) tabCountLiked.textContent = likedMovies.length;
  if (tabCountRated) tabCountRated.textContent = ratedMovies.length;

  const fallbackImg = 'https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=500&auto=format&fit=crop&q=60';

  // 1. Render Liked Movies
  const likedContainer = document.getElementById('profileLikedContainer');
  if (likedContainer) {
    if (!likedMovies.length) {
      likedContainer.innerHTML = `
        <div class="profile-empty-state">
          <div class="profile-empty-icon">💔</div>
          <h4 class="profile-empty-title">No Liked Movies Yet</h4>
          <p class="profile-empty-desc">You haven't liked any movies yet. Explore Telugu, Hindi, Malayalam, or international cinema, or ask CineBot for recommendations, and click the ♡ button to save them here!</p>
          <button class="btn-profile-explore" onclick="closeModal('profileModal')">Explore Movies</button>
        </div>`;
    } else {
      likedContainer.innerHTML = `
        <div class="profile-liked-grid">
          ${likedMovies.map(m => {
            const poster = m.poster || fallbackImg;
            const genres = (m.genres || '').replace(/\|/g, ' · ').slice(0, 26);
            const title = escapeHtml(m.title);
            const trailerBtn = (m.has_trailer || m.trailer_key || m.trailer_url)
              ? `<button class="liked-card-action-btn liked-btn-trailer" onclick="event.stopPropagation(); openTrailer(this)" data-trailer="${m.trailer_key || ''}" data-url="${m.trailer_url || ''}" data-id="${m.id}" data-title="${title}">▶ Trailer</button>`
              : '';

            return `
              <div class="liked-movie-card" id="likedCard-${m.id}" onclick="window.location='/movie/${m.id}'">
                <div class="liked-card-img-wrap">
                  <img src="${poster}" alt="${title}" loading="lazy" onerror="this.onerror=null; this.src='${fallbackImg}'">
                  <span class="liked-card-rating">⭐ ${m.rating}/10</span>
                  <span class="liked-card-lang">🌐 ${escapeHtml(m.language || 'Cinema')}</span>
                </div>
                <div class="liked-card-info">
                  <h4 class="liked-card-title" title="${title}">${title}</h4>
                  <p class="liked-card-meta">${m.year} · ${escapeHtml(genres)}</p>
                  <div class="liked-card-actions" onclick="event.stopPropagation()">
                    ${trailerBtn}
                    <a href="/movie/${m.id}" class="liked-card-action-btn liked-btn-view">ℹ Details</a>
                    <button class="liked-card-action-btn liked-btn-remove" title="Remove from Liked" onclick="removeLikedMovie(${m.id}, this)">✕ Unlike</button>
                  </div>
                </div>
              </div>`;
          }).join('')}
        </div>`;
    }
  }

  // 2. Render Rated Movies with interactive Re-change Ratings
  const ratedContainer = document.getElementById('profileRatedContainer');
  if (ratedContainer) {
    if (!ratedMovies.length) {
      ratedContainer.innerHTML = `
        <div class="profile-empty-state">
          <div class="profile-empty-icon">⭐</div>
          <h4 class="profile-empty-title">No Rated Movies Yet</h4>
          <p class="profile-empty-desc">You haven't rated any movies yet. Rate your favorite films (1 to 5 stars) to train our collaborative filtering ML model and get ultra-personalized recommendations!</p>
          <button class="btn-profile-explore" onclick="closeModal('profileModal')">Explore &amp; Rate Movies</button>
        </div>`;
    } else {
      ratedContainer.innerHTML = `
        <div class="profile-rated-grid">
          ${ratedMovies.map(m => {
            const poster = m.poster || fallbackImg;
            const genres = (m.genres || '').replace(/\|/g, ' · ').slice(0, 26);
            const title = escapeHtml(m.title);
            const currentRating = parseInt(m.user_rating || 5);
            const trailerBtn = (m.has_trailer || m.trailer_key || m.trailer_url)
              ? `<button class="liked-card-action-btn liked-btn-trailer" onclick="event.stopPropagation(); openTrailer(this)" data-trailer="${m.trailer_key || ''}" data-url="${m.trailer_url || ''}" data-id="${m.id}" data-title="${title}">▶ Trailer</button>`
              : '';

            return `
              <div class="rated-movie-card" id="ratedCard-${m.id}" onclick="window.location='/movie/${m.id}'">
                <div class="liked-card-img-wrap">
                  <img src="${poster}" alt="${title}" loading="lazy" onerror="this.onerror=null; this.src='${fallbackImg}'">
                  <span class="liked-card-rating">⭐ ${m.rating}/10 IMDb</span>
                  <span class="liked-card-lang">🌐 ${escapeHtml(m.language || 'Cinema')}</span>
                </div>
                <div class="rated-card-info" onclick="event.stopPropagation()">
                  <h4 class="liked-card-title" title="${title}">${title}</h4>
                  <p class="liked-card-meta">${m.year} · ${escapeHtml(genres)}</p>

                  <!-- Interactive 5-Star Re-rate Widget -->
                  <div class="rated-card-stars-section">
                    <div class="rated-card-stars-header">
                      <span class="rated-stars-title">Your Rating:</span>
                      <span class="rated-current-score" id="ratedScore-${m.id}">${currentRating} / 5 ★</span>
                    </div>
                    <div class="profile-star-row" id="profileStarsRow-${m.id}" data-movie-id="${m.id}" data-current="${currentRating}">
                      ${[1, 2, 3, 4, 5].map(starVal => `
                        <span class="profile-star ${starVal <= currentRating ? 'filled' : ''}"
                              data-star="${starVal}"
                              title="Re-change rating to ${starVal} Star${starVal > 1 ? 's' : ''}"
                              onmouseover="hoverProfileStars(${m.id}, ${starVal})"
                              onmouseout="resetProfileStars(${m.id})"
                              onclick="changeProfileRating(${m.id}, ${starVal}, event)">★</span>
                      `).join('')}
                    </div>
                    <span class="rated-hint-text">Click any star to re-change rating</span>
                  </div>

                  <div class="liked-card-actions">
                    ${trailerBtn}
                    <a href="/movie/${m.id}" class="liked-card-action-btn liked-btn-view">ℹ Details</a>
                    <button class="liked-card-action-btn liked-btn-remove" title="Remove Rating" onclick="changeProfileRating(${m.id}, 0, event)">✕ Clear</button>
                  </div>
                </div>
              </div>`;
          }).join('')}
        </div>`;
    }
  }
}

function hoverProfileStars(movieId, starVal) {
  const row = document.getElementById(`profileStarsRow-${movieId}`);
  if (!row) return;
  const stars = row.querySelectorAll('.profile-star');
  stars.forEach(s => {
    const val = parseInt(s.dataset.star || '0');
    s.classList.toggle('hover', val <= starVal);
  });
}

function resetProfileStars(movieId) {
  const row = document.getElementById(`profileStarsRow-${movieId}`);
  if (!row) return;
  const current = parseInt(row.dataset.current || '0');
  const stars = row.querySelectorAll('.profile-star');
  stars.forEach(s => {
    s.classList.remove('hover');
    const val = parseInt(s.dataset.star || '0');
    s.classList.toggle('filled', val <= current);
  });
}

async function changeProfileRating(movieId, rating, evt) {
  if (evt) evt.stopPropagation();
  const res = await apiFetch(`/api/rate/${movieId}`, 'POST', { rating });
  if (res?.success) {
    if (rating > 0) {
      // Update visual stars on card
      const row = document.getElementById(`profileStarsRow-${movieId}`);
      if (row) {
        row.dataset.current = rating;
        resetProfileStars(movieId);
      }
      const scoreLabel = document.getElementById(`ratedScore-${movieId}`);
      if (scoreLabel) scoreLabel.textContent = `${rating} / 5 ★`;
      showToast(`⭐ Rating updated to ${rating}★! Collaborative model recalculated.`);
    } else {
      // Rating removed
      showToast('Rating removed.');
      const cardEl = document.getElementById(`ratedCard-${movieId}`);
      if (cardEl) {
        cardEl.style.opacity = '0';
        cardEl.style.transform = 'scale(0.9)';
        setTimeout(() => {
          cardEl.remove();
          const remaining = document.querySelectorAll('.rated-movie-card');
          const countEl = document.getElementById('profileRatingsCount');
          const tabCountEl = document.getElementById('tabCountRated');
          if (countEl) countEl.textContent = remaining.length;
          if (tabCountEl) tabCountEl.textContent = remaining.length;

          if (!remaining.length) {
            const container = document.getElementById('profileRatedContainer');
            if (container) {
              container.innerHTML = `
                <div class="profile-empty-state">
                  <div class="profile-empty-icon">⭐</div>
                  <h4 class="profile-empty-title">No Rated Movies Yet</h4>
                  <p class="profile-empty-desc">You haven't rated any movies yet. Rate your favorite films (1 to 5 stars) to train our collaborative filtering ML model and get ultra-personalized recommendations!</p>
                  <button class="btn-profile-explore" onclick="closeModal('profileModal')">Explore &amp; Rate Movies</button>
                </div>`;
            }
          }
        }, 250);
      }
    }
  } else if (res?.error === 'Login required') {
    showToast('Please sign in to rate movies');
    openLoginModal();
  } else {
    showToast(res?.error || 'Failed to update rating');
  }
}

async function removeLikedMovie(movieId, btn) {
  const res = await apiFetch(`/api/like/${movieId}`, 'POST');
  if (res?.success) {
    showToast('Removed from your liked watchlist');
    const cardEl = document.getElementById(`likedCard-${movieId}`);
    if (cardEl) {
      cardEl.style.opacity = '0';
      cardEl.style.transform = 'scale(0.9)';
      setTimeout(() => {
        cardEl.remove();
        const remaining = document.querySelectorAll('.liked-movie-card');
        const likedCountEl = document.getElementById('profileLikedCount');
        const tabCountEl = document.getElementById('tabCountLiked');
        if (likedCountEl) likedCountEl.textContent = remaining.length;
        if (tabCountEl) tabCountEl.textContent = remaining.length;

        // Also update any matching heart buttons on page
        document.querySelectorAll(`.movie-card[data-id="${movieId}"] .card-like, #likeBtn`).forEach(b => {
          b.classList.remove('liked');
          b.textContent = '♡';
        });

        if (!remaining.length) {
          const container = document.getElementById('profileLikedContainer');
          if (container) {
            container.innerHTML = `
              <div class="profile-empty-state">
                <div class="profile-empty-icon">💔</div>
                <h4 class="profile-empty-title">No Liked Movies Yet</h4>
                <p class="profile-empty-desc">You haven't liked any movies yet. Explore recommendations or ask CineBot for ideas, and click the ♡ button to save them here!</p>
                <button class="btn-profile-explore" onclick="closeModal('profileModal')">Explore Movies</button>
              </div>`;
          }
        }
      }, 250);
    }
  } else if (res?.error) {
    showToast('Please sign in to update your watchlist');
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
    document.getElementById('statAccuracy').textContent = data.accuracy_score || '92.6%';
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

// ── Modal UI Helpers (with mobile body scroll-lock) ───────────────────────
function openModal(id)  {
  const m = document.getElementById(id);
  if (m) {
    m.classList.add('open');
    document.body.classList.add('modal-open');
  }
}
function closeModal(id) {
  const m = document.getElementById(id);
  if (m) {
    m.classList.remove('open');
    // If no other modal is open, remove modal-open from body
    const openModals = document.querySelectorAll('.modal-overlay.open');
    if (openModals.length === 0) {
      document.body.classList.remove('modal-open');
    }
  }
}

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
   CINEBOT AI MULTI-TURN CONVERSATIONAL CONTROLLER
════════════════════════════════════════════════════════════════════════════ */
let cinebotHistory = [];
let cinebotState = {};
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
  cinebotState = {};
  const body = document.getElementById('cinebotBody');
  if (body) {
    body.innerHTML = `
      <div class="cinebot-msg bot intro-msg">
        <div class="cinebot-msg-bubble">
          <p>👋 Chat cleared! I'm ready for new questions or recommendations.</p>
          <p>Ask for movies like <em>"I liked Interstellar"</em>, request specific genres/moods, or say <em>"Surprise Me"</em>!</p>
        </div>
      </div>`;
  }
  updateChatChips(["🌌 I liked Interstellar", "🍿 Surprise Me", "😂 Feel-Good Hits", "🧠 How ML Works"]);
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
  // Blockquotes > text
  formatted = formatted.replace(/&gt;\s*\*(.*?)\*/g, '<blockquote><em>$1</em></blockquote>');
  formatted = formatted.replace(/&gt;\s*(.*?)(?=<br>|<\/p>|$)/g, '<blockquote>$1</blockquote>');
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
      history: cinebotHistory,
      session_state: cinebotState
    });

    // Remove typing bubble
    const currentTyping = document.getElementById('cinebotTyping');
    if (currentTyping) currentTyping.remove();

    if (data && data.success) {
      // Update session state
      if (data.session_state) {
        cinebotState = data.session_state;
      }

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

      // Update suggestion chips if provided
      if (data.suggested_prompts && data.suggested_prompts.length > 0) {
        updateChatChips(data.suggested_prompts);
      }

      // Append to local history (limit to last 12 turns)
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
  
  const isMultiSet = movies.length > 1;

  const cards = movies.map((m, idx) => {
    const poster = m.poster || fallbackImg;
    const genres = (m.genres || '').replace(/\|/g, ' · ').slice(0, 26);
    const title = escapeHtml(m.title);
    const year = m.year || '';
    const rating = m.rating || '7.0';
    const trailerKey = m.trailer_key || '';
    const movieId = m.id;

    const trailerBtn = (m.has_trailer || m.trailer_key || m.trailer_url)
      ? `<button class="chat-action-btn chat-btn-trailer" title="Play HD Trailer" onclick="event.stopPropagation(); openTrailer('${trailerKey || movieId}', '${escapeJsStr(m.title)}')">▶ Trailer</button>`
      : `<button class="chat-action-btn chat-btn-trailer" title="Search Trailer" onclick="event.stopPropagation(); openTrailer('${movieId}', '${escapeJsStr(m.title)}')">▶ Trailer</button>`;

    const numBadge = isMultiSet ? `<span class="chat-card-num">#${idx + 1}</span>` : '';

    return `
      <div class="chat-movie-card" onclick="window.location='/movie/${movieId}'" title="View ${title} details">
        <div class="chat-card-poster-wrap">
          <img class="chat-card-poster" src="${poster}" alt="${title}" loading="lazy" onerror="this.onerror=null; this.src='${fallbackImg}'">
          ${numBadge}
        </div>
        <div class="chat-card-content">
          <div>
            <div class="chat-card-title">${title}</div>
            <div class="chat-card-meta">
              <span class="chat-card-rating">⭐ ${rating}/10</span>
              <span>· ${year}</span>
            </div>
            <div class="chat-card-genres">${escapeHtml(genres)}</div>
          </div>
          <div class="chat-card-actions">
            ${trailerBtn}
            <button class="chat-action-btn chat-btn-view" title="View Full Details" onclick="event.stopPropagation(); window.location='/movie/${movieId}'">ℹ Details</button>
          </div>
        </div>
      </div>`;
  }).join('');

  const showMoreBtn = isMultiSet ? `
    <div class="chat-show-more-wrap">
      <button class="chat-show-more-btn" onclick="event.stopPropagation(); sendChatPrompt('Give me more')">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" width="14" height="14"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        <span>Show More Movies</span>
      </button>
    </div>` : '';

  return `<div class="chat-cards-list">${cards}${showMoreBtn}</div>`;
}

function updateChatChips(prompts) {
  const chipsContainer = document.getElementById('cinebotChips');
  if (!chipsContainer || !prompts || !prompts.length) return;
  chipsContainer.innerHTML = prompts.map(p => `
    <button class="chat-chip" onclick="sendChatPrompt('${escapeJsStr(p)}')">${escapeHtml(p)}</button>
  `).join('');
}

// ════════════════════════════════════════════════════════════════════════════
// DYNAMIC SCREEN & DEVICE ADJUSTMENT CONTROLLER (DeviceManager)
// ════════════════════════════════════════════════════════════════════════════
const DeviceManager = {
  init() {
    this.update();
    window.addEventListener('resize', () => this.handleResize(), { passive: true });
    window.addEventListener('orientationchange', () => this.handleOrientationChange(), { passive: true });
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', () => this.updateViewportMetrics(), { passive: true });
    }
  },

  getDeviceType() {
    const w = window.innerWidth;
    if (w >= 1920) return 'ultrawide';
    if (w >= 1366) return 'desktop';
    if (w >= 1024) return 'laptop';
    if (w >= 641) return 'tablet';
    return 'mobile';
  },

  getAspectRatioCategory() {
    const ratio = window.innerWidth / Math.max(1, window.innerHeight);
    if (ratio >= 2.2) return 'ultrawide';
    if (ratio >= 1.6) return 'widescreen';
    if (ratio >= 1.2) return 'standard-landscape';
    if (ratio >= 0.8) return 'square';
    if (ratio >= 0.52) return 'standard-portrait';
    return 'ultratall-portrait';
  },

  update() {
    const root = document.documentElement;
    const w = window.innerWidth;
    const h = window.innerHeight;
    const isTouch = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);
    const orientation = w >= h ? 'landscape' : 'portrait';
    const deviceType = this.getDeviceType();
    const aspectCategory = this.getAspectRatioCategory();

    // Set CSS custom variables for dynamic viewport heights (iOS / Android browser toolbars)
    root.style.setProperty('--app-height', `${h}px`);
    root.style.setProperty('--vh', `${h * 0.01}px`);
    root.style.setProperty('--app-width', `${w}px`);
    root.style.setProperty('--device-aspect-ratio', `${(w / h).toFixed(2)}`);

    // Set semantic HTML data attributes
    root.setAttribute('data-device', deviceType);
    root.setAttribute('data-orientation', orientation);
    root.setAttribute('data-touch', isTouch ? 'true' : 'false');
    root.setAttribute('data-aspect-ratio', aspectCategory);
  },

  updateViewportMetrics() {
    if (!window.visualViewport) return;
    const vh = window.visualViewport.height;
    document.documentElement.style.setProperty('--visual-vh', `${vh}px`);
  },

  handleResize() {
    if (this._resizeTimeout) cancelAnimationFrame(this._resizeTimeout);
    this._resizeTimeout = requestAnimationFrame(() => {
      this.update();
      // If user expands screen to desktop size (>= 950px), ensure mobile drawer is closed
      if (window.innerWidth >= 950) {
        const drawer = document.getElementById('mobileNavDrawer');
        const toggle = document.getElementById('mobileNavToggle');
        if (drawer && drawer.classList.contains('open')) {
          drawer.classList.remove('open');
          if (toggle) toggle.classList.remove('open');
        }
      }
    });
  },

  handleOrientationChange() {
    setTimeout(() => {
      this.update();
      // Close mobile drawer on orientation change to prevent layout jumping
      const drawer = document.getElementById('mobileNavDrawer');
      const toggle = document.getElementById('mobileNavToggle');
      if (drawer && drawer.classList.contains('open')) {
        drawer.classList.remove('open');
        if (toggle) toggle.classList.remove('open');
      }
    }, 150);
  }
};

// ════════════════════════════════════════════════════════════════════════════
// HERO MULTI-MOVIE FEATURED SCROLL CAROUSEL CONTROLLER
// ════════════════════════════════════════════════════════════════════════════
const HeroSlider = {
  currentIndex: 0,
  slides: [],
  dots: [],
  autoPlayTimer: null,
  autoPlayInterval: 6500, // 6.5s per slide
  isPaused: false,
  touchStartX: 0,
  touchEndX: 0,

  init() {
    this.slides = Array.from(document.querySelectorAll('.hero-slide'));
    this.dots = Array.from(document.querySelectorAll('.hero-indicator-dot'));
    if (this.slides.length <= 1) return;

    const heroSection = document.getElementById('heroCarouselSection');
    if (heroSection) {
      // Pause autoplay on mouse enter / hover
      heroSection.addEventListener('mouseenter', () => this.pause());
      heroSection.addEventListener('mouseleave', () => this.resume());

      // Touch swipe support for mobile
      heroSection.addEventListener('touchstart', (e) => {
        if (e.changedTouches && e.changedTouches.length > 0) {
          this.touchStartX = e.changedTouches[0].screenX;
        }
      }, { passive: true });

      heroSection.addEventListener('touchend', (e) => {
        if (e.changedTouches && e.changedTouches.length > 0) {
          this.touchEndX = e.changedTouches[0].screenX;
          this.handleSwipe();
        }
      }, { passive: true });
    }

    // Keyboard navigation (Left / Right Arrow)
    document.addEventListener('keydown', (e) => {
      const hero = document.getElementById('heroCarouselSection');
      if (!hero) return;
      // Only trigger if user is not typing in an input
      if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')) {
        return;
      }
      if (e.key === 'ArrowLeft') {
        this.prev();
      } else if (e.key === 'ArrowRight') {
        this.next();
      }
    });

    this.startAutoPlay();
  },

  handleSwipe() {
    const deltaX = this.touchEndX - this.touchStartX;
    if (Math.abs(deltaX) > 40) {
      if (deltaX < 0) {
        this.next(); // Swiped left -> next slide
      } else {
        this.prev(); // Swiped right -> prev slide
      }
    }
  },

  goTo(index) {
    if (this.slides.length === 0) return;
    this.currentIndex = (index + this.slides.length) % this.slides.length;

    this.slides.forEach((slide, idx) => {
      const isActive = idx === this.currentIndex;
      slide.classList.toggle('active', isActive);
      slide.setAttribute('aria-hidden', isActive ? 'false' : 'true');
    });

    this.dots.forEach((dot, idx) => {
      dot.classList.toggle('active', idx === this.currentIndex);
    });

    this.resetTimer();
  },

  next() {
    this.goTo(this.currentIndex + 1);
  },

  prev() {
    this.goTo(this.currentIndex - 1);
  },

  startAutoPlay() {
    this.stopAutoPlay();
    this.autoPlayTimer = setInterval(() => {
      if (!this.isPaused) {
        this.next();
      }
    }, this.autoPlayInterval);
  },

  stopAutoPlay() {
    if (this.autoPlayTimer) {
      clearInterval(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
  },

  resetTimer() {
    this.startAutoPlay();
  },

  pause() {
    this.isPaused = true;
  },

  resume() {
    this.isPaused = false;
  }
};

// Initialize device manager and hero slider when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    DeviceManager.init();
    HeroSlider.init();
  });
} else {
  DeviceManager.init();
  HeroSlider.init();
}

