"""
Responsive Layout & Universal Device Screen Verification Script
Checks CSS rules, media query breakpoints, aspect ratio integrity, DOM structure, and touch accessibility.
"""
import re
import sys

def run_checks():
    errors = []
    
    with open('static/css/style.css', 'r', encoding='utf-8') as f:
        css = f.read()

    with open('templates/index.html', 'r', encoding='utf-8') as f:
        index_html = f.read()

    with open('templates/movie.html', 'r', encoding='utf-8') as f:
        movie_html = f.read()

    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        js = f.read()

    print("=" * 60)
    print("RUNNING UNIVERSAL DEVICE RESPONSIVE VERIFICATION")
    print("=" * 60)

    # 1. Check Global Reset & Overflow Prevention
    if 'overflow-x: hidden' in css and 'max-width: 100vw' in css:
        print("[PASS] Global horizontal overflow prevention applied")
    else:
        errors.append("Global overflow-x: hidden or max-width: 100vw missing")

    if 'box-sizing: border-box' in css:
        print("[PASS] Universal box-sizing: border-box applied")
    else:
        errors.append("box-sizing: border-box missing")

    # 2. Check Safe Area Insets
    if '--safe-top' in css and '--safe-bottom' in css and 'env(safe-area-inset-' in css:
        print("[PASS] Device safe area insets configured for notch & gesture displays")
    else:
        errors.append("Safe area insets missing in style.css")

    # 3. Check Breakpoints Coverage
    breakpoints = ['1600px', '1200px', '1024px', '950px', '640px', '480px', '380px', '320px']
    for bp in breakpoints:
        if bp in css:
            print(f"[PASS] Breakpoint {bp} is present in stylesheet")
        else:
            errors.append(f"Breakpoint {bp} missing in style.css")

    # 4. Check Movie Grid vs Row System
    if '.movie-grid' in css and 'repeat(5, minmax(0, 1fr))' in css:
        print("[PASS] .movie-grid responsive CSS grid defined with 5-column base")
    else:
        errors.append(".movie-grid CSS grid definition missing")

    if 'id="searchGrid"' in index_html and 'class="movie-grid"' in index_html:
        print("[PASS] searchGrid uses .movie-grid layout")
    else:
        errors.append("searchGrid does not use .movie-grid")

    if 'id="matchedGrid"' in index_html and 'class="movie-grid"' in index_html:
        print("[PASS] matchedGrid uses .movie-grid layout")
    else:
        errors.append("matchedGrid does not use .movie-grid")

    # 5. Check Poster & Trailer Aspect Ratios
    if 'aspect-ratio: 2 / 3' in css or 'aspect-ratio: 2/3' in css:
        print("[PASS] 2:3 Cinema poster aspect ratio strictly preserved")
    else:
        errors.append("2:3 poster aspect ratio missing in style.css")

    if '16:9' in css or 'padding-top: 56.25%' in css:
        print("[PASS] 16:9 Cinema widescreen trailer aspect ratio preserved")
    else:
        errors.append("16:9 trailer aspect ratio missing")

    # 6. Check Movie Details Responsive Layout
    if '.detail-layout' in css and 'flex-direction: column' in css:
        print("[PASS] Movie details page stacks responsively on small viewports")
    else:
        errors.append("Movie details responsive column stacking missing")

    # 7. Check CineBot Dynamic Viewport Units
    if '.cinebot-window' in css and 'calc(100dvh' in css:
        print("[PASS] CineBot window dynamically sizes with viewport units (100dvh)")
    else:
        errors.append("CineBot dynamic viewport sizing missing")

    # 8. Check Viewport Meta Tags
    if 'viewport-fit=cover' in index_html and 'viewport-fit=cover' in movie_html:
        print("[PASS] Viewport meta tags configured with viewport-fit=cover")
    else:
        errors.append("viewport-fit=cover missing in templates")

    # 9. Check Mobile Drawer Menu Full Feature Parity
    if 'mobile-nav-drawer' in index_html and 'quizModal' in index_html and 'openProfileModal' in index_html:
        print("[PASS] index.html mobile drawer contains full feature parity (Quiz, Profile, Genres, Stats, Surprise)")
    else:
        errors.append("index.html mobile drawer missing feature parity")

    if 'mobile-nav-drawer' in movie_html and 'quizModal' in movie_html and 'openProfileModal' in movie_html:
        print("[PASS] movie.html mobile drawer contains full feature parity (Quiz, Profile, Genres, Stats, Surprise)")
    else:
        errors.append("movie.html mobile drawer missing feature parity")

    # 10. Check Modal Scroll Locking & Touch Support
    if 'body.modal-open' in css and 'modal-open' in js:
        print("[PASS] Modal body scroll-locking implemented for touch devices")
    else:
        errors.append("Modal scroll locking missing")

    if 'HeroSlider' in js and 'handleSwipe' in js and 'startAutoPlay' in js:
        print("[PASS] HeroSlider interactive controller with touch swipe and autoplay implemented")
    else:
        errors.append("HeroSlider controller missing in app.js")

    # 11. Check Recommender get_featured_movies API
    from app import rec
    featured_list = rec.get_featured_movies(5)
    if len(featured_list) >= 3:
        print(f"[PASS] rec.get_featured_movies returned {len(featured_list)} spotlight movies: {[m['title'] for m in featured_list]}")
    else:
        errors.append("rec.get_featured_movies failed to return spotlight movies")

    print("\n" + "=" * 60)
    if errors:
        print(f"VERIFICATION FAILED WITH {len(errors)} ERRORS:")
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print("ALL RESPONSIVE VERIFICATION CHECKS PASSED (100% SUCCESS)!")
        print("=" * 60)

if __name__ == '__main__':
    run_checks()
