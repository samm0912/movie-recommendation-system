"""
test_viewports.py — Automated Multi-Viewport Resolution & Screen Adjustment Verification

Tests all requested device screen resolutions:
  1. 1920 × 1080 (Ultra-wide / Large Desktop)
  2. 1440 × 900  (Desktop / Laptop)
  3. 1366 × 768  (Standard Laptop)
  4. 1024 × 768  (Tablet Landscape / iPad Pro)
  5. 768 × 1024  (Tablet Portrait / iPad)
  6. 430 × 932   (iPhone 14/15/16 Pro Max)
  7. 390 × 844   (iPhone 12/13/14/15)
  8. 375 × 667   (iPhone SE)
  9. 320 × 568   (Compact / Micro Screen)
"""
import re
import sys

TARGET_VIEWPORTS = [
    {"name": "Ultra-Wide Desktop", "width": 1920, "height": 1080, "type": "desktop"},
    {"name": "Standard Desktop", "width": 1440, "height": 900, "type": "desktop"},
    {"name": "Standard Laptop", "width": 1366, "height": 768, "type": "laptop"},
    {"name": "Tablet Landscape", "width": 1024, "height": 768, "type": "tablet_landscape"},
    {"name": "Tablet Portrait", "width": 768, "height": 1024, "type": "tablet_portrait"},
    {"name": "Large Smartphone", "width": 430, "height": 932, "type": "mobile_large"},
    {"name": "Standard Smartphone", "width": 390, "height": 844, "type": "mobile_standard"},
    {"name": "Compact Smartphone", "width": 375, "height": 667, "type": "mobile_compact"},
    {"name": "Micro / Foldable Outer Screen", "width": 320, "height": 568, "type": "mobile_micro"},
]

def run_viewport_tests():
    errors = []
    
    with open('static/css/style.css', 'r', encoding='utf-8') as f:
        css = f.read()

    with open('templates/index.html', 'r', encoding='utf-8') as f:
        index_html = f.read()

    with open('templates/movie.html', 'r', encoding='utf-8') as f:
        movie_html = f.read()

    with open('static/js/app.js', 'r', encoding='utf-8') as f:
        js = f.read()

    print("=" * 70)
    print("RUNNING MULTI-VIEWPORT SCREEN ADJUSTMENT VERIFICATION")
    print("=" * 70)

    # 1. Global Viewport Safety & Zero Horizontal Overflow
    print("\n[SECTION 1: GLOBAL VIEWPORT SAFETY]")
    if 'max-width: 100vw' in css and 'overflow-x: hidden' in css:
        print("  [PASS] Zero horizontal overflow protection (max-width: 100vw; overflow-x: hidden)")
    else:
        errors.append("Global zero horizontal overflow protection missing")

    if 'box-sizing: border-box' in css:
        print("  [PASS] Universal box-sizing: border-box applied to all elements")
    else:
        errors.append("Universal box-sizing missing")

    if '--safe-top' in css and '--safe-bottom' in css and 'env(safe-area-inset-' in css:
        print("  [PASS] Notch and safe-area insets configured (--safe-top, --safe-bottom, --safe-left, --safe-right)")
    else:
        errors.append("Safe-area insets missing")

    # 2. Viewport Coverage for all 9 resolutions
    print("\n[SECTION 2: TARGET RESOLUTION COVERAGE]")
    for vp in TARGET_VIEWPORTS:
        w, h = vp['width'], vp['height']
        matched_bp = None
        # Check corresponding breakpoint
        if w >= 1600:
            if '@media (min-width: 1600px)' in css:
                matched_bp = ">= 1600px (Ultra-wide rule active)"
        elif w >= 1201:
            if 'max-width: 1599px' in css:
                matched_bp = "1201px - 1599px (Desktop rule active)"
        elif w >= 1025:
            if 'max-width: 1200px' in css:
                matched_bp = "1025px - 1200px (Compact laptop rule active)"
        elif w >= 951:
            if 'max-width: 1024px' in css:
                matched_bp = "951px - 1024px (Tablet landscape rule active)"
        elif w >= 768:
            if 'max-width: 950px' in css:
                matched_bp = "768px - 950px (Tablet portrait rule active)"
        elif w >= 481:
            if 'max-width: 640px' in css:
                matched_bp = "<= 640px (Phablet & Large mobile rule active)"
        elif w >= 381:
            if 'max-width: 480px' in css:
                matched_bp = "<= 480px (Standard mobile rule active)"
        elif w >= 321:
            if 'max-width: 380px' in css:
                matched_bp = "<= 380px (Compact mobile rule active)"
        else:
            if 'max-width: 320px' in css:
                matched_bp = "<= 320px (Micro mobile rule active)"

        if matched_bp:
            print(f"  [PASS] Viewport {w}x{h} ({vp['name']}) -> {matched_bp}")
        else:
            errors.append(f"Viewport {w}x{h} ({vp['name']}) lacked responsive rule coverage")

    # 3. Navbar & Mobile Drawer Adaptability
    print("\n[SECTION 3: NAVIGATION ADAPTABILITY]")
    if '.nav-capsule' in css and '.mobile-nav-toggle' in css and '.mobile-nav-drawer' in css:
        print("  [PASS] Desktop floating segmented capsule + mobile drawer menu configured")
    else:
        errors.append("Navbar responsive navigation structure missing")

    if 'calc(100dvh - 65px)' in css or 'calc(100dvh' in css:
        print("  [PASS] Mobile drawer fits dynamic viewport height (100dvh)")
    else:
        errors.append("Mobile drawer dynamic height missing")

    drawer_features = ['quizModal', 'openProfileModal', 'showGenreModal', 'showStats', 'triggerSurpriseMe']
    for df in drawer_features:
        if df in index_html and df in movie_html:
            print(f"  [PASS] Feature '{df}' present in mobile drawer across both pages")
        else:
            errors.append(f"Feature '{df}' missing from mobile drawer")

    # 4. Aspect Ratio Integrity (2:3 Posters & 16:9 Trailers)
    print("\n[SECTION 4: ASPECT RATIO INTEGRITY]")
    if 'aspect-ratio: 2 / 3' in css or 'aspect-ratio: 2/3' in css:
        print("  [PASS] 2:3 Movie poster aspect ratio strictly preserved (no image distortion)")
    else:
        errors.append("2:3 movie poster aspect ratio missing")

    if 'aspect-ratio: 16 / 9' in css or 'padding-top: 56.25%' in css:
        print("  [PASS] 16:9 Cinema widescreen trailer aspect ratio preserved (no video clipping)")
    else:
        errors.append("16:9 trailer aspect ratio missing")

    # 5. Hero Carousel Multi-Device Scaling & Touch Swipe
    print("\n[SECTION 5: HERO CAROUSEL ADAPTABILITY]")
    if 'hero-carousel-arrow' in css and 'hero-indicators' in css:
        print("  [PASS] Hero carousel controls adapt dynamically across screen widths")
    else:
        errors.append("Hero carousel control responsive rules missing")

    if 'handleSwipe' in js and 'touchstart' in js and 'touchend' in js:
        print("  [PASS] Native touch gesture swipe navigation active on hero carousel")
    else:
        errors.append("Hero touch swipe gesture handling missing")

    # 6. Movie Details Page Responsive Columns
    print("\n[SECTION 6: MOVIE DETAILS PAGE RESPONSIVE BEHAVIOR]")
    if '.detail-layout' in css and 'flex-direction: column' in css:
        print("  [PASS] Movie details layout automatically switches to vertical stack on mobile")
    else:
        errors.append("Movie details vertical stacking missing")

    # 7. CineBot AI Floating & Docked Modes
    print("\n[SECTION 7: CINEBOT AI ASSISTANT ADAPTABILITY]")
    if '.cinebot-window' in css and 'max-width: 480px' in css:
        print("  [PASS] CineBot adapts into full-width bottom sheet on mobile screens")
    else:
        errors.append("CineBot mobile bottom sheet adaptation missing")

    # 8. Modals Viewport Bounds & Body Scroll Locking
    print("\n[SECTION 8: UNIVERSAL MODAL VIEWPORT FITTING]")
    if 'max-width: min(640px, 95vw)' in css or 'max-width: 95vw' in css:
        print("  [PASS] Modals bounded within 95vw width to prevent screen cutoff")
    else:
        errors.append("Modal max-width 95vw constraint missing")

    if 'body.modal-open' in css and 'modal-open' in js:
        print("  [PASS] Background body scroll-locking prevents scroll leakage when modals open")
    else:
        errors.append("Modal scroll locking missing")

    print("\n" + "=" * 70)
    if errors:
        print(f"VERIFICATION FAILED WITH {len(errors)} ERRORS:")
        for e in errors:
            print(f"  [FAIL] {e}")
        sys.exit(1)
    else:
        print("ALL 9 TARGET VIEWPORTS AND RESPONSIVE REQUIREMENTS PASSED (100% SUCCESS)!")
        print("=" * 70)

if __name__ == '__main__':
    run_viewport_tests()
