#!/usr/bin/env python3
"""Browser checks for the offline HTML decks (optional; needs Playwright + Chromium).

    pip install playwright && python3 -m playwright install chromium
    python3 curriculum/check_html.py

For every module deck: exactly one active slide, keyboard navigation (arrows,
Home, End), fragment handling (#s12 and an invalid fragment), notes toggle,
overview grid, unique element ids, every slide rendered with non-zero size, no
JavaScript errors, and no network requests. Writes build/html_checks.json
and a screenshot of one slide per deck under build/html/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'build'


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('Playwright is not installed; skipping browser checks.')
        return 0
    decks = sorted(ROOT.glob('*_course/*/slides.html'))
    (OUT / 'html').mkdir(parents=True, exist_ok=True)
    results, failures = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        for path in decks:
            page = browser.new_page(viewport={'width': 1440, 'height': 810})
            errors, remote = [], []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('request', lambda r: remote.append(r.url) if r.url.startswith(('http:', 'https:')) else None)
            t0 = time.time()
            page.goto(path.as_uri(), wait_until='load')
            load_s = time.time() - t0
            n = page.locator('section.slide').count()
            active = lambda: page.locator('section.slide.active').get_attribute('id')  # noqa: E731
            checks = {}
            checks['one_active'] = page.locator('section.slide.active').count() == 1
            checks['starts_at_1'] = active() == 's1'
            page.keyboard.press('ArrowRight')
            checks['arrow_right'] = active() == 's2'
            page.keyboard.press('ArrowLeft')
            checks['arrow_left'] = active() == 's1'
            page.keyboard.press('End')
            checks['end'] = active() == f's{n}'
            page.keyboard.press('Home')
            checks['home'] = active() == 's1'
            page.keyboard.press('n')
            checks['notes_open'] = page.locator('section.slide.active aside').is_visible()
            page.keyboard.press('n')
            checks['notes_closed'] = not page.locator('section.slide.active aside').is_visible()
            page.keyboard.press('g')
            checks['grid'] = page.evaluate("document.body.classList.contains('grid')")
            page.keyboard.press('Escape')
            mid = max(1, n // 2)
            page.goto(path.as_uri() + f'#s{mid}', wait_until='load')
            checks['fragment'] = active() == f's{mid}'
            page.wait_for_timeout(250)
            page.screenshot(path=str(OUT / 'html' / f'{path.parent.name}_s{mid}.png'))
            page.goto(path.as_uri() + '#snot-a-slide', wait_until='load')
            checks['bad_fragment'] = active() == 's1'
            checks['unique_ids'] = page.evaluate(
                "(() => {const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);"
                "return ids.length === new Set(ids).size})()")
            sizes = page.evaluate(
                "(() => {const out=[]; for (const s of document.querySelectorAll('section.slide')) {"
                "s.classList.add('active'); const r=s.querySelector('svg').getBoundingClientRect();"
                "out.push(r.width*r.height); s.classList.remove('active');} return out})()")
            checks['all_slides_render'] = len(sizes) == n and min(sizes) > 0
            checks['no_js_errors'] = not errors
            checks['no_network'] = not remote
            ok = all(checks.values())
            if not ok:
                failures.append(path.parent.name)
            results.append({'module': path.parent.name, 'slides': n, 'load_seconds': round(load_s, 2),
                            'size_mb': round(path.stat().st_size / 1e6, 1), 'passed': ok, 'checks': checks,
                            'errors': errors[:5], 'remote_requests': remote[:5]})
            print(f"{path.parent.name:34s} {n:3d} slides  {results[-1]['size_mb']:5.1f} MB  "
                  f"load {load_s:4.2f}s  {'passed' if ok else 'FAILED ' + str([k for k, v in checks.items() if not v])}")
            page.close()
        browser.close()
    (OUT / 'html_checks.json').write_text(json.dumps(results, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
