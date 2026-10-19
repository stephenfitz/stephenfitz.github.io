#!/usr/bin/env python3
"""Summarise every module deck from its slides.md (titles, parts, labs, agenda).

    python3 curriculum/course_summary.py            # Markdown to stdout
    python3 curriculum/course_summary.py --json     # machine-readable

Used to keep README.md and INSTRUCTOR_GUIDE.md consistent with the decks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curriculum.markup import parse, plain  # noqa: E402
from curriculum.theme import ROOT  # noqa: E402


def summarise(path: Path) -> dict:
    slides = parse(path)
    agenda = next((s for s in slides if s['type'] == 'agenda'), None)
    rows = []
    if agenda and agenda.get('table'):
        rows = [r for r in agenda['table'] if r[0].strip().lower() not in ('time', 'start', 'when')]
    labs = [{'title': plain(s['title']), 'time': s.get('time', ''), 'notebook': ', '.join(s.get('notebooks', []))}
            for s in slides if s['type'] == 'lab']
    figures = sorted({f for s in slides for f in s['figures']})
    return {
        'path': str(path.parent.relative_to(ROOT)),
        'label': slides[0].get('label', ''),
        'title': plain(slides[0]['title']),
        'subtitle': plain(slides[0]['body'][0]['text']) if slides[0]['body'] else '',
        'slides': len(slides),
        'parts': [plain(s['title']) for s in slides if s['type'] == 'section'],
        'labs': labs,
        'exercises': sum(s['type'] == 'exercise' for s in slides),
        'figures': len(figures),
        'agenda': rows,
    }


def main() -> None:
    mods = [summarise(p) for p in sorted(ROOT.glob('*_course/*/slides.md'))]
    if '--json' in sys.argv:
        print(json.dumps(mods, indent=2))
        return
    for m in mods:
        print(f"## {m['label']}: {m['title']}\n")
        print(f"{m['slides']} slides · {len(m['parts'])} parts · {len(m['labs'])} labs · "
              f"{m['exercises']} exercises · {m['figures']} figures\n")
        if m['agenda']:
            print('| Time | Session | Kind |\n|---|---|---|')
            for r in m['agenda']:
                print('| ' + ' | '.join(plain(c) for c in r) + ' |')
            print()
        for lab in m['labs']:
            print(f"- {lab['title']} ({lab['time']}): `{lab['notebook']}`")
        print()


if __name__ == '__main__':
    main()
