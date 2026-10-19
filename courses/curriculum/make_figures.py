#!/usr/bin/env python3
"""Generate the course figures.

Figures are grouped by module in curriculum/figs/<key>.py (core = figures
shared by several modules). Each figure function is registered with
@register(KEY) and writes PDF/SVG/PNG files to curriculum/figures.

    python3 curriculum/make_figures.py              # everything + merged manifest
    python3 curriculum/make_figures.py --only ml3   # one module
    python3 curriculum/make_figures.py --only ml3 --match kmeans   # matching functions

Provenance for each figure is written to curriculum/figures/manifest.d/<key>.json
and merged into curriculum/figures/manifest.json.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curriculum import figlib  # noqa: E402

KEYS = ['core', 'layout', 'ml1', 'ml2', 'ml3', 'ml4', 'ml5', 'nn1', 'nn2', 'nn3', 'nn4', 'nn5']


def merge_manifest() -> Path:
    merged = {}
    for frag in sorted(figlib.MANIFEST_DIR.glob('*.json')):
        merged.update(json.loads(frag.read_text()))
    path = figlib.OUT / 'manifest.json'
    path.write_text(json.dumps(dict(sorted(merged.items())), indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='*', choices=KEYS, help='module keys to generate')
    parser.add_argument('--match', help='only run figure functions whose name contains this text')
    parser.add_argument('--list', action='store_true', help='list registered figure functions')
    parser.add_argument('--merge-only', action='store_true',
                        help='only merge manifest.d/*.json into manifest.json')
    args = parser.parse_args()
    if args.merge_only:
        print(f'Merged manifest: {merge_manifest()}')
        return
    keys = args.only or KEYS
    for key in keys:
        importlib.import_module(f'curriculum.figs.{key}')
    if args.list:
        for key in keys:
            for fn in figlib.REGISTRY.get(key, []):
                print(f'{key}: {fn.__name__}')
        return
    for key in keys:
        figlib.MANIFEST.clear()
        for fn in figlib.REGISTRY.get(key, []):
            if args.match and args.match not in fn.__name__:
                continue
            figlib.reset_state()
            fn()
        if figlib.MANIFEST:
            figlib.write_fragment(key)
    if not args.only:
        print(f'Merged manifest: {merge_manifest()}')


if __name__ == '__main__':
    main()
