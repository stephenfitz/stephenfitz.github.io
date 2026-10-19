#!/usr/bin/env python3
"""Scan course figures for drawing defects that are easy to miss at slide size.

Each figure function is run without writing files, and the drawn figure is
checked for:

  hidden   a line, arrow, marker layer or label mostly covered by an opaque
           shape drawn above it (for example network edges under a box fill)
  overlap  two free text labels overlapping each other
  ticks    neighbouring tick labels colliding
  clipped  a shape cut off at the edge of a diagram canvas

Findings are candidates to inspect at full size, not verdicts: some overlaps
are intentional (a line passing under a box it connects to).

    python3 curriculum/check_figures.py                 # every module
    python3 curriculum/check_figures.py --only nn1 ml3  # selected modules
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['COURSE_FIGURES_OUT'] = tempfile.mkdtemp(prefix='course-figure-check-')

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from curriculum import figlib  # noqa: E402

KEYS = ['core', 'layout', 'ml1', 'ml2', 'ml3', 'ml4', 'ml5', 'nn1', 'nn2', 'nn3', 'nn4', 'nn5']
RESULTS: dict[str, list] = {}


def _draw_order(ax):
    arts = [a for a in ax.get_children() if a.get_visible() and a is not ax.patch]
    return sorted(enumerate(arts), key=lambda t: (t[1].get_zorder(), t[0]))


def _opaque(p) -> bool:
    if not isinstance(p, Patch) or isinstance(p, FancyArrowPatch) or not p.get_fill():
        return False
    fc = p.get_facecolor()
    alpha = fc[3] if p.get_alpha() is None else fc[3] * p.get_alpha()
    return alpha > .35


def _samples(a, renderer) -> np.ndarray:
    """Display-space sample points along an artist (middle 70% of lines)."""
    try:
        if isinstance(a, Line2D):
            xy = a.get_transform().transform(a.get_xydata())
        elif isinstance(a, FancyArrowPatch):
            xy = a.get_path().vertices
        elif isinstance(a, LineCollection):
            segs = [s for s in a.get_segments() if len(s)]
            if not segs:
                return np.empty((0, 2))
            xy = np.vstack([a.get_transform().transform(np.asarray(s)) for s in segs])
        elif isinstance(a, PathCollection):
            offs = a.get_offsets()
            if offs is None or len(offs) == 0:
                return np.empty((0, 2))
            xy = a.get_offset_transform().transform(np.asarray(offs))
            return xy[np.isfinite(xy).all(1)][:: max(1, len(xy) // 60)]
        elif isinstance(a, Text):
            if not a.get_text().strip():
                return np.empty((0, 2))
            bb = a.get_window_extent(renderer)
            return np.array([(x, y) for x in np.linspace(bb.x0 + 1, bb.x1 - 1, 5)
                             for y in np.linspace(bb.y0 + 1, bb.y1 - 1, 3)])
        else:
            return np.empty((0, 2))
    except Exception:
        return np.empty((0, 2))
    xy = np.asarray(xy, float)
    xy = xy[np.isfinite(xy).all(1)]
    if len(xy) < 2:
        return xy
    pts = [xy[0]]
    for p0, p1 in zip(xy[:-1], xy[1:]):
        n = int(min(40, max(1, np.hypot(*(p1 - p0)) // 3)))
        pts.extend(p0 + t * (p1 - p0) for t in np.linspace(0, 1, n + 1)[1:])
    pts = np.array(pts)
    arc = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if arc[-1] > 0:
        keep = (arc >= .15 * arc[-1]) & (arc <= .85 * arc[-1])
        if keep.sum() >= 2:
            pts = pts[keep]
    return pts[:: max(1, len(pts) // 80)]


def audit(fig) -> list:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues, texts = [], []
    for ax in fig.get_axes():
        order = _draw_order(ax)
        rank = {id(a): k for k, (_, a) in enumerate(order)}
        fills = []
        for _, a in order:
            if _opaque(a):
                try:
                    fills.append((rank[id(a)], a, a.get_window_extent(renderer)))
                except Exception:
                    pass
        for _, a in order:
            if not isinstance(a, (Line2D, FancyArrowPatch, LineCollection, PathCollection, Text)):
                continue
            if isinstance(a, Text) and a in (ax.title, ax._left_title, ax._right_title):
                continue
            pts = _samples(a, renderer)
            above = [(bb, p) for rk, p, bb in fills if rk > rank[id(a)]]
            if len(pts) == 0 or not above:
                continue
            covered = sum(any(bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1 and p.contains_point((x, y))
                              for bb, p in above) for x, y in pts)
            if covered / len(pts) >= .3:
                desc = a.get_text()[:40] if isinstance(a, Text) else type(a).__name__
                issues.append(('hidden', f'{desc} ({covered / len(pts):.0%} covered)'))
        if not ax.axison:
            abb = ax.get_window_extent(renderer)
            for a in ax.get_children():
                if (a is ax.patch or type(a).__name__ == 'Spine' or not a.get_visible() or not a.get_clip_on()
                        or not isinstance(a, (Patch, Line2D, PathCollection, LineCollection))):
                    continue
                try:
                    bb = a.get_window_extent(renderer)
                except Exception:
                    continue
                if bb.width > 0 and bb.height > 0 and np.isfinite([bb.x0, bb.x1, bb.y0, bb.y1]).all():
                    out = max(abb.x0 - bb.x0, bb.x1 - abb.x1, abb.y0 - bb.y0, bb.y1 - abb.y1)
                    if out > 3:
                        issues.append(('clipped', f'{type(a).__name__} {out:.0f} px outside the canvas'))
        else:
            for axis in (ax.xaxis, ax.yaxis):
                labs = [t for t in axis.get_ticklabels() if t.get_visible() and t.get_text().strip()]
                bbs = [t.get_window_extent(renderer) for t in labs]
                for k in range(len(bbs) - 1):
                    b1, b2 = bbs[k], bbs[k + 1]
                    if min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > 1 and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > 1:
                        issues.append(('ticks', f'{labs[k].get_text()!r} / {labs[k + 1].get_text()!r}'))
        texts += [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
    texts += [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    boxes = []
    for t in texts:
        try:
            boxes.append((t, t.get_window_extent(renderer)))
        except Exception:
            pass
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (t1, b1), (t2, b2) = boxes[i], boxes[j]
            ix, iy = min(b1.x1, b2.x1) - max(b1.x0, b2.x0), min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            small = min(b1.width * b1.height, b2.width * b2.height)
            if ix > 1.5 and iy > 1.5 and small > 0 and ix * iy / small > .12:
                issues.append(('overlap', f'{t1.get_text()[:30]!r} / {t2.get_text()[:30]!r}'))
    return issues


def _checked_save(name, fig, *args, **kwargs):
    try:
        RESULTS[name] = audit(fig)
    except Exception as e:  # keep scanning
        RESULTS[name] = [('error', repr(e)[:160])]
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='*', choices=KEYS)
    parser.add_argument('--json', type=Path, help='also write the findings to this JSON file')
    args = parser.parse_args()
    figlib.save = _checked_save
    for key in args.only or KEYS:
        mod = importlib.import_module(f'curriculum.figs.{key}')
        if key == 'core':
            mod._save = _checked_save
        elif hasattr(mod, 'save'):
            mod.save = _checked_save
        for fn in figlib.REGISTRY.get(key, []):
            figlib.reset_state()
            try:
                fn()
            except Exception as e:
                RESULTS[f'{key}:{fn.__name__}'] = [('error', repr(e)[:160])]
    flagged = {k: v for k, v in RESULTS.items() if v}
    for name, found in sorted(flagged.items()):
        print(name)
        for kind, detail in found[:8]:
            print(f'  {kind:8s} {detail}')
        if len(found) > 8:
            print(f'  ... {len(found) - 8} more')
    print(f'{len(RESULTS)} figures scanned, {len(flagged)} with findings to inspect')
    if args.json:
        args.json.write_text(json.dumps(RESULTS, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
