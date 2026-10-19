"""Figure toolkit shared by every course figure.

Two kinds of figures are produced here:

* computed plots (matplotlib axes with data), and
* conceptual diagrams drawn on a blank canvas with the primitives below
  (boxes, arrows, layered networks, matrices, tensor blocks, flows).

Both use the same fonts and palette as the slides (see theme.py). Figures are
saved as vector PDF (used by the slide builder), SVG and PNG.

Sizing rule: figures are placed on the slide at roughly 1:1 scale, so matplotlib
point sizes are the sizes the audience sees. Use FULL for a full-width figure
and HALF for the right-hand column of a split slide. Keep text >= 12 pt.

Canvas units are inches: canvas(11.8, 4.3) gives an 11.8 x 4.3 inch drawing
area whose x/y coordinates are measured in inches (1 unit = 72 pt).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'course-matplotlib'))
# Fixed timestamp in PDF/SVG metadata so regenerated figures are byte-identical.
os.environ.setdefault('SOURCE_DATE_EPOCH', '1788998400')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgb
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Circle, Polygon,
                                Rectangle, Ellipse, Wedge, Arc)
from matplotlib.lines import Line2D

from curriculum import theme
from curriculum.theme import (INK, INK_2, BLUE, TEAL, ORANGE, RED, PURPLE, GOLD, GREEN,
                              MUTED, FAINT, RULE, PANEL, BLUE_T, TEAL_T, ORANGE_T,
                              RED_T, GOLD_T, PURPLE_T, SERIES, TINTS, SKY, MIST)

ROOT = theme.ROOT
# COURSE_FIGURES_OUT redirects output (used to check reproducibility without
# touching the published figures).
OUT = Path(os.environ.get('COURSE_FIGURES_OUT', ROOT / 'curriculum' / 'figures'))
MANIFEST_DIR = OUT / 'manifest.d'
OUT.mkdir(parents=True, exist_ok=True)

theme.register_matplotlib()

# Short colour aliases used by the shared figure code.
B, T, O, R, G, M = BLUE, TEAL, ORANGE, RED, '#56687A', PURPLE
COLORS = [B, O, T, M, R, G]
FULL = (11.8, 4.3)     # full-width figure (inches)
WIDE = (11.8, 3.6)     # shorter full-width strip
HALF = (6.3, 4.9)      # right column of a split slide
SQUARE = (5.6, 5.0)

plt.rcParams.update({
    'font.family': [theme.FAMILY, 'DejaVu Sans'], 'font.size': 14,  # DejaVu supplies missing glyphs
    'axes.labelsize': 14, 'axes.titlesize': 15, 'axes.titleweight': 'semibold',
    'axes.titlepad': 10, 'axes.labelpad': 6,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'legend.frameon': False, 'legend.handlelength': 1.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9DAFC0', 'axes.linewidth': 1.0,
    'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.major.size': 4, 'ytick.major.size': 4,
    'grid.color': '#E1E8EF', 'grid.linewidth': .8,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'pdf.fonttype': 42, 'svg.fonttype': 'path', 'svg.hashsalt': 'course-figures', 'lines.linewidth': 2.4,
    'lines.markersize': 6, 'patch.linewidth': 1.2,
    'mathtext.fontset': 'custom', 'mathtext.rm': theme.FAMILY,
    'mathtext.it': f'{theme.FAMILY}:italic', 'mathtext.bf': f'{theme.FAMILY}:bold',
    'mathtext.sf': theme.FAMILY, 'mathtext.fallback': 'stixsans',
    'axes.prop_cycle': matplotlib.cycler(color=SERIES),
    'image.cmap': 'course_seq',
})

# ------------------------------------------------------------- colormaps
CMAP_SEQ = LinearSegmentedColormap.from_list('course_seq', ['#FFFFFF', '#CFE1F4', BLUE, INK])
CMAP_TEAL = LinearSegmentedColormap.from_list('course_teal', ['#FFFFFF', '#CBE8E8', TEAL, '#0B4A52'])
CMAP_ORANGE = LinearSegmentedColormap.from_list('course_orange', ['#FFFFFF', '#F6D3BC', ORANGE, '#6E2A08'])
CMAP_DIV = LinearSegmentedColormap.from_list('course_div', [ORANGE, '#F3C9AE', '#FFFFFF', '#BCD6F0', BLUE])
for _cm in (CMAP_SEQ, CMAP_TEAL, CMAP_ORANGE, CMAP_DIV):
    try:
        matplotlib.colormaps.register(_cm)
    except ValueError:
        pass
BG = ListedColormap([BLUE_T, ORANGE_T])                 # two-class regions
BG3 = ListedColormap([BLUE_T, ORANGE_T, TEAL_T])        # three-class regions
MARKERS = ['o', '^', 's', 'D', 'v', 'P', 'X']

RNG = np.random.default_rng(42)
_STYLE_SNAPSHOT = plt.rcParams.copy()


def reset_state(seed: int = 0) -> None:
    """Restore the course style and reseed shared random generators.

    make_figures.py calls this before every figure function, so a figure looks
    the same whether it is generated alone, with its module, or with all modules.
    """
    import random
    import sys as _sys
    plt.close('all')
    plt.rcParams.update(_STYLE_SNAPSHOT)
    RNG.bit_generator.state = np.random.default_rng(42).bit_generator.state
    np.random.seed(seed)
    random.seed(seed)
    if 'torch' in _sys.modules:
        _sys.modules['torch'].manual_seed(seed)

# --------------------------------------------------------------- saving
MANIFEST: dict[str, dict] = {}


def save(name: str, fig, source: str = 'Deterministic synthetic teaching example; seed 42 unless specified.',
         kind: str | None = None, tight: bool = True, pad: float = .12) -> None:
    """Write NAME.pdf/.svg/.png and record provenance.

    kind: 'computed' (calculated from data or an exact formula), 'schematic'
    (conceptual drawing), or 'data' (supplied course data). Recorded in the
    manifest so captions can be audited.
    """
    kw = dict(bbox_inches='tight', pad_inches=pad) if tight else {}
    fig.savefig(OUT / f'{name}.pdf', **kw)
    fig.savefig(OUT / f'{name}.svg', **kw)
    fig.savefig(OUT / f'{name}.png', dpi=200, **kw)
    plt.close(fig)
    entry = {'source': source, 'formats': ['pdf', 'svg', 'png']}
    if kind:
        entry['kind'] = kind
    MANIFEST[name] = entry
    print(name, flush=True)


def write_fragment(key: str) -> Path:
    """Merge this run's figures into curriculum/figures/manifest.d/KEY.json."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f'{key}.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update({k: v for k, v in MANIFEST.items()})
    path.write_text(json.dumps(dict(sorted(current.items())), indent=2))
    return path


# Figure registry: modules decorate figure functions with @register('ml1').
REGISTRY: dict[str, list] = {}


def register(key: str):
    def deco(fn):
        REGISTRY.setdefault(key, []).append(fn)
        return fn
    return deco


# ------------------------------------------------------ axes and plots
def sub(n: int = 1, height: float = FULL[1], width: float = FULL[0], **kw):
    """One row of n axes. Returns (fig, list_of_axes)."""
    fig, axs = plt.subplots(1, n, figsize=(width, height), layout='constrained', squeeze=False, **kw)
    return fig, list(axs[0])


def half(n: int = 1, height: float = HALF[1], width: float = HALF[0], rows: int = 1, **kw):
    """Axes sized for the figure column of a split slide."""
    fig, axs = plt.subplots(rows, n, figsize=(width, height), layout='constrained', squeeze=False, **kw)
    return fig, list(axs.ravel())


def grid_axes(rows: int, cols: int, width: float = FULL[0], height: float = FULL[1], **kw):
    fig, axs = plt.subplots(rows, cols, figsize=(width, height), layout='constrained', squeeze=False, **kw)
    return fig, axs


def clean(ax, x: str | None = None, y: str | None = None, grid: bool = True):
    if x:
        ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)
    if grid:
        ax.grid(alpha=1)
        ax.set_axisbelow(True)
    return ax


def points(ax, X, y, s: float = 24, alpha: float = .9, edge: str = 'white'):
    """Scatter labelled 2-D points; label -1 is drawn as grey noise crosses."""
    for k in np.unique(y):
        mask = y == k
        if k >= 0:
            ax.scatter(X[mask, 0], X[mask, 1], s=s, c=SERIES[int(k) % 6],
                       marker=MARKERS[int(k) % 6], edgecolors=edge, linewidths=.4,
                       alpha=alpha, zorder=3)
        else:
            ax.scatter(X[mask, 0], X[mask, 1], s=s * 1.1, c='#8C99A6', marker='x',
                       linewidths=1.2, alpha=alpha, zorder=3)


def regions(ax, model, xlim, ylim, n_classes: int = 2, res: int = 250, proba: bool = False):
    """Shade a fitted classifier's predicted regions on ax."""
    xx, yy = np.meshgrid(np.linspace(*xlim, res), np.linspace(*ylim, int(res * .75)))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if proba:
        zz = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap=CMAP_DIV.reversed(), alpha=.38)
        ax.contour(xx, yy, zz, levels=[.5], colors=[INK], linewidths=1.4)
    else:
        zz = model.predict(grid).reshape(xx.shape)
        cmap = BG if n_classes == 2 else BG3
        ax.contourf(xx, yy, zz, levels=np.arange(n_classes + 1) - .5, cmap=cmap)
        ax.contour(xx, yy, zz, levels=np.arange(n_classes) + .5, colors=['#7E90A3'], linewidths=.9)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def endlabel(ax, x, y, text: str, color: str, dx: float = 0, dy: float = 0, fs: float = 12,
             ha: str = 'left', va: str = 'center', weight: str = 'medium'):
    """Direct label for a line (preferred over legends)."""
    ax.annotate(text, (x, y), xytext=(4 + dx, dy), textcoords='offset points', color=color,
                fontsize=fs, ha=ha, va=va, fontweight=weight)


def note(ax, x, y, text: str, xytext=None, color: str = INK, fs: float = 12, arrow: bool = True,
         ha: str = 'left', va: str = 'center', rad: float = .2, textcoords: str = 'data'):
    """Annotation with a thin curved arrow."""
    props = dict(arrowstyle='-|>', color=color, lw=1.1, shrinkA=2, shrinkB=3,
                 connectionstyle=f'arc3,rad={rad}') if arrow else None
    ax.annotate(text, (x, y), xytext=xytext or (x, y), textcoords=textcoords, color=color,
                fontsize=fs, ha=ha, va=va, arrowprops=props)


# ------------------------------------------------------ diagram canvas
def canvas(w: float = FULL[0], h: float = FULL[1]):
    """Blank drawing area whose coordinates are inches."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_axis_off()
    ax.set_aspect('equal')
    return fig, ax


def tint(color: str, amount: float = .86) -> tuple:
    """Blend a colour towards white."""
    if color in TINTS and amount == .86:
        return to_rgb(TINTS[color])
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def shade(color: str, amount: float = .25) -> tuple:
    """Blend a colour towards black."""
    r, g, b = to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


class Anchor(dict):
    """Anchor points of a drawn element: .c .l .r .t .b plus corners."""
    def __getattr__(self, k):
        return self[k]


def _anchors(x, y, w, h):
    return Anchor(c=(x + w / 2, y + h / 2), l=(x, y + h / 2), r=(x + w, y + h / 2),
                  t=(x + w / 2, y + h), b=(x + w / 2, y), tl=(x, y + h), tr=(x + w, y + h),
                  bl=(x, y), br=(x + w, y), x=x, y=y, w=w, h=h)


def box(ax, x, y, w, h, text: str = '', color: str = BLUE, fill: str = 'tint', fs: float = 13,
        weight: str = 'medium', sub: str | None = None, sub_fs: float | None = None,
        radius: float = .08, lw: float = 1.5, text_color: str | None = None, ls: str = '-',
        ha: str = 'center', alpha: float = 1, z: float = 2, pad_x: float = .12):
    """Rounded box with a centred label (and optional smaller second line).

    fill: 'tint' (pale fill, coloured border), 'solid' (coloured fill, white
    text), 'white' (white fill, coloured border) or 'none' (outline only).
    """
    if fill == 'solid':
        fc, ec, tc = color, color, 'white'
    elif fill == 'white':
        fc, ec, tc = 'white', color, INK
    elif fill == 'none':
        fc, ec, tc = 'none', color, INK
    else:
        fc, ec, tc = tint(color), color, INK
    tc = text_color or tc
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                           fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=z)
    ax.add_patch(patch)
    tx = x + w / 2 if ha == 'center' else x + pad_x
    if text and sub:
        ax.text(tx, y + h * .60, text, ha=ha, va='center', fontsize=fs, color=tc,
                fontweight=weight, zorder=z + 1)
        ax.text(tx, y + h * .30, sub, ha=ha, va='center', fontsize=sub_fs or fs * .8,
                color=tc if fill == 'solid' else MUTED, zorder=z + 1)
    elif text:
        ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=fs, color=tc,
                fontweight=weight, zorder=z + 1, linespacing=1.25)
    return _anchors(x, y, w, h)


def arrow(ax, p, q, color: str = INK, lw: float = 1.7, style: str = '-|>', rad: float = 0,
          head: float = 14, shrink: float = 0, text: str | None = None, fs: float = 11.5,
          text_color: str | None = None, offset=(0, .14), ls: str = '-', z: float = 1.5,
          text_bg: bool = False):
    """Arrow from p to q; optional label at the midpoint."""
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=head, color=color, lw=lw,
                        shrinkA=shrink, shrinkB=shrink, connectionstyle=f'arc3,rad={rad}',
                        linestyle=ls, zorder=z)
    ax.add_patch(a)
    if text:
        mx, my = (p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1]
        kw = dict(bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none')) if text_bg else {}
        ax.text(mx, my, text, ha='center', va='center', fontsize=fs,
                color=text_color or color, zorder=z + .5, **kw)
    return a


def elbow(ax, p, q, color: str = INK, lw: float = 1.7, first: str = 'h', head: float = 14):
    """Right-angle connector (horizontal then vertical, or the reverse)."""
    mid = (q[0], p[1]) if first == 'h' else (p[0], q[1])
    ax.plot([p[0], mid[0]], [p[1], mid[1]], color=color, lw=lw, solid_capstyle='butt', zorder=1.5)
    arrow(ax, mid, q, color=color, lw=lw, head=head)


def label(ax, x, y, text: str, fs: float = 13, color: str = INK, weight: str = 'normal',
          ha: str = 'center', va: str = 'center', style: str = 'normal', z: float = 4, **kw):
    return ax.text(x, y, text, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va,
                   fontstyle=style, zorder=z, **kw)


def chip(ax, x, y, text: str, color: str = BLUE, fs: float = 11, ha: str = 'center',
         solid: bool = False, weight: str = 'semibold'):
    """Small rounded tag centred at (x, y)."""
    fc = color if solid else tint(color)
    tc = 'white' if solid else color
    return ax.text(x, y, text, fontsize=fs, color=tc, ha=ha, va='center', fontweight=weight,
                   zorder=5, bbox=dict(boxstyle='round,pad=0.32,rounding_size=0.45', fc=fc, ec='none'))


def node(ax, x, y, r: float = .18, color: str = BLUE, fill: str = 'tint', text: str = '',
         fs: float = 12, lw: float = 1.5, z: float = 3, text_color: str | None = None):
    fc = color if fill == 'solid' else ('white' if fill == 'white' else tint(color))
    ax.add_patch(Circle((x, y), r, fc=fc, ec=color, lw=lw, zorder=z))
    if text:
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, zorder=z + 1,
                color=text_color or ('white' if fill == 'solid' else INK), fontweight='medium')
    return (x, y)


def network(ax, sizes, x0, x1, y0, y1, r: float = .17, colors=None, fills=None,
            edge_color: str = '#8C9EB1', edge_lw: float = 1.0, labels=None, max_nodes: int = 7,
            texts=None, fs: float = 12):
    """Fully connected layered network. Returns node positions per layer.

    sizes: nodes per layer. Layers with more than max_nodes are drawn with an
    ellipsis. colors/fills: per-layer colour and fill style. labels: per-layer
    captions drawn below the layer. texts: optional per-layer list of node texts.
    """
    n = len(sizes)
    colors = colors or [BLUE] + [TEAL] * (n - 2) + [ORANGE]
    fills = fills or ['tint'] * n
    xs = np.linspace(x0, x1, n)
    pos = []
    for li, (size, x) in enumerate(zip(sizes, xs)):
        shown = min(size, max_nodes)
        ys = np.linspace(y1, y0, shown) if shown > 1 else np.array([(y0 + y1) / 2])
        layer = [(x, y) for y in ys]
        pos.append((layer, size > max_nodes))
    # Edges sit above container fills (boxes use zorder 2) and below the nodes (zorder 3).
    for (a, _), (b, _) in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=edge_color, lw=edge_lw, zorder=2.6)
    out = []
    for li, (layer, trunc) in enumerate(pos):
        pts = []
        for ni, (x, y) in enumerate(layer):
            if trunc and ni == len(layer) - 2:
                for dy in (-.11, 0, .11):  # vertical ellipsis drawn as dots (no font glyph needed)
                    ax.add_patch(Circle((x, y + dy), .028, fc=MUTED, ec='none', zorder=4))
                continue
            t = texts[li][ni] if texts and texts[li] and ni < len(texts[li]) else ''
            node(ax, x, y, r, colors[li], fills[li], t, fs=fs)
            pts.append((x, y))
        out.append(pts)
        if labels and labels[li]:
            ax.text(x if not layer else layer[0][0], y0 - r - .28, labels[li], ha='center', va='top',
                    fontsize=12.5, color=MUTED)
    return out


def matrix(ax, x, y_top, values, cell: float = .5, fmt: str = '{:g}', cmap=None, vmin=None,
           vmax=None, color: str | None = None, fs: float = 13, highlight=None,
           hl_color: str = ORANGE, edge: str = 'white', text: bool = True, title: str | None = None,
           title_fs: float = 13, lw: float = 1.2):
    """Draw a numeric grid with its top-left corner at (x, y_top).

    cmap colours cells by value; color gives a flat tint instead. highlight is
    a (row0, col0, rows, cols) rectangle outlined in hl_color. Returns anchors.
    """
    a = np.asarray(values, dtype=float)
    rows, cols = a.shape
    vmin = a.min() if vmin is None else vmin
    vmax = a.max() if vmax is None else vmax
    cm = plt.get_cmap(cmap) if cmap else None
    for i in range(rows):
        for j in range(cols):
            if cm is not None:
                t = 0 if vmax == vmin else (a[i, j] - vmin) / (vmax - vmin)
                fc = cm(.08 + .8 * t)
                dark = .08 + .8 * t > .55
            else:
                fc = tint(color or BLUE, .82)
                dark = False
            ax.add_patch(Rectangle((x + j * cell, y_top - (i + 1) * cell), cell, cell, fc=fc,
                                   ec=edge, lw=lw, zorder=2))
            if text:
                ax.text(x + (j + .5) * cell, y_top - (i + .5) * cell, fmt.format(a[i, j]),
                        ha='center', va='center', fontsize=fs, zorder=3,
                        color='white' if dark else INK)
    if highlight is not None:
        r0, c0, hr, hc = highlight
        ax.add_patch(Rectangle((x + c0 * cell, y_top - (r0 + hr) * cell), hc * cell, hr * cell,
                               fc='none', ec=hl_color, lw=3, zorder=4))
    if title:
        ax.text(x + cols * cell / 2, y_top + .22, title, ha='center', va='bottom',
                fontsize=title_fs, fontweight='semibold', color=INK)
    return _anchors(x, y_top - rows * cell, cols * cell, rows * cell)


def block3d(ax, x, y, w, h, d, color: str = BLUE, text: str = '', fs: float = 11.5,
            text_below: str | None = None, alpha: float = 1, lw: float = 1.1, dx_ratio: float = .55,
            dy_ratio: float = .45, z: float = 2):
    """Pseudo-3D block (a tensor / feature-map stack). (x, y) is the front
    face's lower-left corner; d is the depth drawn as an oblique offset."""
    dx, dy = d * dx_ratio, d * dy_ratio
    front = tint(color, .55)
    top = tint(color, .75)
    side = tint(color, .35)
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True,
                         fc=front, ec=shade(color, .2), lw=lw, zorder=z, alpha=alpha))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)],
                         closed=True, fc=top, ec=shade(color, .2), lw=lw, zorder=z, alpha=alpha))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=side, ec=shade(color, .2), lw=lw, zorder=z, alpha=alpha))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
                zorder=z + 1, rotation=90 if h > 2.2 * w else 0)
    if text_below:
        ax.text(x + (w + dx) / 2, y - .12, text_below, ha='center', va='top', fontsize=fs,
                color=MUTED, zorder=z + 1, linespacing=1.2)
    return _anchors(x, y, w + dx, h + dy)


def flow(ax, labels, x0, y, w, h, gap: float = .45, colors=None, fill: str = 'tint', fs: float = 13,
         subs=None, arrow_color: str = INK, weight: str = 'medium'):
    """A left-to-right row of boxes joined by arrows. Returns their anchors."""
    colors = colors or [BLUE] * len(labels)
    out = []
    for i, lab in enumerate(labels):
        s = subs[i] if subs else None
        a = box(ax, x0 + i * (w + gap), y, w, h, lab, colors[i], fill, fs, weight, sub=s)
        out.append(a)
        if i:
            arrow(ax, out[i - 1].r, a.l, color=arrow_color, shrink=2)
    return out


def bracket(ax, x0, x1, y, text: str = '', color: str = MUTED, fs: float = 12, tick: float = .1,
            below: bool = True, lw: float = 1.3):
    """Horizontal measurement bracket with a label."""
    s = -1 if below else 1
    ax.plot([x0, x0, x1, x1], [y - s * tick, y, y, y - s * tick], color=color, lw=lw, zorder=3)
    if text:
        ax.text((x0 + x1) / 2, y + s * .1, text, ha='center', va='bottom' if s > 0 else 'top',
                fontsize=fs, color=color, zorder=3)


def vbracket(ax, x, y0, y1, text: str = '', color: str = MUTED, fs: float = 12, tick: float = .1,
             right: bool = True, lw: float = 1.3):
    s = 1 if right else -1
    ax.plot([x - s * tick, x, x, x - s * tick], [y0, y0, y1, y1], color=color, lw=lw, zorder=3)
    if text:
        ax.text(x + s * .12, (y0 + y1) / 2, text, ha='left' if right else 'right', va='center',
                fontsize=fs, color=color, zorder=3)


def check(ax, x, y, s: float = .16, color: str = TEAL, lw: float = 3):
    ax.plot([x - s, x - s * .25, x + s], [y, y - s * .75, y + s * .8], color=color, lw=lw,
            solid_capstyle='round', solid_joinstyle='round', zorder=5)


def cross(ax, x, y, s: float = .14, color: str = ORANGE, lw: float = 3):
    ax.plot([x - s, x + s], [y - s, y + s], color=color, lw=lw, solid_capstyle='round', zorder=5)
    ax.plot([x - s, x + s], [y + s, y - s], color=color, lw=lw, solid_capstyle='round', zorder=5)


def gaussian_blob(ax, center, cov, color: str = BLUE, n_std=(1, 2), alpha: float = .18, lw: float = 1.6):
    """Draw covariance ellipses (1 and 2 standard deviations by default)."""
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    for k in n_std:
        w, h = 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0])
        ax.add_patch(Ellipse(center, w, h, angle=angle, fc=color if k == n_std[0] else 'none',
                             alpha=alpha if k == n_std[0] else 1, ec=color, lw=lw, zorder=2))


__all__ = [n for n in dir() if not n.startswith('_')]
