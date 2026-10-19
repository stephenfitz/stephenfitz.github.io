"""PDF layout engine for the course decks.

Each slide is drawn with ReportLab on a 960 x 540 pt page. Figures (vector PDF
from curriculum/figures) and display equations (matplotlib mathtext) are
composited afterwards with PyMuPDF. Every layout measures its content before
drawing and shrinks text within fixed limits; content that still does not fit
raises SlideError, so a successful build has no overflowing text.
"""
from __future__ import annotations

import io
import math
import re
from functools import lru_cache

import fitz
import numpy as np
from matplotlib import mathtext, font_manager
import matplotlib
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Paragraph, Table, TableStyle

from curriculum import theme
from curriculum.theme import (INK, INK_2, BLUE, TEAL, ORANGE, RED, PURPLE, GOLD, GREEN, MUTED,
                              FAINT, RULE, PANEL, BLUE_T, TEAL_T, ORANGE_T, RED_T, GOLD_T,
                              PURPLE_T, SKY, MIST, TINTS)
from curriculum.markup import rich, plain, SlideError

theme.register_reportlab()
rl_config.invariant = 1
theme.register_matplotlib()

W, H = 960, 540
MX = 54
CW = W - 2 * MX
BOTTOM = 50
ASSETS = theme.ROOT / 'curriculum' / 'figures'
EQ_FONTSET = 'cm'

TYPE_ACCENT = {'exercise': TEAL, 'debrief': PURPLE, 'lab': ORANGE, 'reference': MUTED,
               'recap': TEAL, 'objectives': BLUE, 'agenda': BLUE}
TYPE_KICKER = {'exercise': 'Practice', 'debrief': 'Discussion', 'lab': 'Laboratory',
               'reference': 'Further study', 'recap': 'Key takeaways',
               'objectives': 'Learning outcomes', 'agenda': "Today's plan"}
TAG_COLORS = {'good': TEAL, 'bad': ORANGE, 'warn': GOLD, 'blue': BLUE, 'teal': TEAL,
              'orange': ORANGE, 'purple': PURPLE, 'red': RED, 'gold': GOLD, 'ink': INK,
              'green': GREEN, '': BLUE}
KIND_COLORS = [(('break', 'lunch', 'coffee'), FAINT), (('lab', 'hands-on', 'workshop', 'project work'), ORANGE),
               (('exercise', 'practice', 'discussion', 'debrief', 'review', 'quiz', 'check'), TEAL),
               (('assessment', 'capstone', 'presentation', 'showcase'), PURPLE),
               (('welcome', 'opening', 'wrap', 'close', 'intro'), INK)]


def C(value):
    return colors.HexColor(value) if isinstance(value, str) else value


def blend(hexcolor: str, amount: float) -> colors.Color:
    """Mix a colour with white (amount 0 = colour, 1 = white)."""
    c = colors.HexColor(hexcolor)
    return colors.Color(c.red + (1 - c.red) * amount, c.green + (1 - c.green) * amount,
                        c.blue + (1 - c.blue) * amount)


def tint_of(color: str) -> colors.Color:
    return C(TINTS[color]) if color in TINTS else blend(color, .88)


# ------------------------------------------------------------ text primitives
@lru_cache(maxsize=512)
def _style(size, font, color, leading, align):
    return ParagraphStyle('s', fontName=font, fontSize=size, leading=leading, textColor=C(color),
                          alignment=align, splitLongWords=False)


def para(text, size, font='Sans', color=INK, leading=None, align=0):
    lead = round(leading or size * 1.28, 2)
    return Paragraph(rich(text), _style(size, font, color, lead, align))


def para_h(text, width, size, **kw) -> float:
    return para(text, size, **kw).wrap(width, 10000)[1]


def draw_para(c, text, x, top, width, size, **kw) -> float:
    p = para(text, size, **kw)
    h = p.wrap(width, 10000)[1]
    p.drawOn(c, x, top - h)
    return top - h


def text_w(text, font, size) -> float:
    return pdfmetrics.stringWidth(text, font, size)


def spaced(c, text, x, y, font, size, color, space=1.0, align='left'):
    """Letter-spaced single line (kickers, labels)."""
    width = text_w(text, font, size) + space * max(len(text) - 1, 0)
    if align == 'right':
        x -= width
    elif align == 'center':
        x -= width / 2
    c.saveState()
    t = c.beginText(x, y)
    t.setFont(font, size)
    t.setCharSpace(space)
    t.setFillColor(C(color))
    t.textOut(text)
    t.setCharSpace(0)
    c.drawText(t)
    c.restoreState()
    return width


def rrect(c, x, y, w, h, r=8, fill=None, stroke=None, lw=1):
    c.saveState()
    if fill is not None:
        c.setFillColor(C(fill))
    if stroke is not None:
        c.setStrokeColor(C(stroke))
        c.setLineWidth(lw)
    c.roundRect(x, y, w, h, r, fill=int(fill is not None), stroke=int(stroke is not None))
    c.restoreState()


# ---------------------------------------------------------------- icons
def icon_check(c, x, y, s=7, color=TEAL, lw=2.2):
    c.saveState()
    c.setStrokeColor(C(color))
    c.setLineWidth(lw)
    c.setLineCap(1)
    c.setLineJoin(1)
    p = c.beginPath()
    p.moveTo(x - s, y)
    p.lineTo(x - s * .25, y - s * .7)
    p.lineTo(x + s, y + s * .75)
    c.drawPath(p, stroke=1, fill=0)
    c.restoreState()


def icon_cross(c, x, y, s=6, color=ORANGE, lw=2.2):
    c.saveState()
    c.setStrokeColor(C(color))
    c.setLineWidth(lw)
    c.setLineCap(1)
    c.line(x - s, y - s, x + s, y + s)
    c.line(x - s, y + s, x + s, y - s)
    c.restoreState()


def icon_clock(c, x, y, r=6.5, color=TEAL):
    c.saveState()
    c.setStrokeColor(C(color))
    c.setLineWidth(1.5)
    c.circle(x, y, r, stroke=1, fill=0)
    c.setLineCap(1)
    c.line(x, y, x, y + r * .62)
    c.line(x, y, x + r * .5, y)
    c.restoreState()


def icon_people(c, x, y, s=6.5, color=TEAL):
    c.saveState()
    c.setFillColor(C(color))
    for dx in (-s * .55, s * .55):
        c.circle(x + dx, y + s * .35, s * .38, stroke=0, fill=1)
        c.wedge(x + dx - s * .62, y - s * .95, x + dx + s * .62, y + s * .2, 0, 180, stroke=0, fill=1)
    c.restoreState()


def icon_book(c, x, y, s=6.5, color=ORANGE):
    c.saveState()
    c.setStrokeColor(C(color))
    c.setLineWidth(1.4)
    c.rect(x - s * .8, y - s, s * 1.6, s * 2, stroke=1, fill=0)
    for k in (.35, 0, -.35):
        c.line(x - s * .4, y + k * s, x + s * .45, y + k * s)
    c.restoreState()


def chip(c, text, x, y, color=TEAL, icon=None, size=11, align='left', solid=False):
    """Rounded tag with optional icon; (x, y) is the left (or right) centre."""
    pad = 9
    iw = 16 if icon else 0
    tw = text_w(text, 'SansSemi', size)
    w = tw + 2 * pad + iw
    h = size + 10
    if align == 'right':
        x -= w
    rrect(c, x, y - h / 2, w, h, r=h / 2, fill=color if solid else tint_of(color))
    if icon:
        {'clock': icon_clock, 'people': icon_people, 'book': icon_book}[icon](
            c, x + pad + 5, y, color='#FFFFFF' if solid else color)
    c.setFillColor(C('#FFFFFF' if solid else color))
    c.setFont('SansSemi', size)
    c.drawString(x + pad + iw, y - size * .35, text)
    return w


# ------------------------------------------------------------ math accents
ACCENT_CHARS = {'hat': 'ˆ', 'bar': '¯', 'overline': '¯', 'tilde': '˜', 'dot': '˙', 'vec': '→'}


@lru_cache(maxsize=16)
def _font_tables(rl_name: str):
    """fontTools glyph data for a registered ReportLab font name."""
    from fontTools.ttLib import TTFont as FTFont
    if rl_name in theme.FONTS:
        path = theme.font_path(rl_name)
    else:
        path = font_manager.findfont(font_manager.FontProperties(
            family='DejaVu Sans', weight='bold' if 'Bold' in rl_name else 'normal'))
    f = FTFont(str(path))
    return f.getBestCmap(), f['glyf'], f['head'].unitsPerEm, f['post'].italicAngle


def _bbox(rl_name, ch):
    cmap, glyf, upem, _ = _font_tables(rl_name)
    g = glyf[cmap[ord(ch)]] if ord(ch) in cmap else None
    if g is None or not hasattr(g, 'yMax'):
        return None
    return g.xMin / upem, g.xMax / upem, g.yMin / upem, g.yMax / upem


def draw_accent(canv, kind, label):
    """onDraw callback: draw a math accent centred over the glyph just drawn.

    ReportLab has no mark positioning, so combining characters would land
    beside the letter. The markup layer emits <onDraw name="_accent"> after an
    accented letter; here the accent glyph is placed from the fonts' outlines.
    """
    info = canv._curr_tx_info
    tx, x, y = info['tx'], info['cur_x'], info['cur_y']
    accent, letter = label.split(' ', 1)
    font = tx._fontname or 'SansItalic'
    size = tx._fontsize or 14
    if ord(letter) not in _font_tables(font)[0]:
        font = 'Fallback'
    glyph = ACCENT_CHARS.get(accent, 'ˆ')
    gfont = font if ord(glyph) in _font_tables(font)[0] else ('Sans' if ord(glyph) in theme.glyphs('Sans') else 'Fallback')
    lb, gb = _bbox(font, letter), _bbox(gfont, glyph)
    adv = pdfmetrics.stringWidth(letter, font, size) / size
    if lb is None or gb is None:
        return
    slope = math.tan(math.radians(-_font_tables(font)[3]))
    gap = .07
    lift = lb[3] + gap - gb[2]                       # accent bottom sits just above the letter top
    letter_cx = (lb[0] + lb[1]) / 2 - adv            # letter centre relative to the pen (em)
    acc_mid_y = lift + (gb[2] + gb[3]) / 2
    letter_mid_y = (lb[2] + lb[3]) / 2
    cx = letter_cx + slope * (acc_mid_y - letter_mid_y)
    gx = cx - (gb[0] + gb[1]) / 2                    # accent origin so its bbox centre is at cx
    canv.saveState()
    canv.setFillColor(info['xs'].textColor or C(INK))
    canv.setFont(gfont, size)
    canv.drawString(x + gx * size, y + lift * size, glyph)
    canv.restoreState()


# ------------------------------------------------------------ item lists
def split_title(text: str):
    m = re.match(r'^\*\*(.+?)\*\*[\s:—–-]*(.*)$', text)
    if m:
        return m.group(1), m.group(2)
    for sep in (' — ', ' – ', ': '):
        if sep in text:
            a, b = text.split(sep, 1)
            return a, b
    return text, ''


def item_geometry(kind, size, mode):
    """(text_indent, text_size) for a list item."""
    if kind == 'sub':
        return size * 2.15, size * .86
    if kind == 'text':
        return 0, size
    if mode in ('q', 'badge'):
        return size * 1.85, size
    if mode in ('check', 'task'):
        return size * 1.55, size
    return size * 1.1, size


def items_height(items, width, size, mode='bullet', gap=None) -> float:
    h = 0
    for k, it in enumerate(items):
        ind, ts = item_geometry(it['kind'], size, mode)
        h += para_h(it['text'], width - ind, ts)
        if k < len(items) - 1:
            h += item_gap(it, items[k + 1], size, gap)
    return h


def item_gap(a, b, size, gap=None):
    if b['kind'] == 'sub':
        return size * .22
    if gap is not None:
        return gap
    return size * (.62 if a['kind'] == 'text' else .5)


def draw_items(c, items, x, top, width, size, mode='bullet', accent=BLUE, color=INK, gap=None) -> float:
    y = top
    qn = 0
    for k, it in enumerate(items):
        kind = it['kind']
        ind, ts = item_geometry(kind, size, mode)
        cy = y - ts * .70  # visual centre of the first line
        if kind == 'bullet' or (kind == 'num' and mode in ('bullet', 'check', 'task')):
            if mode == 'check':
                icon_check(c, x + size * .5, cy + 1, s=size * .33, color=accent, lw=size * .11)
            elif mode == 'task':
                c.saveState()
                c.setStrokeColor(C(accent))
                c.setLineWidth(1.5)
                s = size * .62
                c.roundRect(x + 2, cy - s / 2, s, s, 2.5, stroke=1, fill=0)
                c.restoreState()
            elif mode in ('q', 'badge'):
                qn += 1
                badge(c, x, cy, qn if kind == 'bullet' else it['n'], size, accent,
                      square=(mode == 'q'))
            elif kind == 'num':
                c.setFillColor(C(accent))
                c.setFont('SansSemi', size * .86)
                c.drawRightString(x + size * .85, y - ts, f"{it['n']}.")
            else:
                c.setFillColor(C(accent))
                s = size * .27
                c.rect(x + size * .12, cy - s / 2, s, s, fill=1, stroke=0)
        elif kind == 'num':
            qn = it['n']
            badge(c, x, cy, it['n'], size, accent, square=(mode == 'q'))
        elif kind == 'sub':
            c.setStrokeColor(C(FAINT))
            c.setLineWidth(1.4)
            c.line(x + size * 1.18, cy + 1, x + size * 1.62, cy + 1)
        y = draw_para(c, it['text'], x + ind, y, width - ind, ts,
                      color=MUTED if kind == 'sub' else color)
        if k < len(items) - 1:
            y -= item_gap(it, items[k + 1], size, gap)
    return y


def badge(c, x, cy, n, size, color, square=False):
    d = size * 1.3
    if square:
        rrect(c, x, cy - d / 2, d, d, r=4, fill=color)
    else:
        c.setFillColor(C(color))
        c.circle(x + d / 2, cy, d / 2, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont('SansSemi', size * .78)
    c.drawCentredString(x + d / 2, cy - size * .28, str(n))


# ------------------------------------------------------------ equations
@lru_cache(maxsize=256)
def equation_pdf(expr: str) -> bytes:
    with matplotlib.rc_context({'mathtext.fontset': EQ_FONTSET, 'savefig.transparent': True}):
        buf = io.BytesIO()
        mathtext.math_to_image('$' + expr + '$', buf, prop=font_manager.FontProperties(size=30),
                               format='pdf', color=INK)
    return buf.getvalue()


def equation_size(expr: str):
    doc = fitz.open(stream=equation_pdf(expr), filetype='pdf')
    r = doc[0].rect
    doc.close()
    return r.width, r.height


def equation_block_h(eqs, width, size):
    """Height of an equation panel holding eqs at a nominal font size."""
    total = 0
    for e in eqs:
        w, h = equation_size(e)
        s = min(size / 30, (width - 36) / w)
        total += h * s + 8
    return total + 22


def draw_equations(c, eqs, x, top, width, size, overlays, page):
    h = equation_block_h(eqs, width, size)
    rrect(c, x, top - h, width, h, r=8, fill=PANEL)
    c.setFillColor(C(BLUE))
    c.rect(x, top - h, 3.5, h, fill=1, stroke=0)
    y = top - 11
    for e in eqs:
        w, eh = equation_size(e)
        s = min(size / 30, (width - 36) / w)
        rw, rh = w * s, eh * s
        rx = x + (width - rw) / 2
        overlays.append({'page': page, 'kind': 'eq', 'value': e,
                         'rect': fitz.Rect(rx, H - y, rx + rw, H - y + rh)})
        y -= rh + 8
    return top - h


# ------------------------------------------------------------ callouts
def callout_h(text, width, size=16):
    return para_h(text, width - 44, size) + 26


def draw_callout(c, text, x, bottom, width, color=BLUE, size=16):
    h = callout_h(text, width, size)
    rrect(c, x, bottom, width, h, r=8, fill=tint_of(color))
    c.setFillColor(C(color))
    c.roundRect(x, bottom, 5, h, 2, fill=1, stroke=0)
    # small key glyph: a filled diamond for callouts, a triangle for warnings
    cy = bottom + h - 13 - size * .38
    if color == ORANGE:
        p = c.beginPath()
        p.moveTo(x + 20, cy + 6.5)
        p.lineTo(x + 26.5, cy - 5)
        p.lineTo(x + 13.5, cy - 5)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont('SansBold', 8)
        c.drawCentredString(x + 20, cy - 3.6, '!')
    else:
        p = c.beginPath()
        p.moveTo(x + 20, cy + 6)
        p.lineTo(x + 26, cy)
        p.lineTo(x + 20, cy - 6)
        p.lineTo(x + 14, cy)
        p.close()
        c.drawPath(p, fill=1, stroke=0)
    draw_para(c, text, x + 36, bottom + h - 13, width - 44, size)
    return bottom + h


def callouts_for(s):
    out = []
    if s.get('callout'):
        out.append((s['callout'], BLUE))
    if s.get('warning'):
        out.append((s['warning'], ORANGE))
    return out


def callouts_h(s, width, size=16):
    cs = callouts_for(s)
    return sum(callout_h(t, width, size) for t, _ in cs) + 10 * max(len(cs) - 1, 0)


def draw_callouts(c, s, x, width, bottom=BOTTOM, size=16):
    y = bottom
    for text, color in reversed(callouts_for(s)):
        y = draw_callout(c, text, x, y, width, color, size) + 10
    return y - 10 if callouts_for(s) else bottom


# ------------------------------------------------------------ text stack
def stack_fit(s, width, avail, max_size, min_size, mode, eqs=None, items=None):
    items = s['body'] if items is None else items
    eqs = s['equations'] if eqs is None else eqs
    for size in np.arange(max_size, min_size - .01, -.5):
        h = 0
        if eqs:
            h += equation_block_h(eqs, width, min(28, size * 1.35)) + (size * .9 if items else 0)
        if items:
            h += items_height(items, width, size, mode)
        if h <= avail:
            return float(size), h
    raise SlideError(f'text does not fit ({h:.0f} > {avail:.0f} pt) at minimum size {min_size}')


def draw_stack(c, s, x, top, width, size, mode, accent, overlays, page, eqs=None, items=None,
               color=INK):
    items = s['body'] if items is None else items
    eqs = s['equations'] if eqs is None else eqs
    y = top
    if eqs:
        y = draw_equations(c, eqs, x, y, width, min(28, size * 1.35), overlays, page)
        if items:
            y -= size * .9
    if items:
        y = draw_items(c, items, x, y, width, size, mode, accent, color)
    return y


def mode_for(s):
    return {'exercise': 'q', 'recap': 'check', 'objectives': 'badge', 'lab': 'task'}.get(s['type'], 'bullet')


def accent_for(s):
    return TYPE_ACCENT.get(s['type'], BLUE)


# ------------------------------------------------------------ figures
def figure_overlay(overlays, page, name, rect):
    path = ASSETS / f'{name}.pdf'
    if not path.exists():
        raise SlideError(f'missing figure {name}.pdf')
    overlays.append({'page': page, 'kind': 'fig', 'value': name, 'rect': fitz.Rect(*rect)})


def fig_rect(x, top, w, bottom):
    """ReportLab box -> fitz rect (y axis flipped)."""
    return (x, H - top, x + w, H - bottom)


def caption_h(text, width, size=12.5):
    return para_h(text, width, size, color=MUTED) if text else 0


# ------------------------------------------------------------ chrome
def chrome(c, s, i, n, deck):
    c.setFillColor(colors.white)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    accent = accent_for(s)
    if s['type'] in ('exercise', 'debrief', 'lab'):
        c.setFillColor(C(accent))
        c.rect(0, 0, 7, H, fill=1, stroke=0)
    kicker = s.get('section') or TYPE_KICKER.get(s['type']) or deck.get('current_section') or deck['label']
    c.setFillColor(C(accent if s['type'] in TYPE_ACCENT else TEAL))
    c.rect(MX, H - 43.5, 16, 3.2, fill=1, stroke=0)
    spaced(c, plain(kicker).upper(), MX + 24, H - 45.5, 'SansSemi', 10, accent if s['type'] in TYPE_ACCENT else TEAL, 1.1)
    # chips on the kicker line (exercise / lab timing and mode)
    xr = W - MX
    for key, icon in (('mode', 'people'), ('time', 'clock')):
        if s.get(key):
            w = chip(c, s[key], xr, H - 42, accent, icon=icon, align='right')
            xr -= w + 8
    title_w = CW
    p = para(s['title'], 28, font='SansSemi', leading=33)
    h = p.wrap(title_w, 1000)[1]
    if h > 70:
        raise SlideError(f'title wraps to more than two lines: {s["title"]!r}')
    p.drawOn(c, MX, H - 60 - h)
    # footer with progress rule
    c.setStrokeColor(C(RULE))
    c.setLineWidth(1)
    c.line(MX, 33, W - MX, 33)
    c.setStrokeColor(C(BLUE))
    c.setLineWidth(2)
    c.line(MX, 33, MX + CW * (i + 1) / n, 33)
    c.setFillColor(C(MUTED))
    c.setFont('Sans', 9)
    c.drawString(MX, 18, deck['footer_label'])
    c.setFont('SansMedium', 9)
    c.drawRightString(W - MX, 18, f'{i + 1}')
    return H - 60 - h - 20


# ------------------------------------------------------------ decoration
@lru_cache(maxsize=4)
def contour_paths(seed=7, w=620, h=540):
    """Smooth contour lines (a stylised structure map) for dark slides."""
    import contourpy
    rng = np.random.default_rng(seed)
    xs, ys = np.linspace(0, w, 140), np.linspace(0, h, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    for _ in range(9):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx, sy = rng.uniform(80, 220), rng.uniform(70, 200)
        Z += rng.uniform(-1, 1.4) * np.exp(-((X - cx) ** 2 / (2 * sx ** 2) + (Y - cy) ** 2 / (2 * sy ** 2)))
    Z += .0012 * X
    gen = contourpy.contour_generator(xs, ys, Z)
    lines = []
    for level in np.linspace(Z.min(), Z.max(), 30)[1:-1]:
        for seg in gen.lines(level):
            if len(seg) > 8:
                lines.append(seg[::2])
    return lines


def decorate(c, x0, alpha=.5, color='#2F6390'):
    c.saveState()
    p = c.beginPath()
    p.rect(x0, 0, W - x0, H)
    c.clipPath(p, stroke=0, fill=0)
    c.setStrokeColor(C(color))
    c.setStrokeAlpha(alpha)
    c.setLineWidth(.8)
    for seg in contour_paths(w=W - x0 + 40):
        path = c.beginPath()
        path.moveTo(x0 - 20 + seg[0][0], seg[0][1])
        for px, py in seg[1:]:
            path.lineTo(x0 - 20 + px, py)
        c.drawPath(path, stroke=1, fill=0)
    c.restoreState()
    # fade the decoration into the background on its left edge
    c.saveState()
    steps = 24
    for k in range(steps):
        c.setFillColor(C(INK))
        c.setFillAlpha(1 - k / steps)
        c.rect(x0 + k * 6, 0, 6.5, H, fill=1, stroke=0)
    c.restoreState()


# ------------------------------------------------------------ cover / section
def draw_cover(c, s, deck):
    c.setFillColor(C(INK))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    decorate(c, 470, alpha=.55)
    c.setFillColor(C(SKY))
    c.rect(MX, H - 84, 28, 3.5, fill=1, stroke=0)
    spaced(c, plain(s.get('label', deck['label'])).upper(), MX + 38, H - 86.5, 'SansSemi', 12.5, SKY, 1.6)
    y = draw_para(c, s['title'], MX, H - 118, 640, 46, font='SansSemi', color='#FFFFFF', leading=53)
    texts = [it['text'] for it in s['body']]
    if texts:
        y = draw_para(c, texts[0], MX + 2, y - 16, 600, 21, color=MIST, leading=28)
    for t in texts[1:]:
        y = draw_para(c, t, MX + 2, y - 10, 600, 16, color=MIST, leading=22)
    if s['metas']:
        x = MX
        y -= 34
        for m in s['metas']:
            tw = text_w(m, 'SansMedium', 12)
            rrect(c, x, y - 11, tw + 22, 24, r=12, stroke='#4B7BA7', lw=1)
            c.setFillColor(C(MIST))
            c.setFont('SansMedium', 12)
            c.drawString(x + 11, y - 3, m)
            x += tw + 34
    c.setStrokeColor(C(SKY))
    c.setLineWidth(1.5)
    c.line(MX, 70, MX + 36, 70)
    c.setFillColor(C(MIST))
    c.setFont('Sans', 12)
    c.drawString(MX, 50, s.get('footer', deck['course']))


def draw_section(c, s, deck, number, total):
    c.setFillColor(C(INK))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    decorate(c, 560, alpha=.42)
    spaced(c, f'PART {number} OF {total}', MX, H - 86, 'SansSemi', 12, SKY, 1.6)
    texts = [it for it in s['body'] if it['kind'] == 'text']
    bullets = [it for it in s['body'] if it['kind'] != 'text']

    def content_h(t_size, b_size):
        h = para_h(s['title'], 700, t_size, font='SansSemi', leading=t_size * 1.15)
        h += sum(para_h(it['text'], 640, 20, leading=27) + 12 for it in texts)
        if bullets:
            h += 20 + sum(para_h(it['text'], 600, b_size, leading=b_size * 1.33) + 7 for it in bullets)
        return h

    # Step down the part number and type sizes until the content fits.
    for num_size, t_size, b_size in ((104, 40, 16.5), (84, 38, 16.5), (68, 36, 16), (0, 36, 16),
                                     (0, 32, 15), (0, 30, 14)):
        top = H - 110 - (num_size * .83 + 30 if num_size else 0)
        if top - content_h(t_size, b_size) >= 84:
            break
    else:
        raise SlideError(f'section slide content overflows: {s["title"]!r}')
    if num_size:
        c.setFillColor(C(SKY))
        c.setFont('SansLight', num_size)
        c.drawString(MX - 6, H - 110 - num_size * .83, f'{number:02d}')
    y = draw_para(c, s['title'], MX, top, 700, t_size, font='SansSemi', color='#FFFFFF', leading=t_size * 1.15)
    for it in texts:
        y = draw_para(c, it['text'], MX + 2, y - 12, 640, 20, color=MIST, leading=27)
    if bullets:
        y -= 20
        for it in bullets:
            c.setFillColor(C(SKY))
            c.rect(MX + 2, y - b_size * .7, 7, 7, fill=1, stroke=0)
            y = draw_para(c, it['text'], MX + 20, y, 600, b_size, color=MIST, leading=b_size * 1.33) - 7
    # part progress
    seg, gap = 34, 6
    for k in range(total):
        col = SKY if k + 1 == number else ('#4F7FA8' if k + 1 < number else '#244A70')
        c.setFillColor(C(col))
        c.rect(MX + k * (seg + gap), 52, seg, 4, fill=1, stroke=0)
    c.setFillColor(C(MIST))
    c.setFont('Sans', 11)
    c.drawString(MX + total * (seg + gap) + 10, 49.5, deck['label'])


# ------------------------------------------------------------ layouts
def auto_layout(s):
    if s['layout'] != 'auto':
        return s['layout']
    if s['cards']:
        return 'cards'
    if s['stats']:
        return 'stats'
    if len(s['figures']) >= 2:
        return 'pair'
    if s['figures']:
        n_items = len(s['body'])
        chars = sum(len(it['text']) for it in s['body'])
        return 'split' if (n_items >= 3 or chars > 190 or s['equations']) else 'figure'
    if s['codes']:
        return 'code'
    if 'table' in s:
        return 'table'
    return 'bullets'


def layout_bullets(c, s, top, overlays, page, max_size=23, min_size=15):
    if s['type'] == 'reference':
        max_size, min_size = 17, 12
    ch = callouts_h(s, CW)
    avail = top - BOTTOM - (ch + 22 if ch else 0)
    size, _ = stack_fit(s, CW, avail, max_size, min_size, mode_for(s))
    draw_stack(c, s, MX, top, CW, size, mode_for(s), accent_for(s), overlays, page)
    draw_callouts(c, s, MX, CW)


def layout_split(c, s, top, overlays, page, left_figure=False):
    fw = float(s.get('figwidth', .56))
    gap = 30
    fig_w = CW * fw
    txt_w = CW - fig_w - gap
    fx = MX if left_figure else MX + txt_w + gap
    tx = MX + fig_w + gap if left_figure else MX
    ch = callouts_h(s, txt_w, 15)
    avail = top - BOTTOM - (ch + 20 if ch else 0)
    if s['body'] or s['equations']:
        size, _ = stack_fit(s, txt_w, avail, 21, 13.5, mode_for(s))
        draw_stack(c, s, tx, top, txt_w, size, mode_for(s), accent_for(s), overlays, page)
    draw_callouts(c, s, tx, txt_w, size=15)
    cap = s.get('caption', '')
    cap_h = caption_h(cap, fig_w)
    fb = BOTTOM + (cap_h + 10 if cap else 0)
    if top - fb < 200:
        raise SlideError('figure area too small in split layout')
    figure_overlay(overlays, page, s['figures'][0], fig_rect(fx, top + 4, fig_w, fb))
    if cap:
        draw_para(c, cap, fx, BOTTOM + cap_h, fig_w, 12.5, color=MUTED)


def layout_figure(c, s, top, overlays, page):
    y = top
    ch = callouts_h(s, CW, 15)
    if s['body'] or s['equations']:
        size, h = stack_fit(s, CW, (top - BOTTOM) * .38, 20, 14, mode_for(s))
        y = draw_stack(c, s, MX, top, CW, size, mode_for(s), accent_for(s), overlays, page) - 14
    bottom = draw_callouts(c, s, MX, CW, size=15) + (12 if ch else 0)
    cap = s.get('caption', '')
    cap_h = caption_h(cap, CW)
    if cap:
        draw_para(c, cap, MX, bottom + cap_h, CW, 12.5, color=MUTED)
        bottom += cap_h + 10
    if y - bottom < 180:
        raise SlideError(f'figure area too small ({y - bottom:.0f} pt)')
    n = len(s['figures'])
    gap = 24
    w = (CW - gap * (n - 1)) / n
    for k, name in enumerate(s['figures']):
        figure_overlay(overlays, page, name, fig_rect(MX + k * (w + gap), y, w, bottom))


def card_color(card):
    return TAG_COLORS.get(card['tag'], BLUE)


def card_content_h(card, width, size):
    h = para_h(card['title'], width, size * 1.08, font='SansSemi') if card['title'] else 0
    if card['lines']:
        h += (size * .55 if card['title'] else 0) + items_height(card['lines'], width, size, 'bullet', gap=size * .38)
    return h


def layout_cards(c, s, top, overlays, page):
    cards = s['cards']
    n = len(cards)
    cols = int(s.get('cols', 0)) or (n if n <= 4 else (3 if n in (5, 6, 9) else 4))
    rows = math.ceil(n / cols)
    gap = 18 if s['layout'] != 'compare' else 46
    cw = (CW - gap * (cols - 1)) / cols
    y = top
    lead = [it for it in s['body']]
    if lead or s['equations']:
        size, _ = stack_fit(s, CW, (top - BOTTOM) * .34, 20, 14, mode_for(s))
        y = draw_stack(c, s, MX, top, CW, size, mode_for(s), accent_for(s), overlays, page) - 18
    ch = callouts_h(s, CW, 15)
    bottom = BOTTOM + (ch + 18 if ch else 0)
    avail = y - bottom
    padx, padt, padb = 16, 26, 16
    for size in np.arange(17, 11.9, -.5):
        heights = []
        for r in range(rows):
            row = cards[r * cols:(r + 1) * cols]
            heights.append(max(card_content_h(cd, cw - 2 * padx, size) for cd in row) + padt + padb)
        total = sum(heights) + 16 * (rows - 1)
        if total <= avail:
            break
    else:
        raise SlideError(f'cards do not fit ({total:.0f} > {avail:.0f} pt)')
    # let short cards breathe a little, but never stretch them into empty boxes
    extra = min((avail - total) / rows, 40)
    heights = [h + max(extra, 0) for h in heights]
    for k, cd in enumerate(cards):
        r, col = divmod(k, cols)
        x = MX + col * (cw + gap)
        ytop = y - sum(heights[:r]) - 16 * r
        hh = heights[r]
        color = card_color(cd)
        rrect(c, x, ytop - hh, cw, hh, r=9, fill=blend(color, .93) if color != INK else PANEL)
        c.setFillColor(C(color))
        c.rect(x + padx, ytop - 13, 26, 3.5, fill=1, stroke=0)
        ty = ytop - padt
        tx = x + padx
        tw = cw - 2 * padx
        if cd['tag'] in ('good', 'bad'):
            (icon_check if cd['tag'] == 'good' else icon_cross)(c, tx + 7, ty - size * .72, s=6.5, color=color)
            tx += 20
            tw -= 20
        if cd['title']:
            ty = draw_para(c, cd['title'], tx, ty, tw, size * 1.08, font='SansSemi')
            if cd['lines']:
                ty -= size * .55
        if cd['lines']:
            draw_items(c, cd['lines'], x + padx, ty, cw - 2 * padx, size, 'bullet', color, gap=size * .38)
    if s['layout'] == 'compare' and n == 2:
        cx, cy = MX + cw + gap / 2, y - heights[0] / 2
        c.setFillColor(C(INK))
        c.circle(cx, cy, 17, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont('SansSemi', 13)
        c.drawCentredString(cx, cy - 4.5, 'vs')
    draw_callouts(c, s, MX, CW, size=15)


def layout_steps(c, s, top, overlays, page):
    steps = [it for it in s['body'] if it['kind'] in ('num', 'bullet')]
    lead = [it for it in s['body'] if it['kind'] == 'text']
    n = len(steps)
    if not 2 <= n <= 6:
        raise SlideError('steps layout needs 2-6 numbered items')
    y = top
    if lead or s['equations']:
        size, _ = stack_fit(s, CW, (top - BOTTOM) * .3, 20, 14, 'bullet', items=lead)
        y = draw_stack(c, s, MX, top, CW, size, 'bullet', BLUE, overlays, page, items=lead) - 26
    ch = callouts_h(s, CW, 15)
    bottom = BOTTOM + (ch + 18 if ch else 0)
    colw = CW / n
    r = 19
    cy = y - r
    avail = cy - r - 14 - bottom
    for size in np.arange(15.5, 11.4, -.5):
        hs = []
        for st in steps:
            t, d = split_title(st['text'])
            h = para_h(t, colw - 18, size * 1.1, font='SansSemi', align=1)
            if d:
                h += 6 + para_h(d, colw - 18, size, align=1)
            hs.append(h)
        if max(hs) <= avail:
            break
    else:
        raise SlideError('step descriptions do not fit')
    c.setStrokeColor(C(RULE))
    c.setLineWidth(2)
    c.line(MX + colw / 2, cy, MX + CW - colw / 2, cy)
    for k, st in enumerate(steps):
        cx = MX + colw * (k + .5)
        c.setFillColor(C(BLUE))
        c.circle(cx, cy, r, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont('SansSemi', 16)
        c.drawCentredString(cx, cy - 5.6, str(st.get('n', k + 1)))
        t, d = split_title(st['text'])
        ty = draw_para(c, t, cx - colw / 2 + 9, cy - r - 14, colw - 18, size * 1.1, font='SansSemi', align=1)
        if d:
            draw_para(c, d, cx - colw / 2 + 9, ty - 6, colw - 18, size, align=1, color='#243B53')
    draw_callouts(c, s, MX, CW, size=15)


def layout_steps_vertical(c, s, top, overlays, page):
    items = s['body']
    ch = callouts_h(s, CW)
    avail = top - BOTTOM - (ch + 22 if ch else 0)
    size, _ = stack_fit(s, CW, avail, 21, 14, 'badge')
    draw_stack(c, s, MX, top, CW, size, 'badge', BLUE, overlays, page)
    draw_callouts(c, s, MX, CW)


def layout_stats(c, s, top, overlays, page):
    stats = []
    for raw in s['stats']:
        parts = [p.strip() for p in raw.split('|')]
        stats.append((parts[0], parts[1] if len(parts) > 1 else '', TAG_COLORS.get(parts[2], BLUE) if len(parts) > 2 else BLUE))
    n = len(stats)
    if not 1 <= n <= 4:
        raise SlideError('stats layout needs 1-4 @stat lines')
    y = top
    if s['body'] or s['equations']:
        size, _ = stack_fit(s, CW, (top - BOTTOM) * .36, 20, 14, mode_for(s))
        y = draw_stack(c, s, MX, top, CW, size, mode_for(s), accent_for(s), overlays, page) - 22
    gap = 20
    tw = (CW - gap * (n - 1)) / n
    lab_h = max(para_h(l, tw - 36, 15, align=1) for _, l, _ in stats)
    th = 50 + 16 + lab_h + 36
    ch = callouts_h(s, CW, 15)
    if y - th < BOTTOM + (ch + 18 if ch else 0):
        raise SlideError('stat tiles do not fit')
    for k, (value, lab, color) in enumerate(stats):
        x = MX + k * (tw + gap)
        rrect(c, x, y - th, tw, th, r=10, fill=blend(color, .92))
        size = 46
        while text_w(value, 'SansSemi', size) > tw - 30 and size > 24:
            size -= 2
        c.setFillColor(C(color))
        c.setFont('SansSemi', size)
        c.drawCentredString(x + tw / 2, y - 22 - size * .8, value)
        draw_para(c, lab, x + 18, y - 22 - 50 - 12, tw - 36, 15, align=1)
    draw_callouts(c, s, MX, CW, size=15)


def table_widths(rows, total, size):
    ncols = len(rows[0])
    nat = []
    for j in range(ncols):
        wmax = max(text_w(plain(r[j]), 'SansSemi' if i == 0 else 'Sans', size) for i, r in enumerate(rows))
        nat.append(wmax + 26)
    if sum(nat) <= total:
        return [n + (total - sum(nat)) * n / sum(nat) for n in nat]
    widths = [0.0] * ncols
    remaining, open_cols = total, list(range(ncols))
    while open_cols:
        share = remaining / len(open_cols)
        small = [j for j in open_cols if nat[j] <= share]
        if not small:
            for j in open_cols:
                widths[j] = share
            break
        for j in small:
            widths[j] = nat[j]
            remaining -= nat[j]
            open_cols.remove(j)
    return widths


def build_table(rows, width, size, first_col_bold=True):
    ncols = len(rows[0])
    if any(len(r) != ncols for r in rows):
        raise SlideError('table rows have unequal column counts')
    widths = table_widths(rows, width, size)
    pad = max(6, size * .55)
    cells = []
    for ri, row in enumerate(rows):
        out = []
        for ci, v in enumerate(row):
            font = 'SansSemi' if ri == 0 else ('SansMedium' if ci == 0 and first_col_bold else 'Sans')
            out.append(Paragraph(rich(v), _style(size, font, '#FFFFFF' if ri == 0 else INK, round(size * 1.25, 2), 0)))
        cells.append(out)
    t = Table(cells, colWidths=widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), C(INK)), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), pad + 3), ('RIGHTPADDING', (0, 0), (-1, -1), pad),
        ('TOPPADDING', (0, 0), (-1, -1), pad), ('BOTTOMPADDING', (0, 0), (-1, -1), pad + 1),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C(PANEL), colors.white]),
        ('LINEBELOW', (0, 1), (-1, -1), .6, C(RULE))]))
    h = t.wrap(width, 10000)[1]
    return t, h


def layout_table(c, s, top, overlays, page):
    rows = s['table']
    ch = callouts_h(s, CW, 15)
    bottom = BOTTOM + (ch + 18 if ch else 0)
    body = s['body']
    for size in np.arange(17, 10.9, -.5):
        t, th = build_table(rows, CW, size)
        bh = items_height(body, CW, min(18, size + 1), mode_for(s)) + 18 if body else 0
        if th + bh <= top - bottom:
            break
    else:
        raise SlideError(f'table does not fit ({th + bh:.0f} > {top - bottom:.0f} pt)')
    t.drawOn(c, MX, top - th)
    if body:
        draw_items(c, body, MX, top - th - 18, CW, min(18, size + 1), mode_for(s), accent_for(s))
    draw_callouts(c, s, MX, CW, size=15)


# ------------------------------------------------------------ code
CODE_STYLE = None


def code_style(ttype):
    from pygments.token import Token
    global CODE_STYLE
    if CODE_STYLE is None:
        CODE_STYLE = [
            (Token.Comment, ('#6B7F93', 'MonoItalic')), (Token.Keyword, (BLUE, 'MonoSemi')),
            (Token.Operator.Word, (BLUE, 'MonoSemi')), (Token.Name.Builtin, (TEAL, 'Mono')),
            (Token.Name.Function, ('#0F2A44', 'MonoMedium')), (Token.Name.Class, ('#0F2A44', 'MonoMedium')),
            (Token.Name.Decorator, (PURPLE, 'Mono')), (Token.Literal.String, ('#1D7A4E', 'Mono')),
            (Token.Literal.Number, (ORANGE, 'Mono')), (Token.Operator, ('#4A6580', 'Mono')),
            (Token.Punctuation, ('#4A6580', 'Mono'))]
    t = ttype
    while t is not None:
        for key, st in CODE_STYLE:
            if t is key:
                return st
        t = t.parent
    return (INK, 'Mono')


def code_lines(code, lang):
    """Return a list of lines, each a list of (text, color, font) runs."""
    if lang in ('output', 'console', 'text', 'plain'):
        return [[(line, MIST if lang in ('output', 'console') else INK, 'Mono')] for line in code.split('\n')]
    from pygments import lex
    from pygments.lexers import get_lexer_by_name
    try:
        lexer = get_lexer_by_name(lang)
    except Exception:
        lexer = get_lexer_by_name('text')
    lines, cur = [], []
    for ttype, value in lex(code, lexer):
        parts = value.split('\n')
        for k, part in enumerate(parts):
            if k:
                lines.append(cur)
                cur = []
            if part:
                color, font = code_style(ttype)
                cur.append((part, color, font))
    lines.append(cur)
    while lines and not lines[-1]:
        lines.pop()
    return lines[:len(code.rstrip('\n').split('\n'))]


def parse_highlight(spec):
    out = set()
    for part in (spec or '').replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def code_block_h(code, size):
    n = len(code['text'].rstrip('\n').split('\n'))
    return n * size * 1.42 + 24 + (14 if code['lang'] in ('output', 'console') else 0)


def code_fit(codes, width, avail, max_size=16, min_size=10.5):
    for size in np.arange(max_size, min_size - .01, -.5):
        wide = max(text_w(line.expandtabs(4), 'Mono', size) for cd in codes
                   for line in cd['text'].split('\n'))
        h = sum(code_block_h(cd, size) for cd in codes) + 12 * (len(codes) - 1)
        if wide <= width - 34 and h <= avail:
            return float(size), h
    raise SlideError(f'code does not fit (width {wide:.0f}/{width - 34:.0f}, height {h:.0f}/{avail:.0f})')


def draw_code(c, cd, x, top, width, size, highlight=frozenset()):
    h = code_block_h(cd, size)
    dark = cd['lang'] in ('output', 'console')
    rrect(c, x, top - h, width, h, r=8, fill=INK_2 if dark else '#F3F6FA')
    if not dark:
        c.setFillColor(C(BLUE))
        c.roundRect(x, top - h, 4, h, 2, fill=1, stroke=0)
    lh = size * 1.42
    y0 = top - 12 - (14 if dark else 0)
    if dark:
        spaced(c, 'OUTPUT', x + 16, top - 16, 'SansSemi', 8.5, SKY, 1.2)
    for k, runs in enumerate(code_lines(cd['text'], cd['lang']), start=1):
        base = y0 - k * lh
        if k in highlight:
            c.setFillColor(C('#FFF0BF'))
            c.rect(x + 5, base - size * .38, width - 5, lh, fill=1, stroke=0)
        cx = x + 18
        for text, color, font in runs:
            text = text.expandtabs(4)
            c.setFillColor(C(color))
            c.setFont(font, size)
            c.drawString(cx, base + size * .12, text)
            cx += text_w(text, font, size)
    return top - h


def layout_code(c, s, top, overlays, page, split=False):
    codes = s['codes']
    ch = callouts_h(s, CW, 15)
    bottom = BOTTOM + (ch + 18 if ch else 0)
    hl = parse_highlight(s.get('highlight'))
    if split:
        cw = CW * float(s.get('codewidth', .6))
        size, h = code_fit(codes, cw, top - bottom)
        y = top
        for cd in codes:
            y = draw_code(c, cd, MX, y, cw, size, hl if cd is codes[0] else frozenset()) - 12
        tx, tw = MX + cw + 28, CW - cw - 28
        if s['body']:
            bs, _ = stack_fit(s, tw, top - bottom, 19, 13, mode_for(s), eqs=[])
            draw_stack(c, s, tx, top, tw, bs, mode_for(s), accent_for(s), overlays, page, eqs=[])
    else:
        body_h = lambda sz: items_height(s['body'], CW, sz, mode_for(s)) + 18 if s['body'] else 0  # noqa: E731
        for bsize in (19, 18, 17, 16, 15, 14):
            try:
                size, h = code_fit(codes, CW, top - bottom - body_h(bsize))
                break
            except SlideError:
                continue
        else:
            raise SlideError('code and text do not fit together')
        y = top
        for cd in codes:
            y = draw_code(c, cd, MX, y, CW, size, hl if cd is codes[0] else frozenset()) - 12
        if s['body']:
            draw_items(c, s['body'], MX, y - 6, CW, bsize, mode_for(s), accent_for(s))
    draw_callouts(c, s, MX, CW, size=15)


# ------------------------------------------------------------ special types
def kind_color(kind: str):
    k = kind.lower()
    for keys, color in KIND_COLORS:
        if any(key in k for key in keys):
            return color
    return BLUE


def layout_agenda(c, s, top, overlays, page):
    rows = [r for r in s.get('table', [])]
    if rows and rows[0][0].strip().lower() in ('time', 'start', 'when'):
        rows = rows[1:]
    if not rows:
        raise SlideError('agenda slide needs a table: | time | activity | kind |')
    y = top
    if s['body']:
        y = draw_items(c, s['body'], MX, top, CW, 17, 'bullet', BLUE) - 14
    ncol = 1 if len(rows) <= 16 else 2
    per = math.ceil(len(rows) / ncol)
    gap_c = 36
    colw = (CW - gap_c * (ncol - 1)) / ncol
    avail = y - BOTTOM
    chip_size = 10.5
    kinds = [r[2] if len(r) > 2 else '' for r in rows]
    chip_w = max([text_w(k, 'SansSemi', chip_size) + 18 for k in kinds if k] or [0])

    def geometry(size):
        time_w = max(text_w(r[0], 'MonoMedium', size * .92) for r in rows)
        line_dx = time_w + 16
        title_dx = line_dx + 18
        title_w = colw - title_dx - (chip_w + 14 if chip_w else 0)
        return line_dx, title_dx, title_w

    for size in np.arange(16, 10.9, -.5):
        line_dx, title_dx, title_w = geometry(size)
        rh = [max(para_h(r[1], title_w, size,
                         font='Sans' if kind_color((r[2] if len(r) > 2 else '') or r[1]) == FAINT else 'SansMedium'),
                  size * 1.3) + size * .62 for r in rows]
        colh = [sum(rh[k * per:(k + 1) * per]) for k in range(ncol)]
        if max(colh) <= avail and title_w > 120:
            break
    else:
        raise SlideError('agenda does not fit; shorten session names or merge rows')
    for col in range(ncol):
        x = MX + col * (colw + gap_c)
        yy = y
        chunk = rows[col * per:(col + 1) * per]
        hs = rh[col * per:(col + 1) * per]
        line_x = x + line_dx
        c.setStrokeColor(C(RULE))
        c.setLineWidth(2)
        c.line(line_x, yy - size * .72, line_x, yy - sum(hs[:-1]) - size * .72)
        for r, h in zip(chunk, hs):
            kind = r[2] if len(r) > 2 else ''
            color = kind_color(kind or r[1])
            cy = yy - size * .72
            c.setFillColor(C(MUTED))
            c.setFont('MonoMedium', size * .92)
            c.drawString(x, cy - size * .32, r[0])
            c.setFillColor(C(color))
            c.circle(line_x, cy, 5.5 if color != FAINT else 4, stroke=0, fill=1)
            muted = color == FAINT
            draw_para(c, r[1], x + title_dx, yy, title_w, size,
                      font='Sans' if muted else 'SansMedium', color=MUTED if muted else INK)
            if kind:
                chip(c, kind, x + colw, cy, color, size=chip_size, align='right')
            yy -= h


def layout_statement(c, s, top, overlays, page):
    texts = [it['text'] for it in s['body']]
    main = texts[0] if texts else s['title']
    rest = texts[1:]
    for size in (34, 32, 30, 28, 26, 24):
        h = para_h(main, 760, size, font='SansSemi', leading=size * 1.22, align=1)
        rh = sum(para_h(t, 700, 18, color=MUTED, align=1) + 12 for t in rest)
        if h + rh + 40 <= top - BOTTOM:
            break
    y = BOTTOM + (top - BOTTOM + h + rh + 30) / 2
    c.setFillColor(C(BLUE))
    c.rect(W / 2 - 22, y + 16, 44, 4, fill=1, stroke=0)
    y = draw_para(c, main, (W - 760) / 2, y, 760, size, font='SansSemi', leading=size * 1.22, align=1)
    for t in rest:
        y = draw_para(c, t, (W - 700) / 2, y - 16, 700, 18, color=MUTED, align=1)
    draw_callouts(c, s, MX, CW, size=15)


def path_lines(path, width, max_size=12.5, min_size=9):
    """Split a file path into directory and file-name lines that fit width."""
    parts = path.split('/')
    lines = ['/'.join(parts[:-1]) + '/', parts[-1]] if len(parts) > 1 else [path]
    size = max_size
    while size > min_size and max(text_w(l, 'Mono', size) for l in lines) > width:
        size -= .5
    if max(text_w(l, 'Mono', size) for l in lines) > width:
        # break further at underscores
        out = []
        for l in lines:
            cur = ''
            for chunk in re.split(r'(?<=[_/])', l):
                if cur and text_w(cur + chunk, 'Mono', size) > width:
                    out.append(cur)
                    cur = chunk
                else:
                    cur += chunk
            out.append(cur)
        lines = out
    return lines, size


def layout_lab(c, s, top, overlays, page):
    delivs = s['deliverables']
    nbs = s.get('notebooks') or ([s['notebook']] if s.get('notebook') else [])
    nb = bool(nbs)
    right = bool(delivs or nb)
    rw = 310 if right else 0
    lw = CW - (rw + 28 if right else 0)
    ch = callouts_h(s, lw, 15)
    avail = top - BOTTOM - (ch + 18 if ch else 0)
    if s['body']:
        size, _ = stack_fit(s, lw, avail, 20, 13.5, 'task')
        draw_stack(c, s, MX, top, lw, size, 'task', ORANGE, overlays, page)
    draw_callouts(c, s, MX, lw, size=15)
    if right:
        x = MX + lw + 28
        inner = rw - 36
        parts_h = 0
        nb_size = min((path_lines(n, inner)[1] for n in nbs), default=0)
        nb_lines = []
        for k, n in enumerate(nbs):
            lines, _ = path_lines(n, inner, max_size=nb_size, min_size=min(nb_size, 9))
            nb_lines.extend(lines if k == 0 else [''] + lines)
        nb_lines = [l for l in nb_lines]
        nb_h = sum(nb_size * (1.35 if l else .5) for l in nb_lines)
        for size in np.arange(16, 11.4, -.5):
            dh = items_height([{'kind': 'bullet', 'text': d} for d in delivs], inner, size, 'bullet') if delivs else 0
            parts_h = 22 + (nb_h + 38 if nb else 0) + (dh + 34 if delivs else 0) + 6
            if parts_h <= top - BOTTOM:
                break
        rrect(c, x, top - parts_h, rw, parts_h, r=10, fill=ORANGE_T)
        y = top - 22
        if nb:
            icon_book(c, x + 24, y - 5, s=6, color=ORANGE)
            spaced(c, 'NOTEBOOKS' if len(nbs) > 1 else 'NOTEBOOK', x + 38, y - 9, 'SansSemi', 10, ORANGE, 1.1)
            y -= 18
            c.setFillColor(C('#1C3F63'))
            for line in nb_lines:
                if not line:
                    y -= nb_size * .5
                    continue
                c.setFont('Mono', nb_size)
                c.drawString(x + 18, y - nb_size, line)
                y -= nb_size * 1.35
            y -= 20
        if delivs:
            icon_check(c, x + 24, y - 5, s=5.5, color=ORANGE, lw=2)
            spaced(c, 'EVIDENCE TO SUBMIT', x + 38, y - 9, 'SansSemi', 10, ORANGE, 1.1)
            draw_items(c, [{'kind': 'bullet', 'text': d} for d in delivs], x + 18, y - 22, inner, size,
                       'bullet', ORANGE)


# ------------------------------------------------------------ page driver
SUPPORTS = {
    'bullets': {'body', 'equations', 'callout'},
    'split': {'body', 'equations', 'figures', 'caption', 'callout'},
    'figure': {'body', 'equations', 'figures', 'caption', 'callout'},
    'cards': {'cards', 'body', 'equations', 'callout'},
    'steps': {'body', 'equations', 'callout'},
    'steps-vertical': {'body', 'equations', 'callout'},
    'stats': {'stats', 'body', 'equations', 'callout'},
    'code': {'codes', 'body', 'callout'},
    'code-split': {'codes', 'body', 'callout'},
    'table': {'table', 'body', 'callout'},
    'lab': {'body', 'deliverables', 'notebook', 'callout'},
    'agenda': {'table', 'body'},
    'statement': {'body', 'callout'},
}
SUPPORTS['split-left'] = SUPPORTS['pair'] = SUPPORTS['split']
SUPPORTS['compare'] = SUPPORTS['cards']


def check_content(s, layout):
    present = set()
    for key in ('body', 'equations', 'figures', 'cards', 'stats', 'codes', 'deliverables'):
        if s.get(key):
            present.add(key)
    for key in ('table', 'caption', 'notebook'):
        if s.get(key):
            present.add(key)
    if s.get('callout') or s.get('warning'):
        present.add('callout')
    dropped = present - SUPPORTS[layout]
    if dropped:
        raise SlideError(f'layout {layout!r} cannot show {sorted(dropped)}; split the slide or change @layout')


def render_slide(c, s, i, n, deck, overlays):
    kind = s['type']
    if kind == 'cover':
        draw_cover(c, s, deck)
        return 'cover'
    if kind == 'section':
        deck['current_section'] = s['title']
        draw_section(c, s, deck, s['_section_no'], deck['sections'])
        return 'section'
    top = chrome(c, s, i, n, deck)
    if kind in ('agenda', 'statement'):
        check_content(s, kind)
    elif kind == 'lab' and not s['figures'] and not s['codes'] and 'table' not in s:
        check_content(s, 'lab')
    else:
        check_content(s, auto_layout(s))
    if kind == 'agenda':
        layout_agenda(c, s, top, overlays, i)
        return 'agenda'
    if kind == 'statement':
        layout_statement(c, s, top, overlays, i)
        return 'statement'
    if kind == 'lab' and not s['figures'] and not s['codes'] and 'table' not in s:
        layout_lab(c, s, top, overlays, i)
        return 'lab'
    lay = auto_layout(s)
    if lay in ('split', 'split-left'):
        layout_split(c, s, top, overlays, i, left_figure=(lay == 'split-left'))
    elif lay in ('figure', 'pair'):
        layout_figure(c, s, top, overlays, i)
    elif lay in ('cards', 'compare'):
        layout_cards(c, s, top, overlays, i)
    elif lay == 'steps':
        layout_steps(c, s, top, overlays, i)
    elif lay == 'steps-vertical':
        layout_steps_vertical(c, s, top, overlays, i)
    elif lay == 'stats':
        layout_stats(c, s, top, overlays, i)
    elif lay == 'code':
        layout_code(c, s, top, overlays, i)
    elif lay == 'code-split':
        layout_code(c, s, top, overlays, i, split=True)
    elif lay == 'table':
        layout_table(c, s, top, overlays, i)
    else:
        layout_bullets(c, s, top, overlays, i)
    return lay


def make_pdf(path, slides, deck):
    """Render slides to PATH. Returns per-slide report entries."""
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(W, H), pageCompression=1)
    c._accent = draw_accent  # onDraw callback used by accented inline math
    c.setTitle(deck['label'])
    c.setAuthor('Course teaching materials')
    c.setSubject(deck['course'])
    overlays, report = [], []
    sec = 0
    deck['sections'] = sum(s['type'] == 'section' for s in slides)
    deck['current_section'] = None
    for s in slides:
        if s['type'] == 'section':
            sec += 1
            s['_section_no'] = sec
    for i, s in enumerate(slides):
        try:
            layout = render_slide(c, s, i, len(slides), deck, overlays)
        except SlideError as e:
            raise SlideError(f'{deck["label"]} / slide {i + 1} "{s["title"]}": {e}') from None
        report.append({'slide': i + 1, 'title': plain(s['title']), 'type': s['type'], 'layout': layout})
        c.showPage()
    c.save()
    doc = fitz.open(stream=buf.getvalue(), filetype='pdf')
    for ov in overlays:
        page = doc[ov['page']]
        if ov['kind'] == 'fig':
            src = fitz.open(ASSETS / f"{ov['value']}.pdf")
            sr = src[0].rect
            r = ov['rect']
            scale = min(r.width / sr.width, r.height / sr.height)
            fill = (sr.width * scale * sr.height * scale) / (r.width * r.height)
            spans = [(span['size'], span['text'].strip()) for block in src[0].get_text('dict')['blocks']
                     for line in block.get('lines', []) for span in line['spans'] if span['text'].strip()]
            typical = max(set(round(z) for z, _ in spans), key=[round(z) for z, _ in spans].count) if spans else 0
            # ignore exponents and subscripts (short spans set well below the figure's typical size)
            sizes = [z for z, t in spans if not (len(t) <= 3 and z < .8 * typical)]
            entry = report[ov['page']]
            entry.setdefault('figures', []).append({
                'name': ov['value'], 'scale': round(scale, 3), 'box_fill': round(fill, 3),
                'min_text_pt': round(min(sizes) * scale, 1) if sizes else None})
        else:
            src = fitz.open(stream=equation_pdf(ov['value']), filetype='pdf')
        page.show_pdf_page(ov['rect'], src, 0, keep_proportion=True)
        src.close()
    toc = []
    for i, s in enumerate(slides):
        if s['type'] in ('cover', 'section', 'lab', 'agenda'):
            toc.append([1 if s['type'] != 'lab' else 2, plain(s['title']), i + 1])
    if toc and toc[0][0] != 1:
        toc[0][0] = 1
    doc.set_toc(toc)
    doc.set_metadata({'title': deck['label'] + ': ' + plain(slides[0]['title']),
                      'author': 'Course teaching materials', 'subject': deck['course']})
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, deflate=True, garbage=4, no_new_id=True)
    doc.close()
    return report
