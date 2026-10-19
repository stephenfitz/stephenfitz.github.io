"""Shared visual identity for slides and figures.

One palette and one type family (IBM Plex, SIL Open Font License, vendored in
curriculum/fonts) are used by the slide renderer and by every figure, so text in
charts matches text on slides. Import this module before drawing anything.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / 'curriculum' / 'fonts'

# ---------------------------------------------------------------- palette
INK = '#0F2A44'        # navy: titles, body text, dark backgrounds
INK_2 = '#163A5C'      # slightly lighter navy for panels on dark slides
BLUE = '#005EB8'       # primary accent
TEAL = '#087F8C'       # secondary accent: kickers, positive states
ORANGE = '#B34D17'     # emphasis, warnings, laboratories
RED = '#A33B5C'
PURPLE = '#6B4E9B'
GOLD = '#B07D12'
GREEN = '#2E7D4F'
MUTED = '#526579'      # captions, footers, secondary text
FAINT = '#8A9AAB'
RULE = '#D5DEE7'       # hairlines
PANEL = '#F2F5F9'      # neutral panel background
BLUE_T = '#E7F0FA'     # blue tint
TEAL_T = '#E3F2F2'     # teal tint
ORANGE_T = '#FBEEE6'   # orange tint
RED_T = '#F8E9EE'
GOLD_T = '#FBF3DF'
PURPLE_T = '#EFEAF6'
SKY = '#8EC5EC'        # light accent on dark backgrounds
MIST = '#C4D8E7'       # body text on dark backgrounds

# Ordered categorical palette for data series. The first three are the
# workhorses; keep series count small and label lines directly where possible.
SERIES = [BLUE, ORANGE, TEAL, PURPLE, RED, GOLD, '#56687A']

TINTS = {BLUE: BLUE_T, TEAL: TEAL_T, ORANGE: ORANGE_T, RED: RED_T,
         GOLD: GOLD_T, PURPLE: PURPLE_T, INK: PANEL, GREEN: '#E5F2EA'}

# ------------------------------------------------------------------- type
FONTS = {
    'Sans': 'IBMPlexSans-Regular.ttf',
    'SansItalic': 'IBMPlexSans-Italic.ttf',
    'SansLight': 'IBMPlexSans-Light.ttf',
    'SansMedium': 'IBMPlexSans-Medium.ttf',
    'SansMediumItalic': 'IBMPlexSans-MediumItalic.ttf',
    'SansSemi': 'IBMPlexSans-SemiBold.ttf',
    'SansSemiItalic': 'IBMPlexSans-SemiBoldItalic.ttf',
    'SansBold': 'IBMPlexSans-Bold.ttf',
    'SansBoldItalic': 'IBMPlexSans-BoldItalic.ttf',
    'Cond': 'IBMPlexSansCondensed-Regular.ttf',
    'CondMedium': 'IBMPlexSansCondensed-Medium.ttf',
    'CondSemi': 'IBMPlexSansCondensed-SemiBold.ttf',
    'Mono': 'IBMPlexMono-Regular.ttf',
    'MonoItalic': 'IBMPlexMono-Italic.ttf',
    'MonoMedium': 'IBMPlexMono-Medium.ttf',
    'MonoSemi': 'IBMPlexMono-SemiBold.ttf',
}
FAMILY = 'IBM Plex Sans'
MONO_FAMILY = 'IBM Plex Mono'


def font_path(key: str) -> Path:
    return FONT_DIR / FONTS[key]


_mpl_ready = False


def register_matplotlib() -> None:
    """Make IBM Plex available to matplotlib (idempotent)."""
    global _mpl_ready
    if _mpl_ready:
        return
    from matplotlib import font_manager
    for name in FONTS.values():
        font_manager.fontManager.addfont(str(FONT_DIR / name))
    _mpl_ready = True


_rl_ready = False


def register_reportlab() -> None:
    """Register IBM Plex (plus DejaVu fallback) with ReportLab (idempotent)."""
    global _rl_ready
    if _rl_ready:
        return
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from matplotlib import font_manager
    for key, name in FONTS.items():
        pdfmetrics.registerFont(TTFont(key, str(FONT_DIR / name)))
    pdfmetrics.registerFont(TTFont('Fallback', font_manager.findfont(
        font_manager.FontProperties(family='DejaVu Sans'))))
    pdfmetrics.registerFont(TTFont('FallbackBold', font_manager.findfont(
        font_manager.FontProperties(family='DejaVu Sans', weight='bold'))))
    pdfmetrics.registerFontFamily('Sans', normal='Sans', bold='SansSemi',
                                  italic='SansItalic', boldItalic='SansSemiItalic')
    pdfmetrics.registerFontFamily('SansMedium', normal='SansMedium', bold='SansBold',
                                  italic='SansMediumItalic', boldItalic='SansBoldItalic')
    pdfmetrics.registerFontFamily('SansSemi', normal='SansSemi', bold='SansBold',
                                  italic='SansSemiItalic', boldItalic='SansBoldItalic')
    pdfmetrics.registerFontFamily('Mono', normal='Mono', bold='MonoSemi',
                                  italic='MonoItalic', boldItalic='MonoSemi')
    pdfmetrics.registerFontFamily('Cond', normal='Cond', bold='CondSemi',
                                  italic='Cond', boldItalic='CondSemi')
    _rl_ready = True


_cmap_cache: dict[str, set[int]] = {}


def glyphs(key: str = 'Sans') -> set[int]:
    """Code points available in a vendored font (used for glyph fallback)."""
    if key not in _cmap_cache:
        from fontTools.ttLib import TTFont as FTFont
        _cmap_cache[key] = set(FTFont(str(font_path(key))).getBestCmap())
    return _cmap_cache[key]
