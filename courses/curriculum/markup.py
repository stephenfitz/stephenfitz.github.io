"""Slide-source parser and inline markup for the course decks.

Authoring syntax (see curriculum/AUTHORING.md for the full guide)::

    # Cover or section title          (first slide = cover; @type section = divider)
    ## Content slide title
    @type concept|section|exercise|debrief|lab|reference|recap|objectives|agenda|statement
    @layout auto|split|split-left|figure|pair|cards|steps|steps-vertical|stats|code-split|compare
    @figure name   (repeatable; two figures + @layout pair = side by side)
    @caption text      @callout text      @warning text      @equation latex (repeatable)
    @time 4 min        @mode Pairs        @notebook path (repeatable)   @deliverable text (repeatable)
    @stat value | label [| color]  (repeatable)           @section kicker override
    @source url        @figwidth 0.55     @highlight 2-4,7    @meta chip text (cover)
    ### Card heading {tag}              (cards; tags: good bad warn blue teal orange purple red gold ink)
    - bullet / two-space indented '- ' sub-bullet / '1. ' numbered item / plain paragraph
    | table | rows |       fenced ``` code (language after the fence; 'output' = console panel)
    :::notes ... :::   instructor notes (required on every slide)

Inline: **bold**, *italic*, `code`, ==accent==, $inline math$ (write \\$ for a dollar sign).
"""
from __future__ import annotations

import html
import re
from pathlib import Path

from curriculum import theme

LIST_KEYS = {'figure', 'equation', 'stat', 'deliverable', 'meta', 'notebook'}
TYPES = {'concept', 'cover', 'section', 'exercise', 'debrief', 'lab', 'reference', 'recap',
         'objectives', 'agenda', 'statement'}
LAYOUTS = {'auto', 'split', 'split-left', 'figure', 'pair', 'cards', 'steps', 'steps-vertical',
           'stats', 'code', 'code-split', 'compare', 'table', 'bullets'}


class SlideError(ValueError):
    pass


def parse(path: Path) -> list[dict]:
    """Parse a slides.md file into a list of slide dictionaries."""
    text = Path(path).read_text()
    slides = []
    for chunk_no, raw in enumerate(re.split(r'^---\s*$', text, flags=re.M)):
        if not raw.strip():
            continue
        slide = {'title': '', 'type': 'concept', 'layout': 'auto', 'body': [], 'cards': [],
                 'notes': '', 'source': '', 'codes': []}
        for k in LIST_KEYS:
            slide[k + 's'] = []
        table: list[list[str]] = []
        code: list[str] | None = None
        code_lang = ''
        in_notes = False
        card = None
        for line in raw.strip('\n').splitlines():
            stripped = line.strip()
            if code is not None:
                if stripped.startswith('```'):
                    slide['codes'].append({'lang': code_lang, 'text': '\n'.join(code)})
                    code = None
                else:
                    code.append(line.rstrip())
                continue
            if stripped.startswith('```'):
                code, code_lang = [], stripped[3:].strip().lower() or 'python'
                continue
            if stripped == ':::notes':
                in_notes = True
                continue
            if stripped == ':::' and in_notes:
                in_notes = False
                continue
            if in_notes:
                slide['notes'] += line + '\n'
                continue
            if not stripped:
                continue
            if stripped.startswith('@'):
                key, _, value = stripped[1:].partition(' ')
                value = value.strip()
                if key in LIST_KEYS:
                    slide[key + 's'].append(value)
                elif key == 'source':
                    slide['source'] = (slide['source'] + ' ; ' + value).strip(' ;')
                else:
                    slide[key] = value
                continue
            if stripped.startswith('### '):
                heading = stripped[4:].strip()
                tag = ''
                m = re.search(r'\{(\w+)\}\s*$', heading)
                if m:
                    tag, heading = m.group(1), heading[:m.start()].strip()
                card = {'title': heading, 'tag': tag, 'lines': []}
                slide['cards'].append(card)
                continue
            if stripped.startswith('#'):
                slide['title'] = stripped.lstrip('#').strip()
                if stripped.startswith('# '):
                    slide['type'] = 'cover'
                continue
            if stripped.startswith('|'):
                if not re.fullmatch(r'[|\s:\-]+', stripped):
                    cells = re.split(r'(?<!\\)\|', stripped.strip().removeprefix('|').removesuffix('|'))
                    table.append([c.strip().replace('\\|', '|') for c in cells])
                continue
            item = body_item(line)
            (card['lines'] if card is not None else slide['body']).append(item)
        if code is not None:
            raise SlideError(f'{path}: unterminated code fence in slide {chunk_no + 1}')
        if table:
            slide['table'] = table
        if not slide['title']:
            raise SlideError(f'{path}: missing title in slide chunk {chunk_no + 1}')
        # Backwards-compatible single-value views.
        if slide['figures']:
            slide['figure'] = slide['figures'][0]
        if slide['equations']:
            slide['equation'] = slide['equations'][0]
        if slide['codes']:
            slide['code'] = slide['codes'][0]['text']
        if slide['notebooks']:
            slide['notebook'] = slide['notebooks'][0]
        if slide['type'] not in TYPES:
            raise SlideError(f'{path}: unknown @type {slide["type"]!r} in "{slide["title"]}"')
        if slide['layout'] not in LAYOUTS:
            raise SlideError(f'{path}: unknown @layout {slide["layout"]!r} in "{slide["title"]}"')
        slides.append(slide)
    if slides and slides[0]['type'] == 'concept':
        slides[0]['type'] = 'cover'
    return slides


def body_item(line: str) -> dict:
    indent = len(line) - len(line.lstrip(' '))
    s = line.strip()
    if s.startswith(('- ', '* ')):
        return {'kind': 'sub' if indent >= 2 else 'bullet', 'text': s[2:].strip()}
    m = re.match(r'^(\d+)[.)]\s+(.*)$', s)
    if m:
        return {'kind': 'num', 'n': int(m.group(1)), 'text': m.group(2).strip()}
    return {'kind': 'text', 'text': s}


# ------------------------------------------------------------ inline markup
GREEK = {'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε', 'varepsilon': 'ε',
         'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'vartheta': 'ϑ', 'iota': 'ι', 'kappa': 'κ',
         'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'pi': 'π', 'rho': 'ρ', 'sigma': 'σ',
         'tau': 'τ', 'upsilon': 'υ', 'phi': 'φ', 'varphi': 'φ', 'chi': 'χ', 'psi': 'ψ',
         'omega': 'ω', 'Gamma': 'Γ', 'Delta': 'Δ', 'Theta': 'Θ', 'Lambda': 'Λ', 'Xi': 'Ξ',
         'Pi': 'Π', 'Sigma': 'Σ', 'Phi': 'Φ', 'Psi': 'Ψ', 'Omega': 'Ω'}
SYMBOLS = {'sum': '∑', 'prod': '∏', 'partial': '∂', 'nabla': '∇', 'infty': '∞', 'approx': '≈',
           'neq': '≠', 'ne': '≠', 'leq': '≤', 'le': '≤', 'geq': '≥', 'ge': '≥', 'pm': '±',
           'times': '×', 'cdot': '·', 'to': '→', 'rightarrow': '→', 'leftarrow': '←',
           'Rightarrow': '⇒', 'in': '∈', 'notin': '∉', 'odot': '⊙', 'circ': '∘', 'ldots': '…',
           'dots': '…', 'cdots': '⋯', 'mid': '|', '|': '‖', 'lVert': '‖', 'rVert': '‖',
           'Vert': '‖', 'langle': '⟨', 'rangle': '⟩', 'lfloor': '⌊', 'rfloor': '⌋',
           'lceil': '⌈', 'rceil': '⌉', 'propto': '∝', 'sim': '∼', 'equiv': '≡', 'ell': 'ℓ',
           'top': '⊤', 'forall': '∀', 'exists': '∃', 'subset': '⊂', 'cup': '∪', 'cap': '∩',
           'emptyset': '∅', 'prime': '′', 'star': '⋆', 'ast': '∗', 'oplus': '⊕', 'otimes': '⊗',
           'uparrow': '↑', 'downarrow': '↓', 'mapsto': '↦', 'gg': '≫', 'll': '≪',
           'quad': '\u2003', 'qquad': '\u2003\u2003', ',': '\u2009', ';': '\u2005', ':': '\u2005',
           '!': '', ' ': ' ', '%': '%', '{': '{', '}': '}', '_': '_', '#': '#', '&': '&', '$': '$',
           'lbrace': '{', 'rbrace': '}', 'backslash': '\\', 'neg': '¬', 'land': '∧', 'lor': '∨'}
UPRIGHT_CMDS = {'log', 'ln', 'exp', 'max', 'min', 'sin', 'cos', 'tan', 'tanh', 'arg', 'det',
                'lim', 'sup', 'inf', 'Pr', 'dim', 'ker'}
TEXT_CMDS = {'text', 'mathrm', 'operatorname', 'textrm', 'mathsf', 'textit', 'mathit', 'mathbf',
             'textbf', 'boldsymbol', 'mathcal', 'mathbb'}
IGNORE_CMDS = {'left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'displaystyle', 'limits'}
ACCENTS = {'hat': '\u0302', 'bar': '\u0304', 'tilde': '\u0303', 'dot': '\u0307', 'vec': '\u20D7',
           'overline': '\u0305'}
PRECOMPOSED = {('hat', 'y'): 'ŷ', ('hat', 'Y'): 'Ŷ', ('hat', 'a'): 'â', ('hat', 'e'): 'ê',
               ('hat', 'o'): 'ô', ('hat', 'u'): 'û', ('hat', 'i'): 'î', ('hat', 'c'): 'ĉ',
               ('hat', 'g'): 'ĝ', ('hat', 'h'): 'ĥ', ('hat', 's'): 'ŝ', ('hat', 'w'): 'ŵ',
               ('bar', 'a'): 'ā', ('bar', 'e'): 'ē', ('bar', 'i'): 'ī', ('bar', 'o'): 'ō',
               ('bar', 'u'): 'ū', ('tilde', 'n'): 'ñ', ('tilde', 'a'): 'ã', ('tilde', 'o'): 'õ'}

PH_OPEN, PH_CLOSE = '\uE000', '\uE001'


def fb(text: str, font: str = 'Sans') -> str:
    """HTML-escape text and route glyphs missing from IBM Plex to DejaVu."""
    have = theme.glyphs('Mono' if font.startswith('Mono') else 'Sans')
    out = []
    for ch in text:
        o = ord(ch)
        esc = html.escape(ch, quote=False)
        if ch in '\n\t ' or 0xE000 <= o <= 0xE0FF or o in have:
            out.append(esc)
        else:
            out.append(f'<font name="Fallback">{esc}</font>')
    return ''.join(out)


class _Math:
    def __init__(self, s: str, plain: bool = False):
        self.s, self.i = s, 0
        self.plain = plain  # plain text output (titles, bookmarks) instead of paragraph markup

    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ''

    def group(self) -> str:
        """Read a {...} group or a single token and return its markup."""
        while self.peek() == ' ':
            self.i += 1
        if self.peek() == '{':
            depth, start = 0, self.i
            while self.i < len(self.s):
                c = self.s[self.i]
                depth += c == '{'
                depth -= c == '}'
                self.i += 1
                if depth == 0:
                    break
            return _Math(self.s[start + 1:self.i - 1], self.plain).run()
        return self.token()

    def raw_group(self) -> str:
        while self.peek() == ' ':
            self.i += 1
        if self.peek() == '{':
            depth, start = 0, self.i
            while self.i < len(self.s):
                c = self.s[self.i]
                depth += c == '{'
                depth -= c == '}'
                self.i += 1
                if depth == 0:
                    break
            return self.s[start + 1:self.i - 1]
        ch = self.peek()
        self.i += 1
        return ch

    def command(self) -> str:
        self.i += 1  # backslash
        m = re.match(r'[A-Za-z]+', self.s[self.i:])
        if not m:
            ch = self.peek()
            self.i += 1
            return fb(SYMBOLS.get(ch, ch))
        name = m.group(0)
        self.i += len(name)
        if name in GREEK:
            return fb(GREEK[name])
        if name in SYMBOLS:
            sym = SYMBOLS[name]
            if name in {'approx', 'neq', 'ne', 'leq', 'le', 'geq', 'ge', 'to', 'rightarrow',
                        'leftarrow', 'Rightarrow', 'in', 'notin', 'propto', 'sim', 'equiv',
                        'mapsto', 'times', 'pm', 'gg', 'll'}:
                return fb('\u2009' + sym + '\u2009')
            return fb(sym)
        if name in UPRIGHT_CMDS:
            return fb(name)
        if name in IGNORE_CMDS:
            if self.peek() in '()[]|.':
                ch = self.peek()
                self.i += 1
                return '' if ch == '.' else fb(ch)
            return ''
        if name in TEXT_CMDS:
            inner = self.raw_group()
            if name in {'mathbf', 'textbf', 'boldsymbol'}:
                return '<b>' + _Math(inner, self.plain).run(italic=False) + '</b>'
            if name in {'textit', 'mathit'}:
                return '<i>' + fb(inner) + '</i>'
            return _Math(inner, self.plain).run(italic=False) if name != 'text' else fb(inner)
        if name in ACCENTS:
            inner = self.raw_group().strip()
            key = (name, inner)
            if key in PRECOMPOSED:
                return '<i>' + fb(PRECOMPOSED[key]) + '</i>'
            body = _Math(inner, self.plain).run()
            letter = inner if len(inner) == 1 else GREEK.get(inner.lstrip('\\'), '') if inner.startswith('\\') else ''
            if not letter:
                return body
            if self.plain:
                return body + ACCENTS[name]
            # ReportLab cannot position combining marks, so the renderer draws the
            # accent centred over the preceding glyph (see render.draw_accent).
            return body + f'<onDraw name="_accent" label="{name} {letter}"/>'
        if name == 'frac':
            a, b = self.raw_group(), self.raw_group()
            fa, fbm = _Math(a, self.plain).run(), _Math(b, self.plain).run()
            simple = lambda t: re.fullmatch(r'[\w.]+', t) is not None  # noqa: E731
            fa = fa if simple(a) else '(' + fa + ')'
            fbm = fbm if simple(b) else '(' + fbm + ')'
            return fa + '/' + fbm
        if name == 'sqrt':
            inner = self.raw_group()
            body = _Math(inner, self.plain).run()
            atomic = re.search(r'[+\-*/=,\s]', inner) is None
            return fb('√') + (body if atomic else '(' + body + ')')
        return fb(name)

    def token(self) -> str:
        c = self.peek()
        if not c:
            return ''
        if c == '\\':
            return self.command()
        if c == '{':
            return self.group()
        self.i += 1
        if c.isalpha() and c.isascii():
            return f'<i>{c}</i>' if self.italic else c
        if c == '-':
            return '\u2212'
        if c == '*':
            return fb('∗')
        if c == "'":
            return '′'
        return fb(c)

    def run(self, italic: bool = True) -> str:
        self.italic = italic
        out = []
        while self.i < len(self.s):
            c = self.peek()
            if c in '_^':
                self.i += 1
                body = self.group()
                if c == '_':
                    out.append(f'<sub size="70%" rise="12%">{body}</sub>')
                else:
                    out.append(f'<super size="70%" rise="36%">{body}</super>')
                # keep an italic sub/superscript clear of following punctuation
                if '<i>' in body and self.peek() and not self.peek().isspace() and self.peek() not in '_^':
                    out.append('\u200a')
            elif c == ' ':
                self.i += 1
                out.append(' ')
            else:
                out.append(self.token())
        return ''.join(out)


def math_markup(expr: str, plain: bool = False) -> str:
    m = _Math(expr, plain)
    return m.run()


def rich(text: str, accent: str = theme.BLUE, code_color: str = '#1C3F63') -> str:
    """Convert inline markdown to ReportLab paragraph markup."""
    spans = []

    def stash(markup):
        spans.append(markup)
        return f'{PH_OPEN}{len(spans) - 1}{PH_CLOSE}'

    text = text.replace('\\$', PH_OPEN + 'D' + PH_CLOSE)
    text = re.sub(r'`([^`]+)`', lambda m: stash(
        f'<font name="Mono" color="{code_color}">{fb(m.group(1), "Mono")}</font>'), text)
    text = re.sub(r'\$([^$]+)\$', lambda m: stash(math_markup(m.group(1))), text)
    s = fb(text)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])', r'<i>\1</i>', s)
    s = re.sub(r'==(.+?)==', rf'<font name="SansSemi" color="{accent}">\1</font>', s)
    s = s.replace(PH_OPEN + 'D' + PH_CLOSE, '$')
    s = re.sub(PH_OPEN + r'(\d+)' + PH_CLOSE, lambda m: spans[int(m.group(1))], s)
    return s


def plain(text: str) -> str:
    """Strip inline markup (for HTML titles, bookmarks, and notes)."""
    text = text.replace('\\$', '$')
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'==(.+?)==', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\$([^$]+)\$', lambda m: re.sub(r'<[^>]+>', '', html.unescape(math_markup(m.group(1), plain=True))), text)
    text = re.sub(r'(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])', r'\1', text)
    return text
