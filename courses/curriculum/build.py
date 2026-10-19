#!/usr/bin/env python3
"""Build the course decks from each module's slides.md.

Outputs per module: slides.pdf (learner-facing), slides.html (offline deck with
notes, presenter view and overview), slides.pptx (PowerPoint presentation copy
with speaker notes), instructor_notes.pdf (slide thumbnail plus notes on every
page) and instructor_notes.md. With no arguments, all modules
are built and the two combined course PDFs are written to output/pdf.

No browser, network, TeX installation or presentation software is required.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # noqa: E402
from reportlab import rl_config
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.platypus import (Flowable, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, KeepTogether)

from curriculum import theme  # noqa: E402
from curriculum.markup import parse, plain, fb, SlideError  # noqa: E402,F401
from curriculum.render import make_pdf  # noqa: E402

ROOT = theme.ROOT
rl_config.invariant = 1
OUT = ROOT / 'combined_pdfs'
TMP = ROOT / 'build'
COURSES = [('machine_learning_course', 'Machine_Learning_Course', 'Machine learning for energy applications'),
           ('neural_networks_course', 'Neural_Networks_Course', 'Neural networks for energy applications')]


def deck_info(path: Path, slides) -> dict:
    label = slides[0].get('label', path.parent.name.replace('_', ' '))
    course = next((c[2] for c in COURSES if c[0] in str(path)), 'Machine learning for energy applications')
    return {'label': label, 'course': slides[0].get('footer', course),
            'footer_label': f"{label} · {plain(slides[0]['title'])}"}


# ---------------------------------------------------------------- HTML deck
VIEWER_CSS = """
:root{--bg:#0B1E31;--ink:#0F2A44;--sky:#8EC5EC;--mist:#C4D8E7}
html,body{margin:0;height:100%;background:var(--bg);overflow:hidden;font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
#stage{position:fixed;inset:0}
body.notes #stage{right:36%}
.slide{display:none;position:absolute;inset:0;padding:14px;box-sizing:border-box;align-items:center;justify-content:center}
.slide.active{display:flex}
.slide svg{width:100%;height:100%;filter:drop-shadow(0 8px 28px rgba(0,0,0,.35))}
.slide aside{display:none}
body.notes .slide.active aside{display:block;position:fixed;top:0;right:0;bottom:0;width:36%;box-sizing:border-box;padding:26px 30px 60px;background:#fff;color:var(--ink);white-space:pre-wrap;overflow:auto;font-size:15px;line-height:1.55;border-left:6px solid #087F8C}
aside h2{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#087F8C;margin:0 0 10px}
#bar{position:fixed;left:0;top:0;height:3px;background:var(--sky);z-index:5;transition:width .15s}
#hud{position:fixed;right:18px;bottom:16px;color:#C4D8E7;background:rgba(11,30,49,.78);padding:3px 10px;border-radius:12px;font-size:12px;z-index:5;user-select:none;opacity:0;transition:opacity .4s}
#hud.show{opacity:1}
body.notes #hud{right:calc(36% + 14px)}
#help{position:fixed;inset:0;display:none;align-items:center;justify-content:center;background:rgba(6,18,30,.82);z-index:9;color:#fff}
#help div{background:#0F2A44;border:1px solid #2F6390;border-radius:12px;padding:26px 34px;min-width:340px}
#help kbd{display:inline-block;min-width:26px;padding:1px 7px;margin-right:10px;border-radius:5px;background:#1E4468;color:var(--sky);text-align:center;font:600 13px ui-monospace,monospace}
body.help #help{display:flex}
body.grid{overflow:auto}
body.grid #stage{position:static;display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px;padding:18px}
body.grid .slide{display:block;position:relative;inset:auto;padding:0;aspect-ratio:16/9;cursor:pointer;outline:2px solid transparent;border-radius:4px}
body.grid .slide.active{outline-color:var(--sky)}
body.grid .slide svg{filter:none}
body.grid .slide::after{content:attr(data-n);position:absolute;left:6px;bottom:4px;font:600 11px system-ui;color:#fff;background:rgba(15,42,68,.8);padding:1px 6px;border-radius:4px}
@media print{html,body{overflow:visible;height:auto;background:#fff}#stage{position:static}.slide,.slide.active{display:block;position:relative;inset:auto;padding:0;height:auto;page-break-after:always}.slide svg{filter:none;height:auto}aside,#hud,#bar,#help{display:none!important}}
"""

VIEWER_JS = r"""
const slides=[...document.querySelectorAll('.slide')];
const bar=document.getElementById('bar'),hud=document.getElementById('hud');
function readHash(){const m=/^#s(\d+)$/.exec(location.hash);if(!m)return 0;const k=Number(m[1]);return Number.isFinite(k)?Math.max(0,Math.min(slides.length-1,k-1)):0;}
let i=readHash(),pw=null,t0=Date.now();
function show(){slides.forEach((s,j)=>s.classList.toggle('active',j===i));history.replaceState(null,'','#s'+(i+1));
 bar.style.width=((i+1)/slides.length*100)+'%';hud.textContent=(i+1)+' / '+slides.length;
 hud.classList.add('show');clearTimeout(window._hudT);window._hudT=setTimeout(()=>hud.classList.remove('show'),1800);syncPresenter();}
function go(k){i=Math.max(0,Math.min(slides.length-1,k));show();}
function onKey(e){if(e.metaKey||e.ctrlKey||e.altKey)return;const k=e.key;
 if(document.body.classList.contains('help')&&k!=='?'){document.body.classList.remove('help');return;}
 if(['ArrowRight','ArrowDown','PageDown',' '].includes(k)){go(i+1);e.preventDefault();}
 else if(['ArrowLeft','ArrowUp','PageUp','Backspace'].includes(k)){go(i-1);e.preventDefault();}
 else if(k==='Home'){go(0);}else if(k==='End'){go(slides.length-1);}
 else if(k.toLowerCase()==='n'){document.body.classList.toggle('notes');}
 else if(k.toLowerCase()==='g'){document.body.classList.toggle('grid');if(document.body.classList.contains('grid'))slides[i].scrollIntoView({block:'center'});}
 else if(k.toLowerCase()==='f'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();}
 else if(k.toLowerCase()==='p'){openPresenter();}
 else if(k==='?'){document.body.classList.toggle('help');}
 else if(k==='Escape'){document.body.classList.remove('grid','help');}}
document.addEventListener('keydown',onKey);
window.addEventListener('hashchange',()=>{i=readHash();show();});
slides.forEach((s,j)=>{s.dataset.n=j+1;s.addEventListener('click',e=>{if(e.target.closest('aside'))return;
 if(document.body.classList.contains('grid')){document.body.classList.remove('grid');go(j);return;}
 const r=s.getBoundingClientRect();go(e.clientX-r.left>r.width/3?i+1:i-1);});});
let tx=null;document.addEventListener('touchstart',e=>{tx=e.touches[0].clientX},{passive:true});
document.addEventListener('touchend',e=>{if(tx===null)return;const d=e.changedTouches[0].clientX-tx;if(Math.abs(d)>50)go(d<0?i+1:i-1);tx=null;});
function openPresenter(){pw=window.open('','presenter','width=1280,height=800');if(!pw)return;t0=Date.now();
 pw.document.write('<!doctype html><title>Presenter view</title><style>body{margin:0;background:#0B1E31;color:#fff;font:15px/1.5 system-ui,sans-serif;display:grid;grid-template-columns:62% 38%;grid-template-rows:auto 1fr;height:100vh}#cur{grid-row:1/3;padding:18px}#cur svg{width:100%;height:auto;background:#fff}#side{padding:18px 18px 0 0;display:flex;flex-direction:column;gap:12px;min-height:0}#nxt svg{width:100%;height:auto;opacity:.9}#meta{display:flex;justify-content:space-between;color:#8EC5EC;font-weight:600}#nt{flex:1;overflow:auto;background:#fff;color:#0F2A44;border-radius:8px;padding:14px 18px;white-space:pre-wrap;font-size:16px}</style><div id=cur></div><div id=side><div id=meta><span id=pos></span><span id=tm></span></div><div id=nxt></div><div id=nt></div></div>');
 pw.document.close();pw.document.body.insertAdjacentHTML('afterbegin',document.getElementById('glyphs').outerHTML);pw.document.addEventListener('keydown',onKey);syncPresenter();}
function syncPresenter(){if(!pw||pw.closed)return;const d=pw.document;
 d.getElementById('cur').innerHTML=slides[i].querySelector('svg').outerHTML;
 d.getElementById('nxt').innerHTML=i+1<slides.length?slides[i+1].querySelector('svg').outerHTML:'<p>End of deck</p>';
 const a=slides[i].querySelector('aside');d.getElementById('nt').textContent=a?a.textContent:'';
 d.getElementById('pos').textContent='Slide '+(i+1)+' of '+slides.length;}
setInterval(()=>{if(!pw||pw.closed)return;const s=Math.floor((Date.now()-t0)/1000);
 pw.document.getElementById('tm').textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');},1000);
show();
"""


HEAVY_SLIDE_BYTES = 1_200_000   # slides whose vector SVG exceeds this are embedded as images
HEAVY_SLIDE_DPI = 216            # 2880 x 1620 px: sharp on 4K projectors


def raster_svg(page) -> str:
    """A slide as one high-resolution PNG inside an SVG wrapper."""
    import base64
    png = page.get_pixmap(dpi=HEAVY_SLIDE_DPI).tobytes('png')
    data = base64.b64encode(png).decode('ascii')
    return ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            'version="1.1" width="100%" height="100%" viewBox="0 0 960 540">'
            f'<image width="960" height="540" xlink:href="data:image/png;base64,{data}"/></svg>')


def make_html(pdf_path: Path, slides, target: Path, deck) -> None:
    doc = fitz.open(pdf_path)
    sections = []
    glyphs: dict[str, str] = {}  # outline path data -> shared glyph id
    for i, (page, s) in enumerate(zip(doc, slides)):
        svg = page.get_svg_image(text_as_path=True)
        if len(svg) > HEAVY_SLIDE_BYTES:
            # Dense scatter plots expand to tens of thousands of SVG paths; an image
            # keeps the offline deck fast to open. The PDF stays fully vector.
            svg = raster_svg(page)
        # Text is drawn as <use> references to glyph outlines. Identical outlines
        # recur on every slide, so they are stored once per deck in #glyphs.
        local = {}

        def grab_path(m):
            key = f'<path id="@" d="{m[2]}"/>'
            local[m[1]] = glyphs.setdefault(key, f'g{len(glyphs)}')
            return ''

        def grab_group(m):
            key = f'<g id="@">{m[2]}</g>'
            local[m[1]] = glyphs.setdefault(key, f'g{len(glyphs)}')
            return ''
        svg = re.sub(r'<g id="(font_[^"]+)">(.*?)</g>\s*', grab_group, svg, flags=re.S)
        svg = re.sub(r'<path id="(font_[^"]+)" d="([^"]*)"\s*/>\s*', grab_path, svg)
        svg = re.sub(r'(xlink:href=")#(font_[^"]+)"', lambda m: f'{m[1]}#{local[m[2]]}"', svg)
        svg = re.sub(r'\sdata-text="[^"]*"', '', svg)
        svg = svg.replace('<defs>\n</defs>', '').replace('<defs></defs>', '')
        prefix = f'p{i + 1}_'
        svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m[1]}"', svg)
        svg = re.sub(r'((?:xlink:)?href=")#(?!g\d+")([^"]+)"', lambda m: f'{m[1]}#{prefix}{m[2]}"', svg)
        svg = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}{m[1]})', svg)
        svg = re.sub(r'<svg([^>]*?)\swidth="[^"]*"\s+height="[^"]*"', r'<svg\1 width="100%" height="100%"', svg, count=1)
        svg = re.sub(r'<\?xml[^>]*>\s*', '', svg)
        note = s['notes'].strip()
        if s.get('source'):
            note += '\n\nReference: ' + s['source']
        title = html.escape(plain(s['title']))
        sections.append(f'<section aria-label="{title}" class="slide" id="s{i + 1}">{svg}'
                        f'<aside><h2>Slide {i + 1} · {title}</h2>{html.escape(note)}</aside></section>')
    doc.close()
    help_rows = ''.join(f'<p><kbd>{k}</kbd>{v}</p>' for k, v in [
        ('→', 'next (also Space, click right, swipe)'), ('←', 'previous'), ('N', 'instructor notes'),
        ('P', 'presenter window with timer'), ('G', 'overview grid'), ('F', 'fullscreen'),
        ('Home', 'first slide / End: last'), ('?', 'this help')])
    library = ('<svg id="glyphs" width="0" height="0" style="position:absolute" aria-hidden="true"><defs>'
               + ''.join(markup.replace('id="@"', f'id="{gid}"', 1) for markup, gid in glyphs.items())
               + '</defs></svg>')
    target.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(deck["label"])}: {html.escape(plain(slides[0]["title"]))}</title>'
        f'<style>{VIEWER_CSS}</style></head><body>{library}<div id="bar"></div><main id="stage">'
        + '\n'.join(sections) +
        f'</main><div id="hud"></div><div id="help"><div><h3>Keyboard</h3>{help_rows}</div></div>'
        f'<script>{VIEWER_JS}</script></body></html>')


# ---------------------------------------------------------- instructor notes
class SlideBox(Flowable):
    """Placeholder that records where a slide thumbnail should be placed."""

    def __init__(self, index, width, height, sink):
        super().__init__()
        self.index, self.width, self.height, self.sink = index, width, height, sink

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        x, y = self.canv.absolutePosition(0, 0)
        self.sink.append((self.canv.getPageNumber() - 1, self.index, x, y, self.width, self.height))
        self.canv.setStrokeColor(colors.HexColor(theme.RULE))
        self.canv.setLineWidth(.8)
        self.canv.rect(0, 0, self.width, self.height)


def note_markup(text: str) -> str:
    s = fb(text)
    return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)


def make_notes_pdf(pdf_path: Path, slides, target: Path, deck) -> None:
    sink = []
    st_h = ParagraphStyle('h', fontName='SansSemi', fontSize=13, leading=17, textColor=colors.HexColor(theme.INK))
    st_k = ParagraphStyle('k', fontName='SansSemi', fontSize=8.5, leading=11, textColor=colors.HexColor(theme.TEAL))
    st_b = ParagraphStyle('b', fontName='Sans', fontSize=10.5, leading=15.2, textColor=colors.HexColor(theme.INK),
                          spaceAfter=6)
    st_r = ParagraphStyle('r', parent=st_b, fontSize=9, leading=12.5, textColor=colors.HexColor(theme.MUTED))
    story = []
    width = A4[0] - 2 * 56
    for i, s in enumerate(slides):
        head = [Paragraph(f'{html.escape(deck["label"]).upper()} · SLIDE {i + 1} OF {len(slides)}', st_k),
                Spacer(1, 3), Paragraph(note_markup(plain(s['title'])), st_h), Spacer(1, 10),
                SlideBox(i, width, width * 9 / 16, sink), Spacer(1, 14)]
        story.append(KeepTogether(head))
        for block in re.split(r'\n\s*\n', s['notes'].strip()):
            block = ' '.join(line.strip() for line in block.splitlines())
            if block:
                story.append(Paragraph(note_markup(block), st_b))
        if s.get('source'):
            story.append(Paragraph('Reference: ' + fb(s['source']), st_r))
        if i < len(slides) - 1:
            story.append(PageBreak())

    def footer(canv, doc_):
        canv.setFont('Sans', 8)
        canv.setFillColor(colors.HexColor(theme.MUTED))
        canv.drawString(56, 28, f'{deck["label"]} · instructor notes')
        canv.drawRightString(A4[0] - 56, 28, str(doc_.page))

    tmp = target.with_suffix('.tmp.pdf')
    SimpleDocTemplate(str(tmp), pagesize=A4, leftMargin=56, rightMargin=56, topMargin=48, bottomMargin=48,
                      title=f'{deck["label"]}: instructor notes', author='Course teaching materials'
                      ).build(story, onFirstPage=footer, onLaterPages=footer)
    notes = fitz.open(tmp)
    src = fitz.open(pdf_path)
    for page_no, idx, x, y, w, h in sink:
        page = notes[page_no]
        rect = fitz.Rect(x, page.rect.height - y - h, x + w, page.rect.height - y)
        page.show_pdf_page(rect, src, idx, keep_proportion=True)
    notes.set_toc([[1, f'{i + 1}. {plain(s["title"])}', next(p for p, k, *_ in sink if k == i) + 1]
                   for i, s in enumerate(slides)])
    notes.save(target, deflate=True, garbage=4, no_new_id=True)
    notes.close()
    src.close()
    tmp.unlink()


def make_notes_md(slides, target: Path, deck) -> None:
    guide = [f'# {deck["label"]}: instructor notes', '',
             'Slide numbers match slides.pdf and slides.html. In the HTML deck press N for notes, '
             'P for a presenter window with timer, and G for an overview. '
             'instructor_notes.pdf shows each slide with its notes.', '']
    for i, s in enumerate(slides):
        guide.extend([f'## {i + 1}. {plain(s["title"])}', '', s['notes'].strip(), ''])
        if s.get('source'):
            guide.extend(['Reference: ' + s['source'], ''])
    target.write_text('\n'.join(guide))


# ---------------------------------------------------------------- driver
def build_one(path: Path, html_out: bool = True, notes_out: bool = True, pptx_out: bool = True) -> dict:
    path = Path(path).resolve()
    slides = parse(path)
    deck = deck_info(path, slides)
    missing = [i + 1 for i, s in enumerate(slides) if not s['notes'].strip()]
    report = make_pdf(path.with_suffix('.pdf'), slides, deck)
    if html_out:
        make_html(path.with_suffix('.pdf'), slides, path.with_suffix('.html'), deck)
    if notes_out:
        make_notes_pdf(path.with_suffix('.pdf'), slides, path.with_name('instructor_notes.pdf'), deck)
        make_notes_md(slides, path.with_name('instructor_notes.md'), deck)
    if pptx_out:
        from curriculum.export_pptx import export
        export(path)
    warnings = []
    for entry in report:
        for f in entry.get('figures', []):
            if f['scale'] < .72:
                warnings.append(f"slide {entry['slide']}: figure {f['name']} scaled to {f['scale']:.2f} "
                                f"(text will look small; redraw it at the target size)")
            if f.get('min_text_pt') is not None and f['min_text_pt'] < 8:
                warnings.append(f"slide {entry['slide']}: figure {f['name']} has text at {f['min_text_pt']} pt "
                                f"on the slide (enlarge the smallest labels to at least 11 pt in the figure)")
            if f['box_fill'] < .55:
                warnings.append(f"slide {entry['slide']}: figure {f['name']} fills only {f['box_fill']:.0%} "
                                f"of its box (aspect ratio does not suit the layout)")
    if missing:
        warnings.append(f'slides without notes: {missing}')
    try:
        shown = str(path.relative_to(ROOT))
    except ValueError:
        shown = str(path)
    result = {'path': shown, 'label': deck['label'], 'count': len(slides),
              'warnings': warnings, 'slides': report}
    TMP.joinpath('reports').mkdir(parents=True, exist_ok=True)
    TMP.joinpath('reports', f'{path.parent.name}.json').write_text(json.dumps(result, indent=2))
    print(f'{path.parent.name}: {len(slides)} slides, {len(warnings)} warnings', flush=True)
    for w in warnings:
        print('  warning:', w, flush=True)
    return result


def combine() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for course, name, title in COURSES:
        combined = fitz.open()
        toc = []
        for p in sorted((ROOT / course).glob('*/slides.pdf')):
            d = fitz.open(p)
            offset = len(combined)
            label = parse(p.with_suffix('.md'))[0].get('label', p.parent.name)
            toc.append([1, label, offset + 1])
            for level, t, page, *_ in d.get_toc():
                if page > 1:
                    toc.append([min(level + 1, 3), t, page + offset])
            combined.insert_pdf(d)
            d.close()
        combined.set_toc(toc)
        combined.set_metadata({'title': title, 'author': 'Course teaching materials'})
        combined.save(OUT / f'{name}.pdf', deflate=True, garbage=4, no_new_id=True)
        combined.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path, help='slides.md files (default: all modules)')
    parser.add_argument('--pdf-only', action='store_true', help='skip HTML, notes and PPTX (faster iteration)')
    parser.add_argument('--no-pptx', action='store_true', help='skip the PowerPoint presentation copy')
    args = parser.parse_args()
    paths = args.paths or sorted(ROOT.glob('*_course/*/slides.md'))
    failures = []
    for p in paths:
        try:
            build_one(p, html_out=not args.pdf_only, notes_out=not args.pdf_only,
                      pptx_out=not (args.pdf_only or args.no_pptx))
        except SlideError as e:
            failures.append(str(e))
            print('ERROR:', e, flush=True)
    if failures:
        sys.exit(1)
    if not args.paths:
        combine()


if __name__ == '__main__':
    main()
