#!/usr/bin/env python3
"""Build the offline course website: index.html plus HTML pages under site/.

    python3 curriculum/build_site.py

index.html (package root) is the entry point. It links every deck as HTML, PDF
and PowerPoint, the combined course PDFs, and HTML renderings of README.md,
INSTRUCTOR_GUIDE.md, curriculum/AUTHORING.md and each module's
instructor_notes.md. Every page uses only local files (IBM Plex is embedded in
site/style.css) and web addresses are shown as text, not links, so the site
works offline straight from the folder. Run it again after rebuilding the decks.
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import sys
import xml.etree.ElementTree as etree
from pathlib import Path
from urllib.parse import quote, unquote

import markdown
from markdown.extensions import Extension
from markdown.extensions.toc import TocExtension
from markdown.treeprocessors import Treeprocessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curriculum import theme  # noqa: E402
from curriculum.markup import parse, plain  # noqa: E402

ROOT = theme.ROOT
SITE = ROOT / 'site'
INDEX = ROOT / 'index.html'
ARCHIVE = ROOT / 'courses.zip'  # added by hand next to index.html; the index links to it
COURSES = [('machine_learning_course', 'ml', 'Machine learning for energy applications', 'Machine_Learning_Course.pdf'),
           ('neural_networks_course', 'nn', 'Neural networks for energy applications', 'Neural_Networks_Course.pdf')]
DOCS = [  # source, page, title, summary on the index
    ('README.md', 'readme.html', 'Read me',
     'What the package contains, how the folders are laid out, and how to rebuild the decks.'),
    ('INSTRUCTOR_GUIDE.md', 'instructor-guide.html', 'Instructor guide',
     'Day plans with times, laboratory preparation and runtimes, presenting, and assessment.'),
    ('curriculum/AUTHORING.md', 'authoring.html', 'Authoring guide',
     'Slide syntax, layouts, the figure toolkit and the editorial standards for changing a deck.'),
]
FACES = [('IBM Plex Sans', 'Sans', 400, 'normal'), ('IBM Plex Sans', 'SansItalic', 400, 'italic'),
         ('IBM Plex Sans', 'SansSemi', 600, 'normal'), ('IBM Plex Sans', 'SansSemiItalic', 600, 'italic'),
         ('IBM Plex Mono', 'Mono', 400, 'normal'), ('IBM Plex Mono', 'MonoSemi', 600, 'normal')]
TAGS = {'lab': 'Lab', 'debrief': 'Debrief', 'exercise': 'Exercise'}


def esc(text) -> str:
    return html.escape(str(text))


def href(target: Path, start: Path, fragment: str = '') -> str:
    """Relative URL from directory `start` to `target`."""
    rel = os.path.relpath(target, start).replace(os.sep, '/')
    return quote(rel) + (f'#{fragment}' if fragment else '')


def count(n: int, word: str) -> str:
    return f'{n} {word}' + ('' if n == 1 else 's')


def size(path: Path) -> str:
    n = path.stat().st_size
    if n >= 10e6:
        return f'{n / 1e6:.0f} MB'
    return f'{n / 1e6:.1f} MB' if n >= 1e6 else f'{max(n / 1e3, 1):.0f} kB'


# ---------------------------------------------------------------- content
def modules() -> list[dict]:
    out = []
    for course, key, _, _ in COURSES:
        for src in sorted((ROOT / course).glob('*/slides.md')):
            slides = parse(src)
            label = slides[0].get('label', src.parent.name.replace('_', ' '))
            unit = label.rpartition('/')[2].strip()  # 'Module 1', 'Day 3'
            kind, _, number = unit.rpartition(' ')
            out.append({
                'dir': src.parent, 'course': key, 'label': label, 'unit': unit,
                'kind': kind or unit, 'number': number if kind else '',
                'title': plain(slides[0]['title']),
                'subtitle': plain(slides[0]['body'][0]['text']) if slides[0]['body'] else '',
                'types': [s['type'] for s in slides],
                'parts': sum(s['type'] == 'section' for s in slides),
                'labs': sum(s['type'] == 'lab' for s in slides),
                'exercises': sum(s['type'] == 'exercise' for s in slides),
                'figures': {f for s in slides for f in s['figures']},
                'page': SITE / 'notes' / (src.parent.name.lower().replace('_', '-') + '.html'),
                'notebooks': notebooks(src.parent),
            })
    return out


def notebook_title(path: Path) -> str:
    """The first heading of the notebook's first Markdown cell, without the module prefix."""
    for cell in json.loads(path.read_text()).get('cells', []):
        if cell.get('cell_type') == 'markdown':
            m = re.match(r'#+\s*(.+)', ''.join(cell.get('source', '')).strip())
            if not m:
                return ''
            title = re.sub(r'^(Module|Day) \d+:\s*', '', plain(m.group(1).strip()))
            return re.sub(r'\s+—\s+Hands-On Exercise$', '', title).replace(' -- ', ' — ')
    return ''


def notebooks(module_dir: Path) -> list[dict]:
    """Exercise notebooks as student/solution pairs, with the data files in the same folder."""
    root = module_dir / 'exercises'
    out = []
    for student in sorted(root.rglob('*_Student.ipynb')):
        if any(part.startswith('.') for part in student.relative_to(root).parts):
            continue
        stem = student.name.removesuffix('_Student.ipynb')
        solution = student.with_name(f'{stem}_Solution.ipynb')
        folder = student.parent
        files = folder.rglob('*') if folder != root else folder.glob('*')
        data = sorted(f for f in files if f.is_file() and f.suffix != '.ipynb'
                      and not any(part.startswith('.') for part in f.relative_to(folder).parts))
        out.append({'title': notebook_title(student) or stem.replace('_', ' '), 'student': student,
                    'solution': solution if solution.exists() else None, 'folder': folder, 'data': data,
                    'shown': folder.relative_to(module_dir).as_posix() + '/'})
    return out


class _Links(Treeprocessor):
    """Point links at the rendered pages, relative to the output page; show web addresses as text."""

    def __init__(self, md, source_dir: Path, out_dir: Path, pages: dict[Path, Path]):
        super().__init__(md)
        self.source_dir, self.out_dir, self.pages = source_dir, out_dir, pages

    def run(self, root):
        for a in list(root.iter('a')):
            url = a.get('href', '')
            if re.match(r'[a-z]+:', url):
                a.tag = 'span'
                a.attrib.clear()
                a.set('class', 'ext')
                shown = etree.SubElement(a, 'span', {'class': 'url'})
                shown.text = re.sub(r'^https?://', '', url).rstrip('/')
            elif url and not url.startswith('#'):
                path, _, fragment = url.partition('#')
                target = (self.source_dir / unquote(path)).resolve()
                a.set('href', href(self.pages.get(target, target), self.out_dir, fragment))


class _LinkExtension(Extension):
    def __init__(self, **options):
        super().__init__()
        self.options = options

    def extendMarkdown(self, md):
        md.treeprocessors.register(_Links(md, **self.options), 'course_links', 4)  # after inline and toc


def render_markdown(source: Path, out: Path, pages: dict[Path, Path]) -> tuple[str, list]:
    md = markdown.Markdown(extensions=[
        'tables', 'fenced_code', 'sane_lists', TocExtension(toc_depth='2-3'),
        _LinkExtension(source_dir=source.parent, out_dir=out.parent, pages=pages)])
    body = md.convert(source.read_text())
    body = body.replace('<table>', '<div class="table"><table>').replace('</table>', '</table></div>')
    return body, md.toc_tokens


def doc_toc(tokens) -> str:
    def items(ts):
        return ''.join(f'<li><a href="#{t["id"]}"><span class="t">{t["html"]}</span></a>'
                       + (f'<ol>{items(t["children"])}</ol>' if t['children'] else '') + '</li>' for t in ts)
    return f'<ol>{items(tokens)}</ol>'


def inline(text: str) -> str:
    """Notes are plain text with **bold**, exactly as in instructor_notes.pdf."""
    return re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', esc(text))


def note_blocks(text: str) -> str:
    out, items = [], []

    def flush():
        if items:
            out.append(f'<ol start="{items[0][0]}">' + ''.join(f'<li>{t}</li>' for _, t in items) + '</ol>')
            items.clear()

    for block in re.split(r'\n\s*\n', text.strip()):
        block = ' '.join(line.strip() for line in block.splitlines())
        numbered = re.match(r'(\d+)\.\s+(.*)', block)
        if numbered and (not items or int(numbered.group(1)) == items[-1][0] + 1):
            items.append((int(numbered.group(1)), inline(numbered.group(2))))
            continue
        flush()
        if numbered:
            items.append((int(numbered.group(1)), inline(numbered.group(2))))
        elif block.startswith('Reference: '):
            urls = ''.join(f' <span class="url">{esc(re.sub(r"^https?://", "", u.strip()))}</span>'
                           for u in block[len('Reference: '):].split(' ; '))
            out.append(f'<p class="ref"><span>Reference</span>{urls}</p>')
        elif block:
            out.append(f'<p>{inline(block)}</p>')
    flush()
    return '\n'.join(out)


def render_notes(m: dict) -> tuple[str, str]:
    text = (m['dir'] / 'instructor_notes.md').read_text()
    chunks = re.split(r'^## (\d+)\. (.*)$', text, flags=re.M)
    heads = [(int(n), t.strip()) for n, t in zip(chunks[1::3], chunks[2::3])]
    types = m['types']
    if len(types) != len(heads):
        print(f'warning: {m["label"]}: {len(heads)} notes headings but {len(types)} slides; '
              f'rebuild the deck before the site', file=sys.stderr)
        types = [''] * len(heads)
    out_dir = m['page'].parent
    deck = m['dir'] / 'slides.html'
    body = [note_blocks(re.sub(r'\A# .*\n', '', chunks[0]))]  # the page header replaces the title line
    groups = [['<span class="t">Opening</span>', '', []]]
    part = 0
    for (n, title), kind, notes in zip(heads, types, chunks[3::3]):
        tag = TAGS.get(kind, '')
        if kind == 'section':
            part += 1
            tag = f'Part {part}'
            groups.append([f'<span class="k">Part {part}</span><span class="t">{esc(title)}</span>', f's{n}', []])
        else:
            groups[-1][2].append(f'<li><a href="#s{n}" class="{kind}"><span class="n">{n}</span>'
                                 f'<span class="t">{esc(title)}</span></a></li>')
        chip = f'<span class="tag {kind}">{tag}</span>' if tag else ''
        body.append(f'<h2 id="s{n}" class="slide-h{" is-section" if kind == "section" else ""}">'
                    f'<span class="n">{n}</span><span class="t">{inline(title)}</span>{chip}'
                    f'<a class="go" href="{href(deck, out_dir, f"s{n}")}" '
                    f'title="Open slide {n} in the HTML deck">Open slide</a></h2>')
        body.append(note_blocks(notes))
    toc = ''.join(
        f'<li class="toc-group">'
        + (f'<a href="#{gid}" class="part">{head}</a>' if gid else f'<p class="part">{head}</p>')
        + (f'<ol>{"".join(entries)}</ol>' if entries else '') + '</li>'
        for head, gid, entries in groups if gid or entries)
    return '\n'.join(body), f'<ol>{toc}</ol>'


# ---------------------------------------------------------------- pages
def links_group(label: str, entries: list[tuple[str, str, str]], primary: bool = False) -> str:
    first = ' class="primary"' if primary else ''
    buttons = ''.join(f'<a href="{url}"{first if i == 0 else ""} title="{esc(tip)}">{esc(text)}</a>'
                      for i, (text, url, tip) in enumerate(entries))
    return f'<div class="group"><span>{esc(label)}</span><div class="seg">{buttons}</div></div>'


def module_links(m: dict, start: Path, notes_html: bool = True) -> str:
    d = m['dir']
    slides = [('HTML', href(d / 'slides.html', start), 'Offline presentation deck with notes and presenter view'),
              ('PDF', href(d / 'slides.pdf', start), 'Learner-facing deck'),
              ('PPTX', href(d / 'slides.pptx', start), 'PowerPoint copy with speaker notes')]
    notes = [('HTML', href(m['page'], start), 'Instructor notes as a web page')] if notes_html else []
    notes += [('PDF', href(d / 'instructor_notes.pdf', start), 'Every slide with its notes, for printing'),
              ('MD', href(d / 'instructor_notes.md', start), 'The notes as plain text')]
    return links_group('Slides', slides, primary=True) + links_group('Notes', notes)


def exercise_list(m: dict, start: Path) -> str:
    """Collapsible list of the module's exercise notebooks, as downloads."""
    items = []
    for nb in m['notebooks']:
        files = [('Student', nb['student'], 'Student notebook with the tasks to complete')]
        if nb['solution']:
            files.append(('Solution', nb['solution'], 'Worked solution notebook'))
        buttons = ''.join(f'<a href="{href(path, start)}" download data-nb="{esc(path.relative_to(ROOT).as_posix())}" '
                          f'title="{esc(tip)}">{label}</a>' for label, path, tip in files)
        data = ' · '.join(f'<a href="{href(f, start)}" download>{esc(f.relative_to(nb["folder"]).as_posix())}</a>'
                          f'&nbsp;<span>{size(f)}</span>' for f in nb['data'])
        items.append(f'<li><div class="ex-info"><strong>{esc(nb["title"])}</strong>'
                     f'<span class="ex-path">{esc(nb["shown"])}</span>'
                     + (f'<span class="ex-data">Data: {data}</span>' if data else '')
                     + f'</div><div class="seg">{buttons}</div></li>')
    if not items:
        return ''
    return (f'<details class="exercises"><summary><b>Exercise notebooks</b>'
            f'<span>{count(len(items), "exercise")} · student and solution</span></summary>'
            f'<ol>{"".join(items)}</ol></details>')


def notebook_loader(start: Path) -> str:
    """Browsers show .ipynb files opened from disk as text, so file:// pages save embedded copies instead."""
    return (f"<script>if (location.protocol === 'file:') {{ var s = document.createElement('script'); "
            f"s.src = '{href(SITE / 'notebooks.js', start)}'; document.head.appendChild(s); }}</script>\n")


def shell(title: str, out: Path, content: str, accent: str = '', current: str = '', script: bool = False,
          downloads: bool = False) -> str:
    start = out.parent
    nav = ''.join(f'<a href="{href(SITE / page, start)}"' + (' aria-current="page"' if page == current else '')
                  + f'>{esc(name)}</a>' for _, page, name, _ in DOCS)
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<meta name="color-scheme" content="light dark">\n<title>{esc(title)}</title>\n'
            f'<link rel="stylesheet" href="{href(SITE / "style.css", start)}">\n'
            + (notebook_loader(start) if downloads else '') + '</head>\n'
            f'<body class="doc {accent}">\n'
            f'<header class="topbar"><div class="bar">'
            f'<a class="home" href="{href(INDEX, start)}">Course materials</a>'
            f'<nav class="topnav" aria-label="Guides">{nav}</nav></div></header>\n'
            f'{content}\n' + (f'<script>{SCRIPT}</script>\n' if script else '') + '</body>\n</html>\n')


def doc_page(title: str, out: Path, body: str, toc: str, source_note: str,
             accent: str = '', current: str = '', head: str = '', pager: str = '', downloads: bool = False) -> str:
    content = (f'<div class="layout">\n'
               f'<nav class="toc side" aria-label="Contents"><p class="toc-title">Contents</p>{toc}</nav>\n'
               f'<main>\n{head}'
               f'<details class="toc toc-mobile"><summary>Contents</summary>{toc}</details>\n'
               f'<article class="prose">\n{body}\n</article>\n{pager}'
               f'<footer class="page-foot">{source_note}</footer>\n</main>\n</div>')
    return shell(title, out, content, accent, current, script=True, downloads=downloads)


def notes_page(m: dict, prev: dict | None, nxt: dict | None, course_title: str) -> str:
    out = m['page']
    body, toc = render_notes(m)
    head = (f'<header class="page-head">'
            f'<p class="kicker">{esc(course_title)} · {esc(m["unit"])} · Instructor notes</p>'
            f'<h1>{esc(m["title"])}</h1><p class="sub">{esc(m["subtitle"])}</p>'
            f'<div class="links">{module_links(m, out.parent, notes_html=False)}</div>'
            f'{exercise_list(m, out.parent)}'
            f'<p class="caution">The notes contain the worked answer to every exercise: '
            f'keep them off the shared screen while learners work.</p></header>\n')

    def side(other, rel, arrow):
        if not other:
            return '<span></span>'
        return (f'<a class="{rel}" href="{href(other["page"], out.parent)}" rel="{rel}">'
                f'<span>{arrow} {esc(other["label"])}</span>{esc(other["title"])}</a>')
    pager = f'<nav class="pager" aria-label="Modules">{side(prev, "prev", "←")}{side(nxt, "next", "→")}</nav>\n'
    source = href(m['dir'] / 'instructor_notes.md', out.parent)
    note = f'Rendered from <a href="{source}">instructor_notes.md</a>. Slide numbers match the decks.'
    return doc_page(f'{m["title"]}: instructor notes · {m["label"]}', out, body, toc, note,
                    accent=m['course'], head=head, pager=pager, downloads=bool(m['notebooks']))


def index_page(mods: list[dict], readme: str) -> str:
    title = re.search(r'^# (.+)$', readme, re.M).group(1).strip()
    lead = next(b for b in re.split(r'\n\s*\n', readme.split('\n', 1)[1].strip())
                if not b.startswith(('#', '-', '|', '```')))
    lead = re.sub(r'^<p>|</p>$', '', markdown.markdown(lead))
    blurbs = {t.lower(): d for t, d in re.findall(r'^- \*\*(.+?)\*\*: (.+)$', readme, re.M)}
    facts = [(len(mods), 'teaching days'), (sum(len(m['types']) for m in mods), 'slides'),
             (sum(m['labs'] for m in mods), 'laboratory blocks'), (sum(m['exercises'] for m in mods), 'exercises'),
             (len(set().union(*(m['figures'] for m in mods))), 'figures and diagrams')]
    docs = ''.join(f'<a class="card" href="{href(SITE / page, ROOT)}"><strong>{esc(name)}</strong>'
                   f'<span>{esc(summary)}</span></a>' for _, page, name, summary in DOCS)
    courses = []
    for i, (course, key, course_title, pdf) in enumerate(COURSES, 1):
        rows = []
        for m in (m for m in mods if m['course'] == key):
            meta = ' · '.join([count(len(m['types']), 'slide'), count(m['parts'], 'part'),
                               count(m['labs'], 'lab'), count(m['exercises'], 'exercise')])
            rows.append(f'<li class="module" id="{key}-{m["number"] or m["dir"].name}">'
                        f'<div class="num"><span>{esc(m["kind"])}</span><b>{esc(m["number"])}</b></div>'
                        f'<div class="info"><h3>{esc(m["title"])}</h3><p>{esc(m["subtitle"])}</p>'
                        f'<p class="meta">{meta}</p></div>'
                        f'<div class="links">{module_links(m, ROOT)}</div>{exercise_list(m, ROOT)}</li>')
        blurb = blurbs.get(course_title.lower(), '').rstrip('.')
        courses.append(
            f'<section class="course {key}" aria-labelledby="c-{key}">'
            f'<div class="course-head"><div><p class="kicker">Course {i} · {count(len(rows), "day")}</p>'
            f'<h2 id="c-{key}">{esc(course_title)}</h2>'
            + (f'<p>{esc(blurb[:1].upper() + blurb[1:])}.</p>' if blurb else '') + '</div>'
            f'<a class="pill" href="{href(ROOT / "combined_pdfs" / pdf, ROOT)}" '
            f'title="All five days in one bookmarked PDF">Complete course <b>PDF</b></a></div>'
            f'<ol class="modules">{"".join(rows)}</ol></section>')
    keys = ''.join('<span>' + ''.join(f'<kbd>{esc(k)}</kbd>' for k in ks) + f' {esc(v)}</span>' for ks, v in [
        ('←→', 'move'), ('N', 'notes'), ('P', 'presenter window'), ('G', 'overview'),
        ('F', 'fullscreen'), ('?', 'help')])
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<meta name="color-scheme" content="light dark">\n<title>{esc(title)}</title>\n'
            f'<link rel="stylesheet" href="site/style.css">\n{notebook_loader(ROOT)}</head>\n<body class="home-page">\n'
            f'<header class="hero wrap"><p class="kicker">Course materials</p><h1>{esc(title)}</h1>'
            f'<p class="lead">{lead}</p><dl class="facts">'
            + ''.join(f'<div><dt>{esc(label)}</dt><dd>{n}</dd></div>' for n, label in facts)
            + f'</dl></header>\n<main class="wrap">\n'
            f'<section class="start" aria-labelledby="start"><h2 class="eyebrow" id="start">Start here</h2>'
            f'<div class="cards">{docs}</div>'
            f'<p class="keys"><b>In the HTML decks</b> {keys}</p></section>\n'
            f'<section class="bundle" aria-labelledby="bundle"><div><h2 id="bundle">Download everything</h2>'
            f'<p>The complete package in one archive: decks, instructor notes, exercise notebooks and data.</p></div>'
            f'<a class="download" href="{href(ARCHIVE, ROOT)}" download><span aria-hidden="true">↓</span>'
            f'{ARCHIVE.name}' + (f'<small>{size(ARCHIVE)}</small>' if ARCHIVE.exists() else '') + '</a></section>\n'
            + '\n'.join(courses)
            + f'\n</main>\n<footer class="site-foot wrap"><div><p>Everything here is a local file, so the site, '
            f'the decks and the notes work offline. Generated from the Markdown sources by '
            f'<code>curriculum/build_site.py</code>.</p><p>Type: IBM Plex, SIL Open Font License.</p>'
            f'</div></footer>\n</body>\n</html>\n')


# ---------------------------------------------------------------- style
def font_faces(text: str) -> str:
    """IBM Plex subset to the characters the pages use, embedded so it loads from file:// too."""
    try:
        from fontTools import subset
        from fontTools.ttLib import TTFont
    except ImportError:
        print('fontTools is not installed; pages fall back to system fonts.', file=sys.stderr)
        return ''
    unicodes = sorted({ord(c) for c in text} | set(range(0x20, 0x7F)))
    options = subset.Options()
    options.flavor = 'woff'
    options.drop_tables += ['meta']
    faces = []
    for family, key, weight, style in FACES:
        font = TTFont(theme.font_path(key), recalcTimestamp=False)
        subsetter = subset.Subsetter(options)
        subsetter.populate(unicodes=unicodes)
        subsetter.subset(font)
        buf = io.BytesIO()
        subset.save_font(font, buf, options)
        data = base64.b64encode(buf.getvalue()).decode()
        faces.append(f'@font-face{{font-family:"{family}";font-style:{style};font-weight:{weight};'
                     f'font-display:swap;src:url(data:font/woff;base64,{data}) format("woff")}}')
    return '\n'.join(faces) + '\n'


CSS = """
:root{
  --bg:#F7F9FB;--surface:#FFFFFF;--ink:#0F2A44;--text:#283D53;--muted:#566A7F;--faint:#8597A9;
  --rule:#E0E7EE;--rule-strong:#C8D3DE;--panel:#F0F4F8;
  --blue:#005EB8;--teal:#087F8C;--orange:#B34D17;--purple:#6B4E9B;
  --blue-t:#E7F0FA;--teal-t:#E3F2F2;--orange-t:#FBEEE6;--purple-t:#EFEAF6;
  --accent:var(--blue);--accent-t:var(--blue-t);--on-accent:#FFFFFF;
  --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0B1E31;--surface:#0F263D;--ink:#EBF2F8;--text:#C8D6E3;--muted:#91A7BB;--faint:#6A8399;
  --rule:#1C3650;--rule-strong:#2A4865;--panel:#11293F;
  --blue:#8EC5EC;--teal:#6ACCD3;--orange:#F0A577;--purple:#C3B0E8;
  --blue-t:#14354F;--teal-t:#10363B;--orange-t:#3A2719;--purple-t:#29233F;
  --on-accent:#0B1E31;
}}
.ml{--accent:var(--blue);--accent-t:var(--blue-t)}
.nn{--accent:var(--teal);--accent-t:var(--teal-t)}
*,*::before,*::after{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 var(--sans);
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:2px}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
code,kbd{font-family:var(--mono)}
.wrap{max-width:1040px;margin:0 auto;padding:0 24px}
.kicker,.eyebrow,.toc-title{font:600 12px/1.4 var(--sans);letter-spacing:.09em;text-transform:uppercase;margin:0}
.kicker{color:var(--teal)}
.eyebrow,.toc-title{color:var(--muted)}

/* index */
.hero{padding-top:76px;padding-bottom:8px}
.hero h1{font:600 clamp(31px,5vw,48px)/1.1 var(--sans);letter-spacing:-.018em;color:var(--ink);
  margin:14px 0 0;max-width:19ch;text-wrap:balance}
.lead{font-size:18px;line-height:1.55;color:var(--muted);max-width:58ch;margin:20px 0 0}
.facts{display:flex;flex-wrap:wrap;gap:16px 44px;margin:40px 0 0;padding:22px 0 0;border-top:1px solid var(--rule)}
.facts div{display:flex;flex-direction:column-reverse}
.facts dt{font-size:13px;color:var(--muted)}
.facts dd{margin:0;font:600 26px/1.15 var(--mono);color:var(--ink);font-variant-numeric:tabular-nums}
.start{padding-top:56px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px;margin-top:14px}
.card{display:block;padding:18px 20px 20px;background:var(--surface);border:1px solid var(--rule);
  border-radius:10px;color:var(--text);text-decoration:none;transition:border-color .15s,box-shadow .15s}
.card:hover{border-color:var(--accent);box-shadow:0 4px 18px -10px rgba(15,42,68,.35)}
.card strong{display:flex;justify-content:space-between;gap:12px;color:var(--ink);font-size:16.5px;font-weight:600}
.card strong::after{content:"→";color:var(--faint);transition:transform .15s,color .15s}
.card:hover strong::after{color:var(--accent);transform:translateX(3px)}
.card span{display:block;margin-top:6px;font-size:14px;line-height:1.5;color:var(--muted)}
.keys{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;margin:22px 0 0;font-size:13px;color:var(--muted)}
.keys b{font-weight:600;color:var(--text)}
.keys span{white-space:nowrap}
kbd{display:inline-block;min-width:22px;padding:1px 6px;margin-right:2px;border:1px solid var(--rule-strong);
  border-bottom-width:2px;border-radius:5px;background:var(--surface);color:var(--ink);font-size:12px;
  line-height:1.5;text-align:center}
.bundle{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:16px 32px;margin-top:40px;
  padding:22px 24px;border:1px solid var(--rule);border-radius:12px;background:var(--surface)}
.bundle h2{margin:0;font:600 17px/1.3 var(--sans);color:var(--ink)}
.bundle p{margin:4px 0 0;max-width:60ch;font-size:14px;line-height:1.5;color:var(--muted)}
.download{flex:none;display:inline-flex;align-items:center;gap:10px;padding:11px 20px;border-radius:8px;
  background:var(--ink);color:var(--bg);font-size:15px;font-weight:600;text-decoration:none;transition:background .15s}
.download span{font-size:17px;line-height:1}
.download small{font:500 12.5px/1 var(--mono);opacity:.7}
.download:hover{background:var(--blue);color:var(--on-accent)}
.course{padding-top:64px}
.course-head{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:16px 32px;
  padding-bottom:18px;border-bottom:2px solid var(--accent)}
.course-head .kicker{color:var(--accent)}
.course-head h2{font:600 clamp(23px,3vw,28px)/1.2 var(--sans);letter-spacing:-.01em;color:var(--ink);margin:6px 0 0}
.course-head p:not(.kicker){margin:6px 0 0;max-width:64ch;font-size:15px;color:var(--muted)}
.pill{flex:none;display:inline-flex;align-items:center;gap:8px;padding:8px 14px;border:1px solid var(--rule-strong);
  border-radius:999px;background:var(--surface);color:var(--ink);font-size:14px;text-decoration:none;white-space:nowrap}
.pill b{font:600 11px/1 var(--sans);letter-spacing:.06em;padding:3px 6px;border-radius:4px;background:var(--accent-t);color:var(--accent)}
.pill:hover{border-color:var(--accent);color:var(--accent)}
.modules{list-style:none;margin:0;padding:0}
.module{display:grid;grid-template-columns:76px minmax(0,1fr) auto;gap:6px 28px;align-items:center;
  padding:24px 0;border-bottom:1px solid var(--rule);scroll-margin-top:16px}
.num{align-self:start;display:flex;flex-direction:column;gap:7px;padding-top:2px}
.num span{font:600 11px/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.num b{font:600 36px/1 var(--mono);color:var(--accent)}
.info h3{margin:0;font:600 19px/1.3 var(--sans);color:var(--ink)}
.info p{margin:4px 0 0;font-size:15px;line-height:1.5;color:var(--muted)}
.info .meta{margin-top:9px;font:12.5px/1.4 var(--mono);color:var(--faint)}
.links{display:flex;flex-direction:column;align-items:flex-start;gap:8px}
.group{display:flex;align-items:center;gap:12px}
.group>span{width:46px;font:600 11px/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.seg{display:inline-flex}
.seg a{min-width:60px;padding:7px 12px;border:1px solid var(--rule-strong);background:var(--surface);color:var(--ink);
  font:500 13px/1.2 var(--sans);text-align:center;text-decoration:none;transition:border-color .12s,color .12s,background .12s}
.seg a+a{margin-left:-1px}
.seg a:first-child{border-radius:7px 0 0 7px}
.seg a:last-child{border-radius:0 7px 7px 0}
.seg a:hover{position:relative;border-color:var(--accent);color:var(--accent)}
.seg a.primary{background:var(--accent);border-color:var(--accent);color:var(--on-accent);font-weight:600}
.seg a.primary:hover{background:var(--ink);border-color:var(--ink);color:var(--bg)}
.exercises{grid-column:2/-1;margin-top:16px;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}
.exercises summary{display:flex;flex-wrap:wrap;align-items:center;gap:4px 12px;padding:10px 16px;cursor:pointer;
  list-style:none;font-size:14px;color:var(--ink);user-select:none}
.exercises summary::-webkit-details-marker{display:none}
.exercises summary::before{content:"";width:6px;height:6px;margin:0 2px 2px 0;border-right:1.5px solid var(--muted);
  border-bottom:1.5px solid var(--muted);transform:rotate(-45deg);transition:transform .15s}
.exercises[open] summary::before{transform:rotate(45deg)}
.exercises summary b{font-weight:600}
.exercises summary span{font-size:13px;color:var(--muted)}
.exercises summary:hover b{color:var(--accent)}
.exercises ol{list-style:none;margin:0;padding:0 16px 4px}
.exercises li{display:flex;align-items:center;justify-content:space-between;gap:10px 24px;padding:13px 0;border-top:1px solid var(--rule)}
.ex-info{display:flex;flex-direction:column;gap:3px;min-width:0}
.ex-info strong{color:var(--ink);font-size:14.5px;font-weight:600;line-height:1.35}
.ex-path{font:12px/1.4 var(--mono);color:var(--faint)}
.ex-data{font-size:12.5px;line-height:1.5;color:var(--muted)}
.ex-data a{color:var(--muted);overflow-wrap:anywhere}
.ex-data a:hover{color:var(--accent)}
.ex-data span{color:var(--faint);white-space:nowrap}
.exercises .seg{flex:none}
.exercises .seg a{min-width:80px}
.page-head .exercises{margin-top:22px}
.site-foot{margin-top:80px;padding-bottom:56px;font-size:13px;color:var(--muted)}
.site-foot>div{padding-top:24px;border-top:1px solid var(--rule)}
.site-foot p{margin:0 0 4px}
.site-foot code{font-size:12.5px}

/* guides and notes */
.topbar{position:sticky;top:0;z-index:10;background:var(--bg);
  background:color-mix(in srgb,var(--bg) 90%,transparent);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);
  border-bottom:1px solid var(--rule)}
.bar{display:flex;align-items:center;justify-content:space-between;gap:16px;max-width:1280px;height:54px;margin:0 auto;padding:0 24px}
.home{display:flex;align-items:center;gap:10px;color:var(--ink);font-weight:600;font-size:15px;text-decoration:none;white-space:nowrap}
.home::before{content:"";width:9px;height:9px;border-radius:2px;background:var(--accent);transform:rotate(45deg)}
.topnav{display:flex;gap:22px;font-size:14px;overflow-x:auto}
.topnav a{color:var(--muted);text-decoration:none;white-space:nowrap}
.topnav a:hover{color:var(--ink)}
.topnav a[aria-current]{color:var(--ink);font-weight:600}
.layout{display:grid;grid-template-columns:270px minmax(0,1fr);gap:56px;max-width:1280px;margin:0 auto;padding:44px 24px 0}
.layout>main{min-width:0;max-width:780px}
.toc{font-size:13.5px;line-height:1.4}
.toc.side{position:sticky;top:78px;max-height:calc(100vh - 98px);overflow-y:auto;padding:0 6px 24px 0;
  overscroll-behavior:contain;scrollbar-width:thin}
.toc-title{margin-bottom:10px}
.toc ol{list-style:none;margin:0;padding:0}
.toc ol ol{margin:2px 0 6px}
.toc a{display:flex;gap:8px;padding:4px 8px;border-radius:6px;color:var(--text);text-decoration:none}
.toc a:hover{background:var(--panel);color:var(--ink)}
.toc a[aria-current]{background:var(--accent-t);color:var(--accent)}
.toc ol ol a{padding-left:20px}
.toc .n{flex:none;min-width:1.9em;font:12px/1.5 var(--mono);color:var(--faint)}
.toc a[aria-current] .n{color:inherit}
.toc .toc-group{margin-top:12px}
.toc .toc-group:first-child{margin-top:0}
.toc .part{display:flex;flex-direction:column;gap:1px;margin:0;padding:4px 8px;color:var(--ink);font-weight:600}
.toc .part .k{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--accent)}
.toc .toc-group ol a{padding-left:8px}
.toc .lab .t::after,.toc .debrief .t::after{content:"Lab";margin-left:6px;padding:1px 5px;border-radius:3px;
  font:600 10px/1 var(--sans);letter-spacing:.05em;text-transform:uppercase;vertical-align:1px;background:var(--orange-t);color:var(--orange)}
.toc .debrief .t::after{content:"Debrief";background:var(--purple-t);color:var(--purple)}
.toc-mobile{display:none}
.page-head{margin-bottom:40px;padding-bottom:28px;border-bottom:1px solid var(--rule)}
.page-head .kicker{color:var(--accent)}
.page-head h1{font:600 clamp(28px,4.2vw,40px)/1.14 var(--sans);letter-spacing:-.016em;color:var(--ink);margin:12px 0 0;text-wrap:balance}
.page-head .sub{margin:10px 0 0;font-size:17px;color:var(--muted)}
.page-head .links{flex-direction:row;flex-wrap:wrap;gap:10px 28px;margin-top:24px}
.page-head .group>span{width:auto}
.caution{margin:20px 0 0;padding:10px 14px;border-left:3px solid var(--orange);background:var(--orange-t);
  border-radius:0 6px 6px 0;font-size:14px;color:var(--text)}
.prose{font-size:16.5px;line-height:1.68;overflow-wrap:break-word}
.prose>:first-child{margin-top:0}
.prose h1{font:600 clamp(28px,4.2vw,40px)/1.14 var(--sans);letter-spacing:-.016em;color:var(--ink);margin:0 0 20px}
.prose h2{font:600 23px/1.3 var(--sans);letter-spacing:-.006em;color:var(--ink);margin:52px 0 14px;scroll-margin-top:76px}
.prose h3{font:600 18px/1.35 var(--sans);color:var(--ink);margin:36px 0 10px;scroll-margin-top:76px}
.prose p,.prose ul,.prose ol{margin:0 0 16px}
.prose ul,.prose ol{padding-left:1.4em}
.prose li{margin:5px 0}
.prose li::marker{color:var(--faint)}
.prose strong{color:var(--ink);font-weight:600}
.prose code{padding:.12em .36em;border-radius:4px;background:var(--panel);color:var(--ink);font-size:.86em}
.prose pre{margin:0 0 20px;padding:14px 18px;border:1px solid var(--rule);border-radius:8px;background:var(--panel);
  overflow-x:auto;font-size:13.5px;line-height:1.6}
.prose pre code{padding:0;background:none;font-size:inherit}
.table{margin:0 0 22px;overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}
.prose table{width:100%;border-collapse:collapse;font-size:14.5px;line-height:1.5}
.prose th{background:var(--panel);color:var(--ink);font-weight:600;text-align:left;white-space:nowrap}
.prose th,.prose td{padding:9px 14px;border-bottom:1px solid var(--rule);vertical-align:top}
.prose tbody tr:last-child td{border-bottom:0}
.url{font:12.5px/1.5 var(--mono);color:var(--muted);overflow-wrap:anywhere}
.ext .url{margin-left:.5em}
.ext .url::before{content:"("}
.ext .url::after{content:")"}
.ref{font-size:13.5px;color:var(--muted)}
.ref>span:first-child{margin-right:.2em;font-weight:600;color:var(--text)}
.slide-h{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 12px;margin:44px 0 12px;padding-top:22px;
  border-top:1px solid var(--rule);font-size:19.5px}
.slide-h .n{flex:none;min-width:2em;font:600 14px/1 var(--mono);color:var(--faint)}
.slide-h .t{flex:1 1 16em;min-width:0}
.slide-h.is-section{margin-top:72px;padding-top:26px;border-top:2px solid var(--accent);font-size:25px}
.slide-h.is-section .n{color:var(--accent)}
.tag{align-self:center;padding:4px 7px;border-radius:4px;background:var(--panel);color:var(--muted);
  font:600 11px/1 var(--sans);letter-spacing:.06em;text-transform:uppercase}
.tag.section{background:var(--accent-t);color:var(--accent)}
.tag.lab{background:var(--orange-t);color:var(--orange)}
.tag.exercise{background:var(--blue-t);color:var(--blue)}
.tag.debrief{background:var(--purple-t);color:var(--purple)}
.go{align-self:center;padding:3px 10px;border:1px solid var(--rule);border-radius:999px;color:var(--muted);
  font:500 12.5px/1.4 var(--sans);text-decoration:none;white-space:nowrap}
.go:hover{border-color:var(--accent);color:var(--accent)}
.pager{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:64px}
.pager a{display:flex;flex-direction:column;gap:4px;padding:14px 18px;border:1px solid var(--rule);border-radius:10px;
  background:var(--surface);color:var(--ink);font-weight:600;font-size:15px;line-height:1.35;text-decoration:none}
.pager a:hover{border-color:var(--accent)}
.pager a span{font:500 12.5px/1.4 var(--sans);color:var(--muted)}
.pager .next{grid-column:2;text-align:right}
.page-foot{margin:56px 0 0;padding:20px 0 56px;border-top:1px solid var(--rule);font-size:13px;color:var(--muted)}
.page-foot a{color:inherit}

@media (max-width:980px){
  .layout{grid-template-columns:minmax(0,1fr);padding-top:32px}
  .layout>main{max-width:none}
  .toc.side{display:none}
  .toc-mobile{display:block;margin:0 0 32px;border:1px solid var(--rule);border-radius:10px;background:var(--surface)}
  .toc-mobile summary{padding:12px 16px;cursor:pointer;font-weight:600;color:var(--ink)}
  .toc-mobile>ol{max-height:60vh;overflow-y:auto;padding:0 8px 12px}
}
@media (max-width:760px){
  .module{grid-template-columns:52px minmax(0,1fr);gap:4px 16px}
  .num b{font-size:28px}
  .links{grid-column:2;flex-direction:row;flex-wrap:wrap;gap:10px 22px;margin-top:14px}
  .group>span{width:auto}
  .topnav{gap:16px;font-size:13px}
}
@media (max-width:600px){
  .wrap,.bar,.layout{padding-left:16px;padding-right:16px}
  .hero{padding-top:48px}
  .lead{font-size:16.5px}
  .facts{gap:14px 28px}
  .facts dd{font-size:22px}
  .home{font-size:0;gap:0}
  .exercises li{flex-direction:column;align-items:flex-start}
  .home::before{width:12px;height:12px}
  .pager{grid-template-columns:1fr}
  .pager .next{grid-column:1}
  .prose{font-size:16px}
}
@media print{
  .topbar,.toc,.go,.pager,.page-head .links{display:none!important}
  body{background:#fff;color:#000}
  .layout{display:block;padding:0}
  .slide-h{break-after:avoid}
}
"""

SCRIPT = """(function () {
  var box = document.querySelector('.toc.side');
  document.querySelectorAll('.toc-mobile a').forEach(function (a) {
    a.addEventListener('click', function () { a.closest('details').open = false; });
  });
  if (!box) return;
  var links = {};
  box.querySelectorAll('a[href^="#"]').forEach(function (a) { links[decodeURIComponent(a.hash.slice(1))] = a; });
  var heads = [].filter.call(document.querySelectorAll('main [id]'), function (h) { return links[h.id]; });
  var current = null, queued = false;
  function update() {
    queued = false;
    var id = null;
    for (var i = 0; i < heads.length && heads[i].getBoundingClientRect().top < 110; i++) id = heads[i].id;
    if (id === current) return;
    if (current) links[current].removeAttribute('aria-current');
    current = id;
    if (!id) return;
    var a = links[id], r = a.getBoundingClientRect(), b = box.getBoundingClientRect();
    a.setAttribute('aria-current', 'true');
    if (r.top < b.top || r.bottom > b.bottom) box.scrollTop += r.top - b.top - b.height / 3;
  }
  addEventListener('scroll', function () {
    if (!queued) { queued = true; requestAnimationFrame(update); }
  }, {passive: true});
  update();
})();"""


NOTEBOOK_JS = """// Generated by curriculum/build_site.py: the exercise notebooks, embedded so that pages opened
// from disk can save them (browsers show a local .ipynb as text and ignore the download attribute).
(function () {
  var files = __FILES__;
  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[data-nb]');
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    var name = a.getAttribute('data-nb'), text = files[name];
    if (text === undefined) return;
    e.preventDefault();
    var url = URL.createObjectURL(new Blob([text], {type: 'application/octet-stream'}));
    var link = document.createElement('a');
    link.href = url;
    link.download = name.split('/').pop();
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 30000);
  });
})();
"""


# ---------------------------------------------------------------- checks
def check(pages: dict[Path, str]) -> list[str]:
    """Every link must be local and resolve; anchors into generated pages must exist."""
    ids = {p: set(re.findall(r'\sid="([^"]+)"', s)) for p, s in pages.items()}
    problems = []
    for page, text in pages.items():
        where = page.relative_to(ROOT)
        for url in re.findall(r'\s(?:href|src)="([^"]*)"', text):
            url = html.unescape(url)
            if re.match(r'[a-z]+:', url):
                problems.append(f'{where}: external link {url}')
                continue
            path, _, fragment = url.partition('#')
            target = (page.parent / unquote(path)).resolve() if path else page
            if target not in pages and not target.exists():
                if target == ARCHIVE:
                    continue  # reported separately: it is added after the build
                problems.append(f'{where}: missing target {url}')
            elif fragment and target in ids and fragment not in ids[target]:
                problems.append(f'{where}: missing anchor {url}')
    return problems


def main() -> int:
    mods = modules()
    sources = {(ROOT / src).resolve(): SITE / page for src, page, _, _ in DOCS}
    sources.update({(m['dir'] / 'instructor_notes.md').resolve(): m['page'] for m in mods})
    pages: dict[Path, str] = {INDEX: index_page(mods, (ROOT / 'README.md').read_text())}
    for src, page, name, _ in DOCS:
        out = SITE / page
        body, tokens = render_markdown(ROOT / src, out, sources)
        note = f'Rendered from <a href="{href(ROOT / src, out.parent)}">{esc(src)}</a>.'
        pages[out] = doc_page(f'{name} · Course materials', out, body, doc_toc(tokens), note, current=page)
    titles = {key: title for _, key, title, _ in COURSES}
    for i, m in enumerate(mods):
        prev = mods[i - 1] if i else None
        nxt = mods[i + 1] if i + 1 < len(mods) else None
        pages[m['page']] = notes_page(m, prev, nxt, titles[m['course']].split(' for ')[0])

    shown = html.unescape(re.sub(r'<[^>]+>', '', ''.join(pages.values()))) + CSS
    (SITE / 'notes').mkdir(parents=True, exist_ok=True)
    (SITE / 'style.css').write_text(font_faces(shown) + CSS.lstrip())
    embedded = {nb.relative_to(ROOT).as_posix(): nb.read_text() for m in mods for e in m['notebooks']
                for nb in (e['student'], e['solution']) if nb}
    (SITE / 'notebooks.js').write_text(NOTEBOOK_JS.replace(
        '__FILES__', json.dumps(embedded, ensure_ascii=False, sort_keys=True, separators=(',', ':'))))
    for path, text in pages.items():
        path.write_text(text)
    problems = check(pages)
    for p in problems:
        print('problem:', p, file=sys.stderr)
    if not ARCHIVE.exists():
        print(f'note: {ARCHIVE.name} is not next to index.html yet; the download link will work once it is')
    print(f'index.html and {len(pages) - 1} pages under site/ '
          f'(style.css {(SITE / "style.css").stat().st_size / 1e3:.0f} kB), {len(problems)} link problems')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
