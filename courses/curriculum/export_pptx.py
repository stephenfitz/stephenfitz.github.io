#!/usr/bin/env python3
"""Export a built deck to PowerPoint for presenting with speaker notes.

Each slide becomes a full-bleed high-resolution image of the PDF page, with:
  * the slide title in the (hidden) title placeholder, so PowerPoint's outline,
    navigation pane and screen readers see real titles;
  * the slide text as alternative text on the image;
  * the instructor notes (and reference) in the notes pane, for presenter view;
  * PowerPoint sections for the cover, each part and the closing slides.

The PPTX is a presentation copy. Edit slides.md and rebuild to change content.

    python3 curriculum/export_pptx.py path/to/slides.md [more.md ...]
"""
from __future__ import annotations

import copy
import io
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # noqa: E402
from lxml import etree  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.util import Emu  # noqa: E402

from curriculum.markup import parse, plain  # noqa: E402

SLIDE_W, SLIDE_H = Emu(12192000), Emu(6858000)   # 13.333 x 7.5 in (16:9)
RENDER_WIDTH_PX = 2400                             # 2400 x 1350 per slide
FIXED_DATE = datetime(2026, 9, 10)                 # deterministic package metadata
P14 = 'http://schemas.microsoft.com/office/powerpoint/2010/main'
SECTION_EXT_URI = '{521415D9-36F7-43E2-AB2F-B90AF26B5E84}'


def slide_png(page) -> bytes:
    zoom = RENDER_WIDTH_PX / page.rect.width
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).tobytes('png')


def alt_text(page, limit: int = 1500) -> str:
    text = re.sub(r'\s+', ' ', page.get_text('text', sort=True)).strip()
    return text if len(text) <= limit else text[:limit - 1] + '…'


def add_sections(prs, slide_ids, groups):
    """Add PowerPoint sections. groups = [(name, first_index), ...] in order."""
    pres = prs.part._element
    ext_lst = pres.find('{http://schemas.openxmlformats.org/presentationml/2006/main}extLst')
    if ext_lst is None:
        ext_lst = etree.SubElement(pres, '{http://schemas.openxmlformats.org/presentationml/2006/main}extLst')
    ext = etree.SubElement(ext_lst, '{http://schemas.openxmlformats.org/presentationml/2006/main}ext',
                           uri=SECTION_EXT_URI)
    sec_lst = etree.SubElement(ext, f'{{{P14}}}sectionLst', nsmap={'p14': P14})
    bounds = [g[1] for g in groups] + [len(slide_ids)]
    for k, (name, start) in enumerate(groups):
        sec = etree.SubElement(sec_lst, f'{{{P14}}}section', name=name,
                               id='{%08X-0000-4000-8000-%012X}' % (k + 1, k + 1))
        lst = etree.SubElement(sec, f'{{{P14}}}sldIdLst')
        for sid in slide_ids[start:bounds[k + 1]]:
            etree.SubElement(lst, f'{{{P14}}}sldId', id=str(sid))


def normalise_zip(data: bytes) -> bytes:
    """Rewrite the package with fixed timestamps so rebuilds are byte-identical."""
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            fixed = zipfile.ZipInfo(info.filename, date_time=FIXED_DATE.timetuple()[:6])
            fixed.compress_type = zipfile.ZIP_STORED if info.filename.endswith('.png') else zipfile.ZIP_DEFLATED
            fixed.external_attr = 0o644 << 16
            dst.writestr(fixed, src.read(info.filename))
    return out.getvalue()


def export(md_path: Path) -> Path:
    md_path = Path(md_path).resolve()
    slides = parse(md_path)
    pdf = fitz.open(md_path.with_suffix('.pdf'))
    if len(pdf) != len(slides):
        raise SystemExit(f'{md_path}: PDF has {len(pdf)} pages but source has {len(slides)} slides; rebuild first')
    label = slides[0].get('label', md_path.parent.name)
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    layout = prs.slide_layouts[5]  # "Title Only"
    slide_ids, groups = [], [('Opening', 0)]
    for i, (s, page) in enumerate(zip(slides, pdf)):
        if s['type'] == 'section':
            groups.append((f'Part {sum(g[0].startswith("Part ") for g in groups) + 1}: {plain(s["title"])}', i))
        slide = prs.slides.add_slide(layout)
        title = slide.shapes.title
        title.text = plain(s['title'])
        title.left, title.top, title.width, title.height = Emu(0), Emu(0), SLIDE_W, Emu(600000)
        pic = slide.shapes.add_picture(io.BytesIO(slide_png(page)), 0, 0, SLIDE_W, SLIDE_H)
        c_nv_pr = pic._element.nvPicPr.cNvPr
        c_nv_pr.set('name', f'Slide {i + 1}')
        c_nv_pr.set('descr', alt_text(page))
        # keep the picture above the title placeholder (title stays for outline and accessibility)
        tree = slide.shapes._spTree
        tree.remove(pic._element)
        tree.append(pic._element)
        note = s['notes'].strip()
        if s.get('source'):
            note += '\n\nReference: ' + s['source']
        slide.notes_slide.notes_text_frame.text = note
        slide_ids.append(slide.slide_id)
    groups = [g for k, g in enumerate(groups) if k == 0 or g[1] != groups[k - 1][1]]
    add_sections(prs, slide_ids, groups)
    cp = prs.core_properties
    cp.title = f'{label}: {plain(slides[0]["title"])}'
    cp.author = 'Course teaching materials'
    cp.subject = 'Presentation copy with instructor notes; edit slides.md and rebuild to change content'
    cp.created = cp.modified = FIXED_DATE
    cp.last_modified_by = 'Course build'
    cp.revision = 1
    buf = io.BytesIO()
    prs.save(buf)
    target = md_path.with_suffix('.pptx')
    target.write_bytes(normalise_zip(buf.getvalue()))
    pdf.close()
    return target


def main():
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path(__file__).resolve().parents[1].glob('*_course/*/slides.md'))
    for p in paths:
        t = export(p)
        print(f'{t.parent.name}: {t.name} {t.stat().st_size / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
