#!/usr/bin/env python3
"""Render a deck PDF to numbered contact sheets for visual review.

    python3 curriculum/review.py path/to/slides.pdf out_dir [--cols 3] [--per 9] [--dpi 62] [--pages 10-18]

Each sheet is a PNG grid of slides with the slide number in a yellow tag."""
import argparse, io, sys
from pathlib import Path
import fitz
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser()
ap.add_argument('pdf'); ap.add_argument('out')
ap.add_argument('--cols', type=int, default=3); ap.add_argument('--per', type=int, default=9)
ap.add_argument('--dpi', type=int, default=62); ap.add_argument('--pages', default='')
ap.add_argument('--prefix', default='sheet')
a = ap.parse_args()
doc = fitz.open(a.pdf)
pages = list(range(len(doc)))
if a.pages:
    lo, _, hi = a.pages.partition('-'); pages = list(range(int(lo) - 1, int(hi or lo)))
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
imgs = []
for p in pages:
    pix = doc[p].get_pixmap(dpi=a.dpi)
    im = Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')
    d = ImageDraw.Draw(im); d.rectangle([0, 0, 34, 16], fill=(255, 230, 0)); d.text((3, 2), str(p + 1), fill=(0, 0, 0))
    imgs.append(im)
w, h = imgs[0].size
names = []
for k in range(0, len(imgs), a.per):
    batch = imgs[k:k + a.per]
    rows = (len(batch) + a.cols - 1) // a.cols
    S = Image.new('RGB', (a.cols * w + (a.cols + 1) * 8, rows * h + (rows + 1) * 8), (120, 120, 120))
    for j, im in enumerate(batch):
        S.paste(im, (8 + (j % a.cols) * (w + 8), 8 + (j // a.cols) * (h + 8)))
    name = out / f'{a.prefix}_{k // a.per + 1:02d}.png'; S.save(name); names.append(str(name))
print('\n'.join(names))
