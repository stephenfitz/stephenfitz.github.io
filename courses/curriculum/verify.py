#!/usr/bin/env python3
"""Verify the built course package and the worked examples used in the slides.

Checks every module deck (slides.md, slides.pdf, slides.html, instructor notes),
figure files and provenance, code syntax, page geometry and text bounds, then
re-computes selected worked examples. Writes build/verification.json.
"""
from __future__ import annotations

import ast
import importlib.metadata
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import correlate2d  # noqa: E402
from scipy.special import softmax  # noqa: E402

from curriculum.markup import parse, plain  # noqa: E402
from curriculum.theme import ROOT  # noqa: E402

FIG = ROOT / 'curriculum' / 'figures'
MANIFEST = json.loads((FIG / 'manifest.json').read_text()) if (FIG / 'manifest.json').exists() else {}
VISUAL_LAYOUTS = {'split', 'split-left', 'figure', 'pair', 'cards', 'compare', 'steps', 'steps-vertical',
                  'stats', 'table', 'code', 'code-split', 'agenda'}

errors, summary = [], []
for source in sorted(ROOT.glob('*_course/*/slides.md')):
    rel = source.relative_to(ROOT)
    slides = parse(source)
    pdf = fitz.open(source.with_suffix('.pdf'))
    if len(slides) != len(pdf):
        errors.append(f'{rel}: PDF has {len(pdf)} pages for {len(slides)} slides')
    missing_notes = [i + 1 for i, s in enumerate(slides) if not s['notes'].strip()]
    if missing_notes:
        errors.append(f'{rel}: slides without notes {missing_notes}')
    html = source.with_suffix('.html').read_text()
    if html.count('class="slide"') != len(slides):
        errors.append(f'{rel}: HTML slide count differs')
    if re.search(r'(?:src|href)=["\']https?://', html):
        errors.append(f'{rel}: remote presentation dependency')
    notes_pdf = source.with_name('instructor_notes.pdf')
    if not notes_pdf.exists() or len(fitz.open(notes_pdf)) < len(slides):
        errors.append(f'{rel}: instructor_notes.pdf missing or short')
    titles = Counter(plain(s['title']) for s in slides)
    dups = [t for t, n in titles.items() if n > 1]
    if dups:
        errors.append(f'{rel}: duplicate slide titles {dups}')
    figures = set()
    for i, (page, s) in enumerate(zip(pdf, slides), 1):
        if abs(page.rect.width - 960) > .1 or abs(page.rect.height - 540) > .1:
            errors.append(f'{rel}:{i}: wrong page geometry')
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    if not page.rect.contains(fitz.Rect(span['bbox'])):
                        errors.append(f'{rel}:{i}: out-of-page text {span["text"]!r}')
        for name in s['figures']:
            figures.add(name)
            for suffix in ('pdf', 'svg', 'png'):
                if not (FIG / f'{name}.{suffix}').exists():
                    errors.append(f'{rel}:{i}: missing figure file {name}.{suffix}')
            if MANIFEST and name not in MANIFEST:
                errors.append(f'{rel}:{i}: figure {name} has no provenance entry in manifest.json')
        for code in s['codes']:
            if code['lang'] in ('python', 'py'):
                try:
                    ast.parse(code['text'])
                except SyntaxError as e:
                    errors.append(f'{rel}:{i}: code syntax error: {e}')
    report_path = ROOT / 'build' / 'reports' / f'{source.parent.name}.json'
    layouts, warnings = Counter(), []
    if report_path.exists():
        rep = json.loads(report_path.read_text())
        layouts = Counter(e['layout'] for e in rep['slides'])
        warnings = rep.get('warnings', [])
        errors.extend(f'{rel}: build warning: {w}' for w in warnings)
    content = [s for s in slides if s['type'] not in ('cover', 'section')]
    with_figure = sum(bool(s['figures']) for s in content)
    visual = sum(1 for e in (json.loads(report_path.read_text())['slides'] if report_path.exists() else [])
                 if e['layout'] in VISUAL_LAYOUTS)
    summary.append({'module': source.parent.name, 'label': slides[0].get('label', ''),
                    'title': plain(slides[0]['title']), 'slides': len(slides),
                    'sections': sum(s['type'] == 'section' for s in slides),
                    'labs': sum(s['type'] == 'lab' for s in slides),
                    'exercises': sum(s['type'] == 'exercise' for s in slides),
                    'slides_with_figures': with_figure, 'unique_figures': len(figures),
                    'structured_or_visual_share': round(visual / max(len(content), 1), 2),
                    'layouts': dict(layouts)})

# Worked examples that recur in the decks, recomputed independently of the renderer.
assert abs(np.mean(np.abs([1, -1, 4])) - 2) < 1e-12
assert abs(np.sqrt(np.mean(np.square([1, -1, 4]))) - np.sqrt(6)) < 1e-12
assert np.isclose((85 + 7) / 100, .92) and np.isclose(7 / (7 + 3), .7) and np.isclose(7 / 12, .5833, atol=1e-4)
assert np.isclose(.2 + (1 - .2) / 10, .28)
assert np.isclose(np.mean(np.square(np.array([2, 4, 6]) - [3.8, 4, 4.2])), 2.16)
assert np.isclose(80 / (80 + 495), .1391304347826087)
np.testing.assert_allclose(correlate2d([[1, 2, 0], [0, 1, 3], [2, 1, 0]], [[1, 0], [0, -1]], mode='valid'), [[0, -1], [-1, 1]])
np.testing.assert_allclose(softmax([0, np.log(2)]), [1 / 3, 2 / 3])
assert np.isclose(softmax([0, np.log(2)]) @ np.array([10, 20]), 50 / 3)
assert np.isclose(.9 * 2 + .2 * (-1), 1.6)
assert 6 * 8 + 8 + 8 * 1 + 1 == 65
assert 16 * (3 * 3 * 3 + 1) == 448

try:
    import torch
    from torch import nn
    w = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    h = torch.tensor([2.5, 1.], dtype=torch.float64)
    z = w * h[0] - .2 * h[1] + .1
    loss = nn.functional.binary_cross_entropy_with_logits(z, torch.tensor(1., dtype=torch.float64))
    loss.backward()
    assert np.isclose(z.item(), .9)
    assert np.isclose(w.grad.item(), (torch.sigmoid(z).item() - 1) * 2.5)
    conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
    assert tuple(conv(torch.zeros(2, 3, 32, 32)).shape) == (2, 16, 16, 16)
    assert sum(p.numel() for p in conv.parameters()) == 448
    rnn = nn.LSTM(6, 32, batch_first=True)
    seq, _ = rnn(torch.zeros(8, 24, 6))
    assert tuple(nn.Linear(32, 1)(seq[:, -1, :]).shape) == (8, 1)
    tensor_checks = 'passed'
except ImportError:
    tensor_checks = 'skipped (PyTorch not installed)'

versions = {}
for name in ['numpy', 'pandas', 'scipy', 'matplotlib', 'scikit-learn', 'reportlab', 'PyMuPDF', 'torch']:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = 'not installed'
report = {'modules': summary, 'total_slides': sum(s['slides'] for s in summary),
          'figure_files': len(list(FIG.glob('*.pdf'))), 'worked_examples': 'passed',
          'tensor_checks': tensor_checks, 'errors': errors, 'versions': versions}
target = ROOT / 'build' / 'verification.json'
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(report, indent=2))
for s in summary:
    print(f"{s['module']:34s} {s['slides']:4d} slides  {s['unique_figures']:3d} figures  "
          f"{s['labs']} labs  {s['exercises']:2d} exercises  visual/structured {s['structured_or_visual_share']:.0%}")
print(f"total slides: {report['total_slides']}; errors: {len(errors)}")
for e in errors[:50]:
    print('ERROR', e)
if errors:
    sys.exit(1)
