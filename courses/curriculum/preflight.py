#!/usr/bin/env python3
"""Read-only checks for classroom preparation. No network access or downloads."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
build = ['numpy', 'pandas', 'scipy', 'matplotlib', 'sklearn', 'reportlab', 'fitz', 'pygments', 'PIL']
labs = ['torch', 'joblib', 'nbformat']
optional = ['torchvision', 'tensorflow', 'keras', 'umap', 'imblearn', 'h5py', 'seaborn', 'openpyxl', 'nbclient']


def report(title, names, missing_word):
    print(title)
    for name in names:
        print(f'  {name:12s} {"available" if importlib.util.find_spec(name) else missing_word}')


report('SLIDE BUILD DEPENDENCIES', build, 'MISSING')
report('\nCORE LAB DEPENDENCIES', labs, 'MISSING')
report('\nOPTIONAL EXERCISE DEPENDENCIES', optional, 'not installed')
fonts = sorted((ROOT / 'curriculum' / 'fonts').glob('*.ttf'))
print(f'\nVENDORED FONTS: {len(fonts)} IBM Plex files ({"ok" if len(fonts) >= 12 else "MISSING"})')
print('\nCOURSE MATERIALS')
for p in sorted(ROOT.glob('*_course/*/slides.md')):
    status = {name: p.with_name(name).exists() for name in ['slides.pdf', 'slides.html', 'instructor_notes.pdf']}
    missing = [k for k, v in status.items() if not v]
    print(f'  {p.parent.parent.name}/{p.parent.name}: {"ok" if not missing else "missing " + ", ".join(missing)}')
print('\nLABORATORY DATASETS')
manifest = ROOT / 'curriculum' / 'dataset_manifest.json'
if manifest.exists():
    import hashlib
    import json
    entries = json.loads(manifest.read_text())['files']
    problems = []
    for e in entries:
        f = ROOT / e['path']
        if not f.exists():
            problems.append(f'missing: {e["path"]}')
        elif hashlib.sha256(f.read_bytes()).hexdigest() != e['sha256']:
            problems.append(f'changed: {e["path"]}')
    total = sum(e['bytes'] for e in entries)
    print(f'  {len(entries)} files, {total / 1e6:,.0f} MB: '
          + ('all present and unchanged' if not problems else f'{len(problems)} PROBLEMS'))
    for msg in problems:
        print('   ', msg)
print('\nPREPARE BEFORE CLASS: image dataset downloads, pretrained weights, optional packages for the '
      'chosen labs, and a manageable time-series subset. See INSTRUCTOR_GUIDE.md.')
