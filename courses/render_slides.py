#!/usr/bin/env python3
"""Rebuild the course decks.

    python3 render_slides.py --all                 # every module + combined course PDFs
    python3 render_slides.py path/to/slides.md     # one module

Each build writes slides.pdf, slides.html, slides.pptx, instructor_notes.pdf and
instructor_notes.md next to slides.md. See curriculum/AUTHORING.md.
"""
import argparse
import shutil
from pathlib import Path
from curriculum.build import ROOT, build_one, main as build_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all or args.source is None:
        import sys
        sys.argv = [sys.argv[0]]
        build_all()
    else:
        source = args.source.resolve()
        build_one(source)
        if args.output:
            suffix = args.output.suffix.lower()
            if suffix not in {".pdf", ".html"}:
                parser.error("Output must be .pdf or .html")
            generated = source.with_suffix(suffix)
            if generated != args.output.resolve():
                args.output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(generated, args.output)
