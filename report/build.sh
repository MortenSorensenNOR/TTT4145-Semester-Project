#!/usr/bin/env bash
# Rebuild the README diagrams from the standalone .tex sources.
#
# Requires: pdflatex (TeX Live), pdftoppm (poppler), fontawesome.sty.
# Outputs PNGs into report/, intermediate .aux/.log/.pdf into report/build/.

set -euo pipefail
cd "$(dirname "$0")"

mkdir -p build

for fig in pipeline frame_format; do
    pdflatex -interaction=nonstopmode -halt-on-error \
        -output-directory=build "$fig.tex" >/dev/null
    pdftoppm -png -r 300 "build/$fig.pdf" "$fig"
    # pdftoppm appends -1 for single-page PDFs; rename to plain .png.
    mv "$fig-1.png" "$fig.png"
    echo "  built $fig.png"
done
