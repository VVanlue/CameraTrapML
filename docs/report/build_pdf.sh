#!/usr/bin/env bash
# Build docs/report/report.pdf from report.md (needs pandoc >= 2.19 and Google Chrome).
#   bash docs/report/build_pdf.sh
set -euo pipefail
cd "$(dirname "$0")"
pandoc report.md --citeproc --standalone --embed-resources --number-sections=false \
  --metadata link-citations=true --css style.css -o report.html
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(pwd)/report.pdf" "file://$(pwd)/report.html" 2>/dev/null
rm -f report.html
echo "wrote $(pwd)/report.pdf"
