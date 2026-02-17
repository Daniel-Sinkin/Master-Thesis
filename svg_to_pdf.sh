#!/usr/bin/env bash
set -euo pipefail

IMAGES_DIR="${1:-images}"

if ! command -v inkscape >/dev/null 2>&1; then
  echo "Error: inkscape not found in PATH."
  echo "Install Inkscape and ensure 'inkscape' is on PATH."
  exit 1
fi

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Error: directory not found: $IMAGES_DIR"
  exit 1
fi

count_total=0
count_converted=0
count_skipped=0

find "$IMAGES_DIR" -type f -name "*.svg" -print0 | \
while IFS= read -r -d '' svg; do
  count_total=$((count_total + 1))

  pdf="${svg%.svg}.pdf"

  # Skip if PDF exists and is newer than SVG
  if [[ -f "$pdf" && "$pdf" -nt "$svg" ]]; then
    count_skipped=$((count_skipped + 1))
    continue
  fi

  echo "Converting: $svg -> $pdf"
  inkscape "$svg" --export-type=pdf --export-filename="$pdf" >/dev/null 2>&1
  count_converted=$((count_converted + 1))
done

echo
echo "Done."
echo "  SVGs found:     $count_total"
echo "  Converted:      $count_converted"
echo "  Up-to-date:     $count_skipped"
