
#!/usr/bin/env bash
set -euo pipefail

IMAGES_DIR="${1:-images}"

if ! command -v inkscape >/dev/null 2>&1; then
  echo "Error: inkscape not found in PATH."
  exit 1
fi

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Error: directory not found: $IMAGES_DIR"
  exit 1
fi

TMP_DIR="$(mktemp -d)"
LIST_FILE="$(mktemp)"
trap 'rm -rf "$TMP_DIR" "$LIST_FILE"' EXIT

find "$IMAGES_DIR" -type f -name "*.svg" -print0 > "$LIST_FILE"

count_total=0
count_converted=0
count_skipped=0

while IFS= read -r -d '' svg; do
  count_total=$((count_total + 1))
  pdf="${svg%.svg}.pdf"

  if [[ -f "$pdf" && "$pdf" -nt "$svg" ]]; then
    count_skipped=$((count_skipped + 1))
    continue
  fi

  echo "Converting: $svg -> $pdf"

  base="$(basename "${svg%.svg}")"
  plain_svg="$TMP_DIR/${base}.plain.svg"
  fixed_svg="$TMP_DIR/${base}.fixed.svg"

  # 1) Normalize to plain SVG (reduces CSS weirdness from Excalidraw)
  inkscape "$svg" --export-plain-svg --export-filename="$plain_svg"

  # 2) Force a known-good font (Helvetica) everywhere
  #    - handles both style="font-family:..." and font-family="..."
  perl -pe '
    s/font-family\s*:\s*[^;"]+/font-family: Helvetica/g;
    s/font-family\s*=\s*"[^"]+"/font-family="Helvetica"/g;
  ' "$plain_svg" > "$fixed_svg"

  # 3) Export to PDF and bake text to paths
  inkscape "$fixed_svg" \
    --export-type=pdf \
    --export-text-to-path \
    --export-filename="$pdf"

  count_converted=$((count_converted + 1))
done < "$LIST_FILE"

echo
echo "Done."
echo "  SVGs found:     $count_total"
echo "  Converted:      $count_converted"
echo "  Up-to-date:     $count_skipped"
