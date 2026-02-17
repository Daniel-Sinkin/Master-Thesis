#!/bin/bash
set -e

SRC="build/main.pdf"
DST="docs/main.pdf"

if [ ! -f "$SRC" ]; then
    echo "Error: $SRC not found. Run make first."
    exit 1
fi

if [ -f "$DST" ]; then
    OLD_HASH=$(md5 -q "$DST")
    NEW_HASH=$(md5 -q "$SRC")

    OLD_AGE=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M" "$DST")
    echo "Current docs/main.pdf: $OLD_AGE ($OLD_HASH)"

    if [ "$OLD_HASH" = "$NEW_HASH" ]; then
        echo "Hashes match. Nothing to publish."
        exit 0
    fi
    echo "Hash changed ($OLD_HASH -> $NEW_HASH)"
else
    echo "No existing docs/main.pdf"
fi

cp "$SRC" "$DST"
echo "Copied $SRC -> $DST"

if [ -z "$(git diff --cached --name-only)" ]; then
    git add docs/main.pdf
    git commit -m "publish thesis pdf"
    git push
    echo "Pushed."
else
    echo "Staged changes detected, skipping git operations."
    echo "PDF copied but not committed."
fi
