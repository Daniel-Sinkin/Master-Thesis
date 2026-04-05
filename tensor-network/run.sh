#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

cmake -S . -B build
ln -sfn build/compile_commands.json compile_commands.json
cmake --build build -j

if [[ -t 1 && -n "${TERM:-}" ]]; then
  clear
fi

exec ./build/tensor-network
