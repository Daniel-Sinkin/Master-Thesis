#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  /usr/bin/time -l "$@"
else
  /usr/bin/time -v "$@"
fi
