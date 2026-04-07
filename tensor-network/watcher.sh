#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

watchexec --restart \
  --postpone \
  --poll 100ms \
  --stop-signal SIGTERM \
  --stop-timeout 2s \
  --watch src \
  --watch tests \
  --watch CMakeLists.txt \
  --watch run.sh \
  --ignore 'build/**' \
  --ignore '**/*.swp' \
  --ignore '**/*.swo' \
  --ignore '**/*~' \
  --ignore '**/.DS_Store' \
  -- ./run.sh
