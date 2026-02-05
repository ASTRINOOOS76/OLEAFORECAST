#!/usr/bin/env bash
set -euo pipefail

rm -rf .tmp_test
./node_modules/.bin/tsc -p tsconfig.json --outDir .tmp_test --noEmit false

SMOKE_JS="$(find .tmp_test -type f -name 'winner.smoke.js' | head -n 1)"
if [ -z "${SMOKE_JS}" ]; then
  echo "Smoke test JS not found. Listing .tmp_test:"
  find .tmp_test -maxdepth 5 -type f | sed -n '1,200p'
  exit 1
fi

node "${SMOKE_JS}"
