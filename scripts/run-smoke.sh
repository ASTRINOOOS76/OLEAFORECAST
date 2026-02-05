#!/usr/bin/env bash
set -euo pipefail

rm -rf .tmp_test
./node_modules/.bin/tsc -p tsconfig.json --outDir .tmp_test --noEmit false

# Run all compiled smoke tests
echo "Running smoke tests:"
find .tmp_test -type f -name '*.smoke.js' | sort

while IFS= read -r f; do
  echo "==> node $f"
  node "$f"
done < <(find .tmp_test -type f -name '*.smoke.js' | sort)

node "${SMOKE_JS}"
