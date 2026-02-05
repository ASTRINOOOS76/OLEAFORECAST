#!/usr/bin/env bash
set -euo pipefail

rm -rf .tmp_test
./node_modules/.bin/tsc -p tsconfig.json --outDir .tmp_test --noEmit false

echo "Running smoke tests:"
find .tmp_test -type f -name '*.smoke.js' | sort

FOUND="$(find .tmp_test -type f -name '*.smoke.js' | head -n 1 || true)"
if [ -z "${FOUND}" ]; then
  echo "No smoke tests found after compilation. Listing .tmp_test:"
  find .tmp_test -maxdepth 6 -type f | sed -n '1,200p'
  exit 1
fi

while IFS= read -r f; do
  echo "==> node $f"
  node "$f"
done < <(find .tmp_test -type f -name '*.smoke.js' | sort)
