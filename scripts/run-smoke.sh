#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] typecheck"
npm run -s typecheck

echo "[smoke] clean reports"
rm -rf reports

echo "[smoke] batch over fixtures"
npx -y tsx server/analytics/run-batch.ts data/_fixtures

echo "[smoke] run-eval smoke"
npx -y tsx server/analytics/__tests__/run-eval.smoke.ts

echo "[smoke] batch smoke"
npx -y tsx server/analytics/__tests__/batch.smoke.ts

echo "[smoke] OK"
