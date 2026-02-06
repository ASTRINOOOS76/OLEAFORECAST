import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { CanonicalPoint } from "../analytics/canonical-price";

export function writeActualsCsv(path: string, points: CanonicalPoint[]) {
  mkdirSync(dirname(path), { recursive: true });

  const rows: string[] = [];
  rows.push("date,value");

  for (const p of points) {
    // skip NaN days (no data)
    if (!Number.isFinite(p.p50)) continue;
    rows.push(`${p.date},${p.p50}`);
  }

  writeFileSync(path, rows.join("\n") + "\n", "utf-8");
}
