import { readFileSync } from "node:fs";
import type { ActualPoint } from "./metrics";

/**
 * Expected CSV (header required):
 * date,value
 * 2026-02-01,4200
 * 2026-02-08,4210
 *
 * Also supports:
 * t+1,4200
 */
export function loadActualsCsv(path: string): ActualPoint[] {
  const raw = readFileSync(path, "utf-8");
  const lines = raw
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length < 2) throw new Error(`CSV has no data: ${path}`);

  const header = lines[0].toLowerCase();
  const hasHeader = header.includes("date") && header.includes("value");

  const rows = hasHeader ? lines.slice(1) : lines;

  const out: ActualPoint[] = [];
  for (let i = 0; i < rows.length; i++) {
    const parts = rows[i].split(",").map((p) => p.trim());
    if (parts.length < 2) throw new Error(`Bad CSV row at line ${i + 1}: "${rows[i]}"`);

    const date = parts[0];
    const value = Number(parts[1]);

    if (!date) throw new Error(`Missing date at line ${i + 1}`);
    if (!Number.isFinite(value)) throw new Error(`Invalid value at line ${i + 1}: "${parts[1]}"`);

    out.push({ date, value });
  }

  // stable ordering
  out.sort((a, b) => a.date.localeCompare(b.date));
  return out;
}
