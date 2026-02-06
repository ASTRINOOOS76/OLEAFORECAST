import { readdirSync, readFileSync, mkdirSync, writeFileSync, existsSync } from "node:fs";
import { dirname } from "node:path";

type Row = {
  sourceId: string;
  market: string;
  grade: string;
  date: string;
  value: number;
  unit: "EUR_KG" | "EUR_T";
  currency: "EUR";
  fetchedAt: string;
};

function ensureDir(path: string) {
  mkdirSync(path, { recursive: true });
}

function eurTtoEurKg(v: number) { return v / 1000.0; }

function parseNdjsonFile(path: string): Row[] {
  const txt = readFileSync(path, "utf8");
  const out: Row[] = [];
  for (const ln of txt.split("\n")) {
    if (!ln.trim()) continue;
    const o = JSON.parse(ln);
    if (!Number.isFinite(o.value)) continue;
    out.push(o);
  }
  return out;
}

function listAllNdjsonUnder(dir: string): string[] {
  if (!existsSync(dir)) return [];
  const out: string[] = [];
  for (const sourceId of readdirSync(dir)) {
    const p = `${dir}/${sourceId}`;
    for (const f of readdirSync(p)) {
      if (f.endsWith(".ndjson")) out.push(`${p}/${f}`);
    }
  }
  return out.sort();
}

function ymdToDate(s: string): Date {
  const [y, m, d] = s.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d));
}

function dateToYmd(d: Date): string {
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function addDays(d: Date, n: number): Date {
  return new Date(d.getTime() + n * 86400000);
}

function main() {
  const rawDir = "data/observations/raw";
  const files = listAllNdjsonUnder(rawDir);
  if (!files.length) {
    console.log("[canonical] no raw files");
    return;
  }

  let rows: Row[] = [];
  for (const f of files) rows = rows.concat(parseNdjsonFile(f));

  // normalize to EUR/kg
  const norm = rows.map(r => ({
    ...r,
    value_eur_kg: r.unit === "EUR_T" ? eurTtoEurKg(r.value) : r.value
  }));

  // group by market+grade, keep latest per date (highest fetchedAt)
  type Key = string;
  const byKeyDate = new Map<Key, { fetchedAt: string; v: number; sources: Set<string> }>();
  for (const r of norm) {
    const k = `${r.market}|${r.grade}|${r.date}`;
    const cur = byKeyDate.get(k);
    if (!cur || r.fetchedAt > cur.fetchedAt) {
      byKeyDate.set(k, { fetchedAt: r.fetchedAt, v: r.value_eur_kg, sources: new Set([r.sourceId]) });
    } else if (cur && r.fetchedAt === cur.fetchedAt) {
      cur.sources.add(r.sourceId);
    }
  }

  // materialize per (market,grade); default to EVOO only
  const groups = new Map<string, Array<{ date: string; v: number; fetchedAt: string; sources: string[] }>>();
  for (const [k, v] of byKeyDate.entries()) {
    const [market, grade, date] = k.split("|");
    if (grade !== "EVOO") continue;
    const gk = `${market}|${grade}`;
    if (!groups.has(gk)) groups.set(gk, []);
    groups.get(gk)!.push({ date, v: v.v, fetchedAt: v.fetchedAt, sources: Array.from(v.sources) });
  }

  ensureDir("data/canonical");

  for (const [gk, pts] of groups.entries()) {
    pts.sort((a,b) => a.date.localeCompare(b.date));
    const [market, grade] = gk.split("|");

    const first = ymdToDate(pts[0].date);
    const last = ymdToDate(pts[pts.length - 1].date);
    const today = ymdToDate(dateToYmd(new Date()));
    const end = last >= today ? last : today;

    // map date -> value
    const map = new Map<string, { v: number; sources: string[] }>();
    for (const p of pts) map.set(p.date, { v: p.v, sources: p.sources });

    // carry-forward daily canonical
    const lines: string[] = ["date,value_eur_kg,staleDays,sources"];
    let curV: number | null = null;
    let stale = 0;

    for (let d = first; d <= end; d = addDays(d, 1)) {
      const ymd = dateToYmd(d);
      const hit = map.get(ymd);
      if (hit) {
        curV = hit.v;
        stale = 0;
        lines.push(`${ymd},${curV.toFixed(4)},${stale},"${hit.sources.join("+")}"`);
      } else if (curV !== null) {
        stale += 1;
        lines.push(`${ymd},${curV.toFixed(4)},${stale},"carry_forward"`);
      }
    }

    const outPath = `data/canonical/${market}_${grade}.csv`;
    writeFileSync(outPath, lines.join("\n") + "\n", "utf8");
    console.log(`[canonical] wrote ${outPath} (${lines.length-1} rows)`);
  }
}

main();
