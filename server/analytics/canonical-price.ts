export type Unit = "EUR_TON" | "EUR_KG";

export type SourceObs = {
  sourceId: string; // e.g. "poolred", "oleista", "ismea", "sedik"
  market: string; // ES/IT/GR/TN/PT/TR
  date: string; // ISO date YYYY-MM-DD (observation date)
  value: number; // price in unit below
  unit: Unit; // EUR_TON or EUR_KG
  grade?: "EVOO"; // for now fixed
};

export type CanonicalConfig = {
  // stable reliability weights (per source)
  reliability: Record<string, number>; // e.g. { poolred: 1.0, oleista: 0.7 }
  // freshness half-life in days (per market)
  halfLifeDays: Record<string, number>; // e.g. { ES: 14, IT: 21, ... }
  // base uncertainty (EUR/ton) and growth factor
  sigma0: Record<string, number>; // base sigma per market
  sigmaGrow: Record<string, number>; // sigma growth per sqrt(day)
};

export type CanonicalPoint = {
  date: string; // daily date
  p10: number;
  p50: number;
  p90: number;
  ageDays: number;
  confidence: number; // 0..1
  sourcesUsed: Array<{ sourceId: string; w: number; ageDays: number; valueEurTon: number }>;
};

function toEurTon(v: number, unit: Unit): number {
  if (!Number.isFinite(v)) throw new Error(`Non-finite value: ${v}`);
  return unit === "EUR_KG" ? v * 1000 : v;
}

function daysBetween(a: Date, b: Date): number {
  // b - a in days
  const ms = b.getTime() - a.getTime();
  return Math.floor(ms / (24 * 3600 * 1000));
}

function expDecay(ageDays: number, halfLifeDays: number): number {
  // weight halves every halfLifeDays
  if (halfLifeDays <= 0) return 0;
  return Math.exp((-Math.log(2) * ageDays) / halfLifeDays);
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

// simple weighted median (robust)
function weightedMedian(values: number[], weights: number[]): number {
  const pairs = values
    .map((v, i) => ({ v, w: weights[i] }))
    .filter((p) => Number.isFinite(p.v) && Number.isFinite(p.w) && p.w > 0)
    .sort((a, b) => a.v - b.v);

  const totalW = pairs.reduce((s, p) => s + p.w, 0);
  if (totalW <= 0) throw new Error("No positive weights for weightedMedian");

  let cum = 0;
  for (const p of pairs) {
    cum += p.w;
    if (cum >= totalW / 2) return p.v;
  }
  return pairs[pairs.length - 1].v;
}

export function buildCanonicalDaily(
  market: string,
  startDate: string, // YYYY-MM-DD
  endDate: string, // YYYY-MM-DD
  observations: SourceObs[],
  cfg: CanonicalConfig
): CanonicalPoint[] {
  const halfLife = cfg.halfLifeDays[market] ?? 21;
  const sigma0 = cfg.sigma0[market] ?? 120; // EUR/ton baseline
  const sigmaGrow = cfg.sigmaGrow[market] ?? 35; // EUR/ton per sqrt(day)

  const start = new Date(startDate + "T00:00:00Z");
  const end = new Date(endDate + "T00:00:00Z");
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
    throw new Error(`Bad date range: ${startDate}..${endDate}`);
  }

  // filter only this market and EVOO
  const obs = observations
    .filter((o) => o.market === market && (o.grade ?? "EVOO") === "EVOO")
    .map((o) => ({
      ...o,
      valueEurTon: toEurTon(o.value, o.unit),
      dt: new Date(o.date + "T00:00:00Z")
    }))
    .filter((o) => !Number.isNaN(o.dt.getTime()));

  // group obs by sourceId
  const bySource = new Map<string, typeof obs>();
  for (const o of obs) {
    if (!bySource.has(o.sourceId)) bySource.set(o.sourceId, []);
    bySource.get(o.sourceId)!.push(o);
  }
  for (const [sid, arr] of bySource.entries()) {
    arr.sort((a, b) => a.dt.getTime() - b.dt.getTime());
    bySource.set(sid, arr);
  }

  const out: CanonicalPoint[] = [];
  for (let d = new Date(start); d.getTime() <= end.getTime(); d.setUTCDate(d.getUTCDate() + 1)) {
    const date = d.toISOString().slice(0, 10);

    // take last observation <= d per source
    const vals: number[] = [];
    const ws: number[] = [];
    const used: CanonicalPoint["sourcesUsed"] = [];

    let bestAge = Infinity;

    for (const [sid, arr] of bySource.entries()) {
      // binary-ish scan from end (arrays are usually small)
      let last: any = null;
      for (let i = arr.length - 1; i >= 0; i--) {
        if (arr[i].dt.getTime() <= d.getTime()) {
          last = arr[i];
          break;
        }
      }
      if (!last) continue;

      const ageDays = Math.max(0, daysBetween(last.dt, d));
      bestAge = Math.min(bestAge, ageDays);

      const rel = cfg.reliability[sid] ?? 0.5;
      const fresh = expDecay(ageDays, halfLife);
      const w = rel * fresh;
      if (w <= 0) continue;

      vals.push(last.valueEurTon);
      ws.push(w);
      used.push({ sourceId: sid, w, ageDays, valueEurTon: last.valueEurTon });
    }

    if (vals.length === 0) {
      // no data: emit null-ish wide band but still structured
      out.push({
        date,
        p10: NaN,
        p50: NaN,
        p90: NaN,
        ageDays: 9999,
        confidence: 0,
        sourcesUsed: []
      });
      continue;
    }

    const p50 = weightedMedian(vals, ws);

    // uncertainty expands with age of freshest source
    const sigma = sigma0 + sigmaGrow * Math.sqrt(bestAge);
    const z = 1.28155; // ~p10/p90 for normal
    const p10 = p50 - z * sigma;
    const p90 = p50 + z * sigma;

    // confidence: 1 at age 0, decays with same halfLife
    const confidence = clamp01(expDecay(bestAge, halfLife));

    out.push({
      date,
      p10,
      p50,
      p90,
      ageDays: bestAge,
      confidence,
      sourcesUsed: used.sort((a, b) => b.w - a.w)
    });
  }

  return out;
}
