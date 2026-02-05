import type { IntervalForecast } from "./contracts";

export type MetricsResult = {
  availability: number;   // 0..1
  biasEurPerTon: number;  // signed (forecast - actual)
  mase: number;           // >=0
  coverage: number;       // 0..1 for p10/p90
};

export type ActualPoint = { date: string; value: number };

function mean(xs: number[]): number {
  if (xs.length === 0) return NaN;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function abs(x: number): number {
  return Math.abs(x);
}

export function computeAvailability(forecast: IntervalForecast[], actuals: ActualPoint[]): number {
  if (forecast.length === 0) return 0;
  const actualMap = new Map(actuals.map(a => [a.date, a.value]));
  const have = forecast.filter(f => actualMap.has(f.date)).length;
  return have / forecast.length;
}

export function computeBias(forecast: IntervalForecast[], actuals: ActualPoint[]): number {
  const actualMap = new Map(actuals.map(a => [a.date, a.value]));
  const diffs: number[] = [];
  for (const f of forecast) {
    const a = actualMap.get(f.date);
    if (a === undefined) continue;
    diffs.push(f.p50 - a);
  }
  return diffs.length ? mean(diffs) : NaN;
}

/**
 * Coverage for central interval [p10, p90] across points with actuals.
 * coverage = share of actuals that lie within [p10,p90].
 */
export function computeCoverage(forecast: IntervalForecast[], actuals: ActualPoint[]): number {
  const actualMap = new Map(actuals.map(a => [a.date, a.value]));
  let n = 0;
  let hit = 0;
  for (const f of forecast) {
    const a = actualMap.get(f.date);
    if (a === undefined) continue;
    n += 1;
    if (a >= f.p10 && a <= f.p90) hit += 1;
  }
  return n ? hit / n : NaN;
}

/**
 * MASE:
 *  MAE(model) / MAE(naive one-step)
 * naive one-step forecast: y[t-1] for y[t]
 *
 * For forecast evaluation, we compute MAE on overlapping dates.
 * For naive scale denominator, we use the provided actuals series order by date.
 */
export function computeMASE(forecast: IntervalForecast[], actuals: ActualPoint[]): number {
  const actualMap = new Map(actuals.map(a => [a.date, a.value]));

  // MAE model
  const absErr: number[] = [];
  for (const f of forecast) {
    const a = actualMap.get(f.date);
    if (a === undefined) continue;
    absErr.push(abs(f.p50 - a));
  }
  const maeModel = absErr.length ? mean(absErr) : NaN;

  // MAE naive one-step computed on actuals in order
  const ordered = [...actuals].sort((x, y) => x.date.localeCompare(y.date));
  const naiveErr: number[] = [];
  for (let i = 1; i < ordered.length; i++) {
    naiveErr.push(abs(ordered[i].value - ordered[i - 1].value));
  }
  const maeNaive = naiveErr.length ? mean(naiveErr) : NaN;

  if (!Number.isFinite(maeModel) || !Number.isFinite(maeNaive) || maeNaive === 0) return NaN;
  return maeModel / maeNaive;
}

export function computeMetrics(
  forecast: IntervalForecast[],
  actuals: ActualPoint[]
): MetricsResult {
  const availability = computeAvailability(forecast, actuals);
  const biasEurPerTon = computeBias(forecast, actuals);
  const coverage = computeCoverage(forecast, actuals);
  const mase = computeMASE(forecast, actuals);

  return { availability, biasEurPerTon, mase, coverage };
}
