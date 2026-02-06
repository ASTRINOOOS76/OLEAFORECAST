import type { IntervalForecast } from "./contracts";

export type ActualPoint = { date: string; value: number };

export type MetricsResult = {
  availability: number;   // 0..1
  biasEurPerTon: number;  // signed (p50 - actual)
  mase: number;           // >=0
  coverage: number;       // 0..1 for p10/p90
};

function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
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

export function computeMASE(forecast: IntervalForecast[], actuals: ActualPoint[]): number {
  const actualMap = new Map(actuals.map(a => [a.date, a.value]));

  const absErr: number[] = [];
  for (const f of forecast) {
    const a = actualMap.get(f.date);
    if (a === undefined) continue;
    absErr.push(Math.abs(f.p50 - a));
  }
  const maeModel = absErr.length ? mean(absErr) : NaN;

  const ordered = [...actuals].sort((x, y) => x.date.localeCompare(y.date));
  const naiveErr: number[] = [];
  for (let i = 1; i < ordered.length; i++) {
    naiveErr.push(Math.abs(ordered[i].value - ordered[i - 1].value));
  }
  const maeNaive = naiveErr.length ? mean(naiveErr) : NaN;

  if (!Number.isFinite(maeModel) || !Number.isFinite(maeNaive) || maeNaive === 0) return NaN;
  return maeModel / maeNaive;
}

export function computeMetrics(forecast: IntervalForecast[], actuals: ActualPoint[]): MetricsResult {
  return {
    availability: computeAvailability(forecast, actuals),
    biasEurPerTon: computeBias(forecast, actuals),
    mase: computeMASE(forecast, actuals),
    coverage: computeCoverage(forecast, actuals)
  };
}
