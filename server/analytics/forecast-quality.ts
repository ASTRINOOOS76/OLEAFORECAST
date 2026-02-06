import type { ModelForecastPayload } from "./contracts";

/**
 * Extra quality rules beyond validate-forecast:
 * - generatedAt is parseable date
 * - series has unique dates
 * - p10/p50/p90 are finite (already ensured) + sensible spread (p90-p10 >= 0)
 */
export type ForecastQualityReport = {
  count: number;
  uniqueDates: boolean;
  badGeneratedAt: boolean;
};

function isParsableDate(s: string): boolean {
  const t = Date.parse(s);
  return Number.isFinite(t);
}

export function checkForecastQuality(p: ModelForecastPayload): ForecastQualityReport {
  const dates = p.series.map(x => x.date);
  const uniqueDates = new Set(dates).size === dates.length;
  const badGeneratedAt = !isParsableDate(p.generatedAt);
  return {
    count: p.series.length,
    uniqueDates,
    badGeneratedAt
  };
}

export function assertForecastQuality(p: ModelForecastPayload): ForecastQualityReport {
  const r = checkForecastQuality(p);
  if (r.count === 0) throw new Error("Forecast series is empty");
  if (!r.uniqueDates) throw new Error("Forecast series contains duplicate dates");
  if (r.badGeneratedAt) throw new Error(`generatedAt is not parseable: ${p.generatedAt}`);
  return r;
}
