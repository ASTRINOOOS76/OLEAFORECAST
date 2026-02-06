import type { ActualPoint } from "./metrics";

export type DataQualityReport = {
  count: number;
  duplicates: string[];
  nonFinite: number;
  isSorted: boolean;
};

export function checkActualsQuality(actuals: ActualPoint[]): DataQualityReport {
  const seen = new Set<string>();
  const duplicates: string[] = [];
  let nonFinite = 0;

  for (const p of actuals) {
    if (!Number.isFinite(p.value)) nonFinite++;
    if (seen.has(p.date)) duplicates.push(p.date);
    seen.add(p.date);
  }

  let isSorted = true;
  for (let i = 1; i < actuals.length; i++) {
    if (actuals[i - 1].date > actuals[i].date) {
      isSorted = false;
      break;
    }
  }

  return { count: actuals.length, duplicates, nonFinite, isSorted };
}

export function assertActualsQuality(actuals: ActualPoint[]) {
  const r = checkActualsQuality(actuals);
  if (r.count === 0) throw new Error("Actuals empty");
  if (r.nonFinite > 0) throw new Error(`Actuals contain non-finite values: ${r.nonFinite}`);
  if (r.duplicates.length > 0) throw new Error(`Duplicate dates in actuals: ${r.duplicates.join(", ")}`);
  return r;
}
