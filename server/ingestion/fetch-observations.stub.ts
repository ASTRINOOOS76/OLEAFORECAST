import type { SourceObs } from "../analytics/canonical-price";

/**
 * TODO: wire real connectors here.
 * This returns normalized observations in EUR/kg or EUR/ton, with ISO dates.
 */
export async function fetchObservationsForMarkets(
  markets: string[],
  startDate: string,
  endDate: string
): Promise<SourceObs[]> {
  // Example shape. Replace with real connector outputs.
  const obs: SourceObs[] = [];

  // TODO: PoolRed / Oleista / ISMEA / SEDIK
  // obs.push({ sourceId:"poolred", market:"ES", date:"2026-02-01", value:4.20, unit:"EUR_KG", grade:"EVOO" });

  return obs;
}
