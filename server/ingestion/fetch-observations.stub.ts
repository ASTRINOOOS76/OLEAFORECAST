import { existsSync, readFileSync } from "node:fs";
import type { SourceObs, Unit } from "../analytics/canonical-price";

type SeedSpec = {
  sourceId: string;
  market: string;
  path: string;
};

const SEEDS: SeedSpec[] = [
  { sourceId: "poolred", market: "ES", path: "seeds/poolred_ES.csv" },
  { sourceId: "ismea", market: "IT", path: "seeds/ismea_IT.csv" },
  { sourceId: "oleista", market: "GR", path: "seeds/oleista_GR.csv" },
  { sourceId: "sedik", market: "TN", path: "seeds/sedik_TN.csv" },
  { sourceId: "oleista", market: "PT", path: "seeds/oleista_PT.csv" },
  { sourceId: "oleista", market: "TR", path: "seeds/oleista_TR.csv" }
];

function parseSeedCsv(text: string, sourceId: string, market: string): SourceObs[] {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length < 2) return [];

  const header = lines[0].toLowerCase();
  if (header !== "date,value,unit") {
    throw new Error(
      `Bad seed header for ${sourceId}/${market}. Expected "date,value,unit" got "${lines[0]}"`
    );
  }

  const out: SourceObs[] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(",").map((x) => x.trim());
    if (parts.length !== 3) continue;

    const [date, valueRaw, unitRaw] = parts;
    const value = Number(valueRaw);
    const unit = unitRaw as Unit;

    if (!date || !Number.isFinite(value)) continue;
    if (unit !== "EUR_KG" && unit !== "EUR_TON") {
      throw new Error(`Bad unit "${unitRaw}" in seed ${sourceId}/${market} at line ${i + 1}`);
    }

    out.push({
      sourceId,
      market,
      date,
      value,
      unit,
      grade: "EVOO"
    });
  }

  return out;
}

/**
 * Seed-driven fetcher:
 * - Reads CSV files from /seeds
 * - Normalizes to SourceObs[]
 * - You later replace each seed with a real connector, keeping the same output contract.
 */
export async function fetchObservationsForMarkets(
  markets: string[],
  startDate: string,
  endDate: string
): Promise<SourceObs[]> {
  // startDate/endDate currently not used for slicing; keep it simple for v1
  void startDate;
  void endDate;

  const want = new Set(markets);
  const obs: SourceObs[] = [];

  for (const s of SEEDS) {
    if (!want.has(s.market)) continue;
    if (!existsSync(s.path)) continue;

    const txt = readFileSync(s.path, "utf-8");
    obs.push(...parseSeedCsv(txt, s.sourceId, s.market));
  }

  return obs;
}
