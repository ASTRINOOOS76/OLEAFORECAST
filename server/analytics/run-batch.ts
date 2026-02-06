import { readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadActualsCsv } from "./load-actuals";
import { loadForecastJson } from "./load-forecast";
import { toEvalInput } from "./evaluator";
import { decideWinner } from "./winner-service";
import type { WinnerDecision } from "./types";

type BatchResult = {
  market: string;
  horizon: string;
  winnerModelId: string;
  ranked: WinnerDecision["ranked"]; // keep exact type from WinnerDecision
};

function listDirs(path: string): string[] {
  return readdirSync(path, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name);
}

function listJsonFiles(path: string): string[] {
  return readdirSync(path, { withFileTypes: true })
    .filter((d) => d.isFile() && d.name.toLowerCase().endsWith(".json"))
    .map((d) => d.name);
}

function usage() {
  console.log(`Usage:
  tsx server/analytics/run-batch.ts <dataRoot>

Expected structure:
  <dataRoot>/<MARKET>/<HORIZON>/actuals.csv
  <dataRoot>/<MARKET>/<HORIZON>/*.json  (forecast payloads)

Example:
  tsx server/analytics/run-batch.ts data
`);
}

const [dataRoot] = process.argv.slice(2);
if (!dataRoot) {
  usage();
  process.exit(2);
}

const markets = listDirs(dataRoot);
const results: BatchResult[] = [];

for (const market of markets) {
  const marketPath = join(dataRoot, market);
  const horizons = listDirs(marketPath);

  for (const horizon of horizons) {
    const hzPath = join(marketPath, horizon);
    const actualsPath = join(hzPath, "actuals.csv");

    if (!existsSync(actualsPath)) continue;

    const actuals = loadActualsCsv(actualsPath);

    // all json files in the horizon folder are treated as model payloads
    const forecastFiles = listJsonFiles(hzPath);

    // ignore accidental files named like actuals.json etc.
    const modelFiles = forecastFiles.filter((f) => f.toLowerCase() !== "actuals.json");
    if (modelFiles.length === 0) continue;

    const evals = modelFiles.map((f) => {
      const payload = loadForecastJson(join(hzPath, f));
      return toEvalInput(payload, actuals, { medianPriceEurPerTon: actuals.at(-1)?.value });
    });

    const decision = decideWinner(market as any, horizon as any, evals);

    results.push({
      market,
      horizon,
      winnerModelId: decision.winnerModelId,
      ranked: decision.ranked
    });
  }
}

console.log(JSON.stringify({ generatedAt: new Date().toISOString(), results }, null, 2));
