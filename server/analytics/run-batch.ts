import { readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadActualsCsv } from "./load-actuals";
import { loadForecastJson } from "./load-forecast";
import { toEvalInput } from "./evaluator";
import { decideWinner } from "./winner-service";

type BatchResult = {
  market: string;
  horizon: string;
  winnerModelId: string;
  ranked: Array<{ modelId: string; score: number }>;
};

function listDirs(path: string): string[] {
  return readdirSync(path, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name);
}

function listJsonFiles(path: string): string[] {
  return readdirSync(path, { withFileTypes: true })
    .filter(d => d.isFile() && d.name.toLowerCase().endsWith(".json"))
    .map(d => d.name);
}

function usage() {
  console.log(`Usage:
  tsx server/analytics/run-batch.ts <dataRoot>

Expected structure:
  data/<MARKET>/<HORIZON>/actuals.csv
  data/<MARKET>/<HORIZON>/*.json (forecast payloads)

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
    const forecastFiles = listJsonFiles(hzPath).filter(f => f !== "actuals.json");

    if (forecastFiles.length === 0) continue;

    const evals = forecastFiles.map((f) => {
      const payload = loadForecastJson(join(hzPath, f));
      return toEvalInput(payload, actuals, { medianPriceEurPerTon: actuals.at(-1)?.value });
    });

    const decision = decideWinner(market as any, horizon as any, evals);

    results.push({
      market,
      horizon,
      winnerModelId: decision.winnerModelId,
      ranked: decision.ranked.map(r => ({ modelId: r.modelId, score: r.score }))
    });
  }
}

console.log(JSON.stringify({ generatedAt: new Date().toISOString(), results }, null, 2));
