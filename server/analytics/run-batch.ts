import { readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadActualsCsv } from "./load-actuals";
import { loadForecastJson } from "./load-forecast";
import { toEvalInput } from "./evaluator";
import { decideWinner } from "./winner-service";
import { writeJson, writeText } from "./write-report";
import { assertActualsQuality } from "./data-quality";

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

const results: Array<{
  market: string;
  horizon: string;
  winnerModelId: string;
  top3: string[];
  ranked: any;
  actualsQuality: {
    count: number;
    duplicates: string[];
    nonFinite: number;
    isSorted: boolean;
  };
}> = [];

for (const market of listDirs(dataRoot)) {
  const marketPath = join(dataRoot, market);

  for (const horizon of listDirs(marketPath)) {
    const hzPath = join(marketPath, horizon);
    const actualsPath = join(hzPath, "actuals.csv");

    if (!existsSync(actualsPath)) continue;

    const actuals = loadActualsCsv(actualsPath);
    const q = assertActualsQuality(actuals);

    const modelFiles = listJsonFiles(hzPath).filter((f) => f.toLowerCase() !== "actuals.json");
    if (modelFiles.length === 0) continue;

    const evals = modelFiles.map((f) => {
      const payload = loadForecastJson(join(hzPath, f));
      return toEvalInput(payload, actuals, { medianPriceEurPerTon: actuals.at(-1)?.value });
    });

    const decision = decideWinner(market as any, horizon as any, evals);

    const top3 = decision.ranked.slice(0, 3).map((r: any) => r.modelId);

    results.push({
      market,
      horizon,
      winnerModelId: decision.winnerModelId,
      top3,
      ranked: decision.ranked,
      actualsQuality: q
    });
  }
}

const report = { generatedAt: new Date().toISOString(), results };

writeJson("reports/winners.json", report);

const md: string[] = [];
md.push("# Batch Winners");
md.push("");
md.push(`Generated at: ${report.generatedAt}`);
md.push("");
md.push("| Market | Horizon | Winner | Top-3 | Actuals N |");
md.push("|---|---|---:|---|---:|");

for (const r of results) {
  md.push(`| ${r.market} | ${r.horizon} | ${r.winnerModelId} | ${r.top3.join(", ")} | ${r.actualsQuality.count} |`);
}

md.push("");
md.push("> Note: Forecast JSON is validated + quality-gated + path-consistent. Actuals are quality-gated.");

writeText("reports/README.md", md.join("\n"));

console.log(JSON.stringify(report, null, 2));
