import { readdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { loadActualsCsv } from "./load-actuals";
import { loadForecastJson } from "./load-forecast";
import { toEvalInput } from "./evaluator";
import { decideWinner } from "./winner-service";
import { writeJson, writeText } from "./write-report";

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
  ranked: any; // decision schema may evolve; keep output tolerant
}> = [];

for (const market of listDirs(dataRoot)) {
  const marketPath = join(dataRoot, market);

  for (const horizon of listDirs(marketPath)) {
    const hzPath = join(marketPath, horizon);
    const actualsPath = join(hzPath, "actuals.csv");

    if (!existsSync(actualsPath)) continue;

    const actuals = loadActualsCsv(actualsPath);

    const modelFiles = listJsonFiles(hzPath).filter(
      (f) => f.toLowerCase() !== "actuals.json"
    );
    if (modelFiles.length === 0) continue;

    const evals = modelFiles.map((f) => {
      const payload = loadForecastJson(join(hzPath, f));
      return toEvalInput(payload, actuals, {
        medianPriceEurPerTon: actuals.at(-1)?.value
      });
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

const report = { generatedAt: new Date().toISOString(), results };

// Write reports/
writeJson("reports/winners.json", report);

const md: string[] = [];
md.push("# Batch Winners");
md.push("");
md.push(`Generated at: ${report.generatedAt}`);
md.push("");
md.push("| Market | Horizon | Winner |");
md.push("|---|---|---|");
for (const r of results) {
  md.push(`| ${r.market} | ${r.horizon} | ${r.winnerModelId} |`);
}
writeText("reports/README.md", md.join("\n"));

// Also print to stdout (useful in CI logs)
console.log(JSON.stringify(report, null, 2));
