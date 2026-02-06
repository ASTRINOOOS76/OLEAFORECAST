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

type IneligibleItem = {
  modelId: string;
  reasons: string[];
};

const results: Array<{
  market: string;
  horizon: string;
  winnerModelId: string;
  top3: string[];
  ineligible: IneligibleItem[];
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

    const decision: any = decideWinner(market as any, horizon as any, evals);

    const top3 = decision.ranked.slice(0, 3).map((r: any) => r.modelId);

    // decision.gates is assumed present from selectWinner output
    const ineligible: IneligibleItem[] = [];
    const gates = decision.gates ?? {};
    for (const [modelId, gateInfo] of Object.entries(gates)) {
      const gi: any = gateInfo;
      if (gi?.eligible === true) continue;

      const reasons: string[] = [];
      const failed = gi?.failedGates ?? gi?.failed ?? gi?.reasons ?? [];
      if (Array.isArray(failed) && failed.length) {
        for (const x of failed) reasons.push(String(x));
      }
      if (reasons.length === 0) reasons.push("ineligible");

      ineligible.push({ modelId, reasons });
    }

    results.push({
      market,
      horizon,
      winnerModelId: decision.winnerModelId,
      top3,
      ineligible,
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
md.push("| Market | Horizon | Winner | Top-3 | Ineligible | Actuals N |");
md.push("|---|---|---:|---|---:|---:|");

for (const r of results) {
  md.push(
    `| ${r.market} | ${r.horizon} | ${r.winnerModelId} | ${r.top3.join(", ")} | ${r.ineligible.length} | ${r.actualsQuality.count} |`
  );
}

md.push("");
md.push("## Ineligible details");
md.push("");

for (const r of results) {
  if (r.ineligible.length === 0) continue;

  md.push(`### ${r.market} / ${r.horizon}`);
  for (const x of r.ineligible) {
    md.push(`- **${x.modelId}**: ${x.reasons.join(", ")}`);
  }
  md.push("");
}

md.push("> Note: Forecast JSON is validated + quality-gated + path-consistent. Actuals are quality-gated.");

writeText("reports/README.md", md.join("\n"));

console.log(JSON.stringify(report, null, 2));
