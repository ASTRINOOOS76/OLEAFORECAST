import { loadActualsCsv } from "./load-actuals";
import { loadForecastJson } from "./load-forecast";
import { toEvalInput } from "./evaluator";
import { decideWinner } from "./winner-service";

function usage() {
  console.log(`Usage:
  tsx server/analytics/run-eval.ts <market> <horizon> <actuals.csv> <forecast1.json> [forecast2.json]...

Example:
  tsx server/analytics/run-eval.ts ES W4 data/ES_W4_actuals.csv data/E0_ES_W4.json data/E2_ES_W4.json
`);
}

const [market, horizon, actualsPath, ...forecastPaths] = process.argv.slice(2);

if (!market || !horizon || !actualsPath || forecastPaths.length === 0) {
  usage();
  process.exit(2);
}

const actuals = loadActualsCsv(actualsPath);
const evals = forecastPaths.map((p) => {
  const payload = loadForecastJson(p);
  return toEvalInput(payload, actuals, { medianPriceEurPerTon: actuals.at(-1)?.value });
});

const decision = decideWinner(market as any, horizon as any, evals);

console.log(JSON.stringify(decision, null, 2));
