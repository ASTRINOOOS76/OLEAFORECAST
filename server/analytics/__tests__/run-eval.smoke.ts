import { loadActualsCsv } from "../load-actuals";
import { loadForecastJson } from "../load-forecast";
import { toEvalInput } from "../evaluator";
import { decideWinner } from "../winner-service";

function assert(cond: unknown, msg: string) {
  if (!cond) throw new Error(`ASSERT FAIL: ${msg}`);
}

const actuals = loadActualsCsv("data/ES/W4/actuals.csv");
const p0 = loadForecastJson("data/ES/W4/E0.json");
const p2 = loadForecastJson("data/ES/W4/E2.json");

const e0 = toEvalInput(p0, actuals, { medianPriceEurPerTon: actuals.at(-1)?.value });
const e2 = toEvalInput(p2, actuals, { medianPriceEurPerTon: actuals.at(-1)?.value });

const d = decideWinner("ES" as any, "W4" as any, [e0, e2]);
assert(d.winnerModelId === "E0", `expected E0 winner, got ${d.winnerModelId}`);

console.log("run-eval.smoke OK");
