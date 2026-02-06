import { loadActualsCsv } from "../load-actuals";
import { loadForecastJson } from "../load-forecast";
import { toEvalInput } from "../evaluator";

const actualsPath = "data/_fixtures/ES/W4/actuals.csv";
const forecastPath = "data/_fixtures/ES/W4/E0.json";

const actuals = loadActualsCsv(actualsPath);
const payload = loadForecastJson(forecastPath);

const evalInput = toEvalInput(payload, actuals, {
  medianPriceEurPerTon: actuals.at(-1)?.value
});

if (!evalInput || evalInput.modelId !== "E0") {
  throw new Error("run-eval.smoke failed: bad evalInput");
}

console.log("run-eval.smoke OK");
