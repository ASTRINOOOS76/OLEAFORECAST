import { validateForecastPayload } from "../validate-forecast";
import { toEvalInput } from "../evaluator";
import { selectWinner } from "../winner";

function assert(cond: unknown, msg: string) {
  if (!cond) throw new Error(`ASSERT FAIL: ${msg}`);
}

const actuals = [
  { date: "t+1", value: 4200 },
  { date: "t+2", value: 4210 },
  { date: "t+3", value: 4190 },
  { date: "t+4", value: 4220 }
];

// Two models: E0 good, E2 biased
const p0 = validateForecastPayload({
  modelId: "E0",
  market: "ES",
  horizon: "W4",
  generatedAt: new Date().toISOString(),
  series: [
    { date: "t+1", p10: 4100, p50: 4200, p90: 4300 },
    { date: "t+2", p10: 4100, p50: 4210, p90: 4320 },
    { date: "t+3", p10: 4100, p50: 4190, p90: 4320 },
    { date: "t+4", p10: 4100, p50: 4220, p90: 4320 }
  ]
});

const p2 = validateForecastPayload({
  modelId: "E2",
  market: "ES",
  horizon: "W4",
  generatedAt: new Date().toISOString(),
  series: [
    { date: "t+1", p10: 4100, p50: 4400, p90: 4700 },
    { date: "t+2", p10: 4100, p50: 4410, p90: 4720 },
    { date: "t+3", p10: 4100, p50: 4390, p90: 4720 },
    { date: "t+4", p10: 4100, p50: 4420, p90: 4720 }
  ]
});

const e0 = toEvalInput(p0, actuals, { medianPriceEurPerTon: 4200 });
const e2 = toEvalInput(p2, actuals, { medianPriceEurPerTon: 4200 });

const d = selectWinner("ES", "W4", [e0, e2]);

assert(d.winnerModelId === "E0", `expected E0 winner, got ${d.winnerModelId}`);
console.log("endtoend.smoke OK");
