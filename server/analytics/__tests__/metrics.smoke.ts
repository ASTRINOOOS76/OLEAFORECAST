import { computeMetrics } from "../metrics";
import type { IntervalForecast } from "../contracts";

function assert(cond: unknown, msg: string) {
  if (!cond) throw new Error(`ASSERT FAIL: ${msg}`);
}

const actuals = [
  { date: "t+1", value: 4200 },
  { date: "t+2", value: 4210 },
  { date: "t+3", value: 4190 },
  { date: "t+4", value: 4220 }
];

const forecast: IntervalForecast[] = [
  { date: "t+1", p10: 4100, p50: 4200, p90: 4300 },
  { date: "t+2", p10: 4100, p50: 4210, p90: 4320 },
  { date: "t+3", p10: 4100, p50: 4190, p90: 4320 },
  { date: "t+4", p10: 4100, p50: 4220, p90: 4320 }
];

const m = computeMetrics(forecast, actuals);

assert(m.availability === 1, "availability should be 1");
assert(m.biasEurPerTon === 0, "bias should be 0");
assert(m.coverage === 1, "coverage should be 1");
assert(Number.isFinite(m.mase), "mase must be finite");

console.log("metrics.smoke OK");
