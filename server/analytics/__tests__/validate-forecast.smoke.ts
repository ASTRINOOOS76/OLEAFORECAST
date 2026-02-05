import { validateForecastPayload } from "../validate-forecast";

function assertThrows(fn: () => void, msg: string) {
  let threw = false;
  try {
    fn();
  } catch {
    threw = true;
  }
  if (!threw) throw new Error(`ASSERT FAIL (expected throw): ${msg}`);
}

function assert(cond: unknown, msg: string) {
  if (!cond) throw new Error(`ASSERT FAIL: ${msg}`);
}

const okPayload = {
  modelId: "E0",
  market: "ES",
  horizon: "W4",
  generatedAt: new Date().toISOString(),
  series: [
    { date: "t+1", p10: 4100, p50: 4200, p90: 4300 },
    { date: "t+2", p10: 4090, p50: 4200, p90: 4310 }
  ]
};

const validated = validateForecastPayload(okPayload);
assert(validated.series.length === 2, "expected 2 points");

assertThrows(
  () =>
    validateForecastPayload({
      modelId: "E0",
      market: "ES",
      horizon: "W4",
      generatedAt: new Date().toISOString(),
      series: []
    }),
  "must reject empty series"
);

assertThrows(
  () =>
    validateForecastPayload({
      modelId: "E0",
      market: "ES",
      horizon: "W4",
      generatedAt: new Date().toISOString(),
      series: [{ date: "t+1", p10: 4200, p50: 4100, p90: 4300 }]
    }),
  "must reject p10>p50"
);

console.log("validate-forecast.smoke OK");
