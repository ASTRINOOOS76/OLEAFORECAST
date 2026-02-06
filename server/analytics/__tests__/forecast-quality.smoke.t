import { validateForecastPayload } from "../validate-forecast";
import { assertForecastQuality } from "../forecast-quality";

function assertThrows(fn: () => void) {
  let ok = false;
  try {
    fn();
  } catch {
    ok = true;
  }
  if (!ok) throw new Error("Expected throw");
}

const p = validateForecastPayload({
  modelId: "E0",
  market: "ES",
  horizon: "W4",
  generatedAt: new Date().toISOString(),
  series: [
    { date: "t+1", p10: 1, p50: 2, p90: 3 },
    { date: "t+1", p10: 1, p50: 2, p90: 3 }
  ]
});

assertThrows(() => assertForecastQuality(p));
console.log("forecast-quality.smoke OK");
