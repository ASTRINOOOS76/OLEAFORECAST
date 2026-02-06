import { readFileSync } from "node:fs";
import type { ModelForecastPayload } from "./contracts";
import { validateForecastPayload } from "./validate-forecast";
import { assertForecastQuality } from "./forecast-quality";
import { assertPathConsistency } from "./path-consistency";

export function loadForecastJson(path: string): ModelForecastPayload {
  const raw = readFileSync(path, "utf-8");
  const obj = JSON.parse(raw) as unknown;

  const payload = validateForecastPayload(obj);
  assertForecastQuality(payload);
  assertPathConsistency(path, payload);

  return payload;
}
