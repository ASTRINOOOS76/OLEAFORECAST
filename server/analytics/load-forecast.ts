import { readFileSync } from "node:fs";
import type { ModelForecastPayload } from "./contracts";
import { validateForecastPayload } from "./validate-forecast";

export function loadForecastJson(path: string): ModelForecastPayload {
  const raw = readFileSync(path, "utf-8");
  const obj = JSON.parse(raw) as unknown;
  return validateForecastPayload(obj);
}
