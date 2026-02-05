import type { ModelForecastPayload } from "./contracts";

function isObject(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

function isNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

function isString(x: unknown): x is string {
  return typeof x === "string" && x.length > 0;
}

export function validateForecastPayload(raw: unknown): ModelForecastPayload {
  if (!isObject(raw)) throw new Error("Forecast payload must be an object");

  const modelId = raw["modelId"];
  const market = raw["market"];
  const horizon = raw["horizon"];
  const generatedAt = raw["generatedAt"];
  const series = raw["series"];
  const diagnostics = raw["diagnostics"];

  if (!isString(modelId)) throw new Error("payload.modelId must be a non-empty string");
  if (!isString(market)) throw new Error("payload.market must be a non-empty string");
  if (!isString(horizon)) throw new Error("payload.horizon must be a non-empty string");
  if (!isString(generatedAt)) throw new Error("payload.generatedAt must be a non-empty string");

  if (!Array.isArray(series) || series.length === 0) {
    throw new Error("payload.series must be a non-empty array");
  }

  const normalizedSeries = series.map((pt, idx) => {
    if (!isObject(pt)) throw new Error(`series[${idx}] must be an object`);

    const date = pt["date"];
    const p10 = pt["p10"];
    const p50 = pt["p50"];
    const p90 = pt["p90"];

    if (!isString(date)) throw new Error(`series[${idx}].date must be a non-empty string`);
    if (!isNumber(p10)) throw new Error(`series[${idx}].p10 must be a finite number`);
    if (!isNumber(p50)) throw new Error(`series[${idx}].p50 must be a finite number`);
    if (!isNumber(p90)) throw new Error(`series[${idx}].p90 must be a finite number`);
    if (!(p10 <= p50 && p50 <= p90)) {
      throw new Error(`series[${idx}] must satisfy p10 <= p50 <= p90`);
    }

    return { date, p10, p50, p90 };
  });

  const payload: ModelForecastPayload = {
    modelId,
    market,
    horizon,
    generatedAt,
    series: normalizedSeries
  };

  if (isObject(diagnostics)) payload.diagnostics = diagnostics;

  return payload;
}
