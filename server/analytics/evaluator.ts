import type { ModelEvalInput, MarketCode, Horizon } from "./types";
import type { ModelForecastPayload } from "./contracts";
import type { ActualPoint } from "./metrics";
import { computeMetrics } from "./metrics";

export type EvaluateOptions = {
  medianPriceEurPerTon?: number;
};

export function toEvalInput(
  payload: ModelForecastPayload,
  actuals: ActualPoint[],
  opts: EvaluateOptions = {}
): ModelEvalInput {
  const m = computeMetrics(payload.series, actuals);

  return {
    modelId: payload.modelId,
    market: (payload.market as MarketCode) ?? "OTHER",
    horizon: payload.horizon as Horizon,
    medianPriceEurPerTon: opts.medianPriceEurPerTon,
    metrics: {
      availability: m.availability,
      biasEurPerTon: m.biasEurPerTon,
      mase: m.mase,
      coverage: m.coverage,
      intervalsAvailable: true
    }
  };
}
