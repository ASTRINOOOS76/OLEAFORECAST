export type IntervalForecast = {
  date: string; // ISO date
  p10: number;
  p50: number;
  p90: number;
};

export type ModelForecastPayload = {
  modelId: string;
  market: string;
  horizon: string;
  generatedAt: string; // ISO datetime
  series: IntervalForecast[];
  diagnostics?: Record<string, unknown>;
};
