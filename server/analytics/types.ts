export type MarketCode =
  | "ES"
  | "IT"
  | "GR"
  | "TN"
  | "TR"
  | "PT"
  | "MA"
  | "TU"
  | "OTHER";

export type Horizon =
  | "W1"
  | "W2"
  | "W4"
  | "W8"
  | "M1"
  | "M3"
  | "M6";

export type Interval = {
  p10: number;
  p50: number;
  p90: number;
};

export type ForecastPoint = {
  date: string; // ISO date
  value: number;
};

export type ForecastWithIntervals = {
  date: string; // ISO date
  p10: number;
  p50: number;
  p90: number;
};

export type ModelId = string;

export type ModelMetrics = {
  availability: number;      // 0..1
  biasEurPerTon: number;     // signed
  mase: number;              // >=0
  coverage: number;          // 0..1 (for p10/p90)
  intervalsAvailable: boolean;
};

export type ModelEvalInput = {
  modelId: ModelId;
  market: MarketCode;
  horizon: Horizon;
  medianPriceEurPerTon?: number;
  metrics: ModelMetrics;
};

export type GateDecision = {
  eligible: boolean;
  status: string;        // PASS / FAIL_*
  reasons: string[];
};

export type WinnerScoreBreakdown = {
  score: number;               // lower is better
  masePenalty: number;
  biasPenalty: number;
  coveragePenalty: number;
};

export type WinnerDecision = {
  market: MarketCode;
  horizon: Horizon;
  winnerModelId: ModelId;
  score: WinnerScoreBreakdown;
  gates: Record<ModelId, GateDecision>;
  ranked: Array<{
    modelId: ModelId;
    score: WinnerScoreBreakdown;
  }>;
  createdAt: string; // ISO datetime
};
