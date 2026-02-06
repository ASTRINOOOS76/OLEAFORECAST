/**
 * Serving Gates & Winner Scoring
 * - Hard-fail eligibility checks
 * - Winner score computation
 */

import type { GateDecision, ModelEvalInput, WinnerScoreBreakdown } from "./types";

export type GateStatus =
  | "PASS"
  | "FAIL_AVAILABILITY"
  | "FAIL_BIAS"
  | "FAIL_MASE"
  | "FAIL_COVERAGE"
  | "NO_INTERVALS";

export const GATE_THRESHOLDS = {
  AVAILABILITY_MIN: 0.0,
  BIAS_MAX: 250, // EUR/t
  MASE_MAX: 1.5,
  COVERAGE_MIN: 0.75,
  COVERAGE_MAX: 0.85,
  COVERAGE_TARGET: 0.8
} as const;

export function evaluateHardFailGates(input: ModelEvalInput): GateDecision {
  const { metrics } = input;
  const reasons: string[] = [];

  if (metrics.availability < GATE_THRESHOLDS.AVAILABILITY_MIN) {
    reasons.push(
      `availability ${metrics.availability.toFixed(2)} < ${GATE_THRESHOLDS.AVAILABILITY_MIN.toFixed(
        2
      )}`
    );
    return { eligible: false, status: "FAIL_AVAILABILITY", reasons };
  }

  if (Math.abs(metrics.biasEurPerTon) > GATE_THRESHOLDS.BIAS_MAX) {
    reasons.push(
      `bias ${Math.abs(metrics.biasEurPerTon).toFixed(0)} > ${GATE_THRESHOLDS.BIAS_MAX}`
    );
    return { eligible: false, status: "FAIL_BIAS", reasons };
  }

  if (metrics.mase > GATE_THRESHOLDS.MASE_MAX) {
    reasons.push(`MASE ${metrics.mase.toFixed(2)} > ${GATE_THRESHOLDS.MASE_MAX}`);
    return { eligible: false, status: "FAIL_MASE", reasons };
  }

  if (!metrics.intervalsAvailable || (metrics.coverage ?? 0) <= 0) {
    reasons.push("prediction intervals not available");
    return { eligible: false, status: "NO_INTERVALS", reasons };
  }

  if (metrics.coverage < GATE_THRESHOLDS.COVERAGE_MIN) {
    reasons.push(
      `coverage ${metrics.coverage.toFixed(2)} < ${GATE_THRESHOLDS.COVERAGE_MIN.toFixed(2)}`
    );
    return { eligible: false, status: "FAIL_COVERAGE", reasons };
  }

  return { eligible: true, status: "PASS", reasons };
}

export function computeWinnerScoreBreakdown(input: ModelEvalInput): WinnerScoreBreakdown {
  const { metrics } = input;

  // Lower score is better.
  const masePenalty = metrics.mase;

  const absBiasPenalty = Math.abs(metrics.biasEurPerTon) / GATE_THRESHOLDS.BIAS_MAX;

  const price = input.medianPriceEurPerTon ?? 0;
  const relBiasPenalty =
    price > 0 ? Math.abs(metrics.biasEurPerTon) / Math.max(1, 0.05 * price) : absBiasPenalty;

  // IMPORTANT: stricter penalty is the MAX, not the MIN
  const biasPenalty = Math.max(absBiasPenalty, relBiasPenalty);

  const cov = metrics.coverage ?? 0;
  const covTarget = GATE_THRESHOLDS.COVERAGE_TARGET;

  const coveragePenalty = cov > 0 ? Math.abs(cov - covTarget) / covTarget : 1.0;

  const score = masePenalty + 0.35 * biasPenalty + 0.25 * coveragePenalty;

  return { score, masePenalty, biasPenalty, coveragePenalty };
}
