import type { GateDecision, ModelEvalInput } from "./types";
import { GATE_THRESHOLDS, evaluateHardFailGates } from "./serving-gates";

export type GateSummary = {
  eligible: boolean;
  status: string;
  notes: string[];
};

export function checkGates(input: ModelEvalInput): GateSummary {
  // Target band is 75–85% coverage for p10/p90 intervals.
  const notes: string[] = [];

  const r: GateDecision = evaluateHardFailGates(input);
  if (!r.eligible) {
    notes.push(...r.reasons);
    return { eligible: false, status: r.status, notes };
  }

  const cov = input.metrics.coverage ?? 0;
  if (cov > 0 && (cov < GATE_THRESHOLDS.COVERAGE_MIN || cov > GATE_THRESHOLDS.COVERAGE_MAX)) {
    notes.push(
      `coverage ${cov.toFixed(2)} outside preferred band ${GATE_THRESHOLDS.COVERAGE_MIN.toFixed(
        2
      )}-${GATE_THRESHOLDS.COVERAGE_MAX.toFixed(2)}`
    );
  }

  return { eligible: true, status: "PASS", notes };
}
