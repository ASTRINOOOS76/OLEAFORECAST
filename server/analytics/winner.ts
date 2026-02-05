import type { ModelEvalInput, WinnerDecision, WinnerScoreBreakdown, GateDecision, ModelId } from "./types";
import { evaluateHardFailGates, computeWinnerScoreBreakdown } from "./serving-gates";

export type WinnerSelectOptions = {
  // Minimum number of eligible models required; otherwise throw
  minEligible?: number;
};

function nowIso(): string {
  return new Date().toISOString();
}

export function selectWinner(
  market: WinnerDecision["market"],
  horizon: WinnerDecision["horizon"],
  inputs: ModelEvalInput[],
  opts: WinnerSelectOptions = {}
): WinnerDecision {
  const minEligible = opts.minEligible ?? 1;

  // Filter to matching (market, horizon) defensively
  const candidates = inputs.filter((x) => x.market === market && x.horizon === horizon);

  const gates: Record<ModelId, GateDecision> = {};
  const ranked: Array<{ modelId: ModelId; score: WinnerScoreBreakdown }> = [];

  for (const c of candidates) {
    const g = evaluateHardFailGates(c);
    gates[c.modelId] = g;

    if (!g.eligible) continue;

    const score = computeWinnerScoreBreakdown(c);
    ranked.push({ modelId: c.modelId, score });
  }

  ranked.sort((a, b) => a.score.score - b.score.score);

  if (ranked.length < minEligible) {
    const reasons = Object.entries(gates)
      .filter(([, g]) => !g.eligible)
      .map(([id, g]) => `${id}:${g.status}`)
      .join(", ");

    throw new Error(
      `No eligible models for ${market}/${horizon}. minEligible=${minEligible}. Ineligible: ${reasons || "none"}`
    );
  }

  const winner = ranked[0];

  return {
    market,
    horizon,
    winnerModelId: winner.modelId,
    score: winner.score,
    gates,
    ranked,
    createdAt: nowIso()
  };
}
