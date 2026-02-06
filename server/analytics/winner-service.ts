import type { ModelEvalInput } from "./types";
import type { WinnerDecision } from "./types";
import { selectWinner } from "./winner";

export function decideWinner(
  market: WinnerDecision["market"],
  horizon: WinnerDecision["horizon"],
  evals: ModelEvalInput[]
): WinnerDecision {
  return selectWinner(market, horizon, evals, { minEligible: 1 });
}
