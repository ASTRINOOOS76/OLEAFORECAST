import { selectWinner } from "../winner";
import type { ModelEvalInput } from "../types";

function assert(cond: unknown, msg: string) {
  if (!cond) throw new Error(`ASSERT FAIL: ${msg}`);
}

const base = { market: "ES", horizon: "W4" } as const;

const inputs: ModelEvalInput[] = [
  {
    modelId: "E0",
    market: base.market,
    horizon: base.horizon,
    medianPriceEurPerTon: 4200,
    metrics: {
      availability: 0.95,
      biasEurPerTon: 40,
      mase: 1.05,
      coverage: 0.8,
      intervalsAvailable: true
    }
  },
  {
    modelId: "E1",
    market: base.market,
    horizon: base.horizon,
    medianPriceEurPerTon: 4200,
    metrics: {
      availability: 0.98,
      biasEurPerTon: 10,
      mase: 0.95,
      coverage: 0,
      intervalsAvailable: false
    }
  },
  {
    modelId: "E2",
    market: base.market,
    horizon: base.horizon,
    medianPriceEurPerTon: 4200,
    metrics: {
      availability: 0.92,
      biasEurPerTon: 180,
      mase: 1.0,
      coverage: 0.8,
      intervalsAvailable: true
    }
  }
];

const d = selectWinner(base.market, base.horizon, inputs);

assert(d.winnerModelId === "E0", `expected winner E0, got ${d.winnerModelId}`);
assert(d.gates["E1"].eligible === false, "E1 must be ineligible");
assert(d.ranked.length === 2, `expected 2 eligible models ranked, got ${d.ranked.length}`);

console.log("winner.smoke OK");
