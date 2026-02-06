import type { ModelEvalInput } from "./types";

export type EvalRegistry = {
  updatedAt: string;
  items: ModelEvalInput[];
};

export function makeRegistry(items: ModelEvalInput[]): EvalRegistry {
  return {
    updatedAt: new Date().toISOString(),
    items
  };
}

export function upsertEval(items: ModelEvalInput[], next: ModelEvalInput): ModelEvalInput[] {
  const i = items.findIndex(
    (x) => x.modelId === next.modelId && x.market === next.market && x.horizon === next.horizon
  );
  if (i === -1) return [...items, next];
  const copy = items.slice();
  copy[i] = next;
  return copy;
}
