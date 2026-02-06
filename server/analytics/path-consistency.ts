import type { ModelForecastPayload } from "./contracts";

export function assertPathConsistency(filePath: string, payload: ModelForecastPayload) {
  // expected: data/<MARKET>/<HORIZON>/<MODEL>.json
  const norm = filePath.replace(/\\/g, "/");
  const parts = norm.split("/").filter(Boolean);

  const i = parts.findIndex((p) => p === "data");
  if (i === -1 || parts.length < i + 4) return; // if not under /data, skip

  const market = parts[i + 1];
  const horizon = parts[i + 2];
  const filename = parts[i + 3];
  const model = filename.toLowerCase().endsWith(".json")
    ? filename.slice(0, -5)
    : filename;

  if (payload.market !== market) {
    throw new Error(`Path/JSON market mismatch: path=${market} json=${payload.market} (${filePath})`);
  }
  if (payload.horizon !== horizon) {
    throw new Error(`Path/JSON horizon mismatch: path=${horizon} json=${payload.horizon} (${filePath})`);
  }
  if (payload.modelId !== model) {
    throw new Error(`Path/JSON modelId mismatch: path=${model} json=${payload.modelId} (${filePath})`);
  }
}
