import { existsSync, readFileSync } from "node:fs";

function assert(cond: any, msg: string) {
  if (!cond) throw new Error(msg);
}

const winnersPath = "reports/winners.json";

// run batch over fixtures root
// NOTE: run-smoke.sh should invoke this file; it will call eval:batch itself there.
// Here we just validate output exists and contains ES/W4 once created.

assert(true, "batch.smoke precheck OK");

// If run-smoke.sh executes eval:batch before this test, winners.json will exist.
// If not, we fail loudly so you fix the script order.
assert(existsSync(winnersPath), `Missing ${winnersPath}. Ensure smoke runner executes eval:batch over data/_fixtures first.`);

const raw = readFileSync(winnersPath, "utf-8");
const obj = JSON.parse(raw);

assert(Array.isArray(obj.results), "winners.json missing results[]");

const hit = obj.results.find((r: any) => r.market === "ES" && r.horizon === "W4");
assert(hit, "No ES/W4 result in winners.json");
assert(typeof hit.winnerModelId === "string" && hit.winnerModelId.length > 0, "Missing winnerModelId");

console.log("batch.smoke OK");
