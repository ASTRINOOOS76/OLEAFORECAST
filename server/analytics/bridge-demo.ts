import { spawnSync } from "node:child_process";
import { validateForecastPayload } from "./validate-forecast";

const input = {
  y: [4200, 4210, 4190, 4220, 4230, 4240, 4250, 4260],
  future_dates: ["t+1", "t+2", "t+3", "t+4"]
};

const res = spawnSync(
  "python",
  ["ml/run_baseline.py", "E0", "ES", "W4", "4"],
  {
    input: JSON.stringify(input),
    encoding: "utf-8"
  }
);

if (res.status !== 0) {
  console.error("Python failed:", res.stderr);
  process.exit(1);
}

const raw = JSON.parse(res.stdout);
const validated = validateForecastPayload(raw);

console.log("Bridge OK. First point:", validated.series[0]);
