import { spawn } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

function run(cmd: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: "inherit" });
    p.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited with ${code}`))));
  });
}

function ensureDir(path: string) {
  mkdirSync(dirname(path), { recursive: true });
}

function parseArgs(argv: string[]) {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) continue;
    const k = a.slice(2);
    const v = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
    out[k] = v;
  }
  return out;
}

function csvList(s: string): string[] {
  return s.split(",").map((x) => x.trim()).filter(Boolean);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const markets = args.markets ? csvList(args.markets) : ["ES", "IT", "GR", "TN", "PT", "TR"];
  const horizon = (args.horizon || "W4").trim();
  const engine = (args.engine || "naive").trim(); // naive|xgb|prophet|ets
  const modelId = (args.modelId || "E0").trim();
  const grade = (args.grade || "EVOO").trim();
  const python = (args.python || "python3").trim();

  for (const m of markets) {
    const actualsPath = `data/${m}/${horizon}/actuals.csv`;
    if (!existsSync(actualsPath)) {
      throw new Error(`Missing actuals: ${actualsPath}`);
    }

    const outPath = `data/${m}/${horizon}/${modelId}.json`;
    ensureDir(outPath);

    await run(python, [
      "ml_legacy/run_model.py",
      "--market",
      m,
      "--horizon",
      horizon,
      "--model-id",
      modelId,
      "--engine",
      engine,
      "--grade",
      grade,
      "--actuals",
      actualsPath,
      "--out",
      outPath,
    ]);
  }

  console.log(
    `[forecasts] done modelId=${modelId} engine=${engine} horizon=${horizon} grade=${grade} markets=${markets.join(",")}`
  );
}

main().catch((e) => {
  console.error("[forecasts] ERROR:", e?.stack || e);
  process.exit(1);
});
