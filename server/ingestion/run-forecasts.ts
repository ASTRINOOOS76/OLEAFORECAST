import { spawn } from "node:child_process";
import { existsSync, readFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

type Registry = {
  defaults: { horizons: string[]; markets: string[]; grade?: string };
  models: Array<{
    modelId: string;
    engine: "xgb" | "prophet" | "ets" | "naive";
    grade?: string;
    variant?: string;   // naive variants
    seriesKey?: string; // override artifact key
  }>;
};

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
  return s.split(",").map(x => x.trim()).filter(Boolean);
}

function loadRegistry(path: string): Registry {
  const raw = readFileSync(path, "utf8");
  return JSON.parse(raw) as Registry;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const registryPath = (args.registry || "server/models/model-registry.json").trim();
  const python = (args.python || "python3").trim();

  if (!existsSync(registryPath)) throw new Error(`Missing registry file: ${registryPath}`);

  const reg = loadRegistry(registryPath);

  const markets = args.markets ? csvList(args.markets) : reg.defaults.markets;
  const horizons = args.horizons ? csvList(args.horizons) : reg.defaults.horizons;

  for (const h of horizons) {
    for (const m of markets) {
      const actualsPath = `data/${m}/${h}/actuals.csv`;
      if (!existsSync(actualsPath)) throw new Error(`Missing actuals: ${actualsPath}`);

      for (const model of reg.models) {
        const outPath = `data/${m}/${h}/${model.modelId}.json`;
        ensureDir(outPath);

        const grade = (model.grade || reg.defaults.grade || "EVOO").trim();
        const engine = model.engine;
        const variant = model.variant || "last";
        const seriesKey = model.seriesKey || `${grade}_${m}`;

        await run(python, [
          "ml_legacy/run_model.py",
          "--market", m,
          "--horizon", h,
          "--model-id", model.modelId,
          "--engine", engine,
          "--grade", grade,
          "--variant", variant,
          "--series-key", seriesKey,
          "--actuals", actualsPath,
          "--out", outPath
        ]);
      }
    }
  }

  console.log(`[forecasts] done registry=${registryPath}`);
}

main().catch((e) => {
  console.error("[forecasts] ERROR:", e?.stack || e);
  process.exit(1);
});
