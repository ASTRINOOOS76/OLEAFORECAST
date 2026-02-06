import { buildCanonicalDaily, type CanonicalConfig } from "../analytics/canonical-price";
import { writeActualsCsv } from "./build-actuals-from-canonical";
import { fetchObservationsForMarkets } from "./fetch-observations.stub";

function toIsoDate(d: Date): string {
  return d.toISOString().slice(0, 10);
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

function defaultConfig(): CanonicalConfig {
  return {
    reliability: {
      poolred: 1.0,
      ismea: 0.95,
      sedik: 0.85,
      oleista: 0.7,
      dga: 0.8
    },
    halfLifeDays: {
      ES: 14,
      IT: 21,
      GR: 21,
      TN: 30,
      PT: 30,
      TR: 30
    },
    sigma0: {
      ES: 120,
      IT: 160,
      GR: 150,
      TN: 170,
      PT: 150,
      TR: 170
    },
    sigmaGrow: {
      ES: 35,
      IT: 45,
      GR: 45,
      TN: 55,
      PT: 50,
      TR: 55
    }
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  const markets = args.markets ? csvList(args.markets) : ["ES", "IT", "GR", "TN", "PT", "TR"];
  const horizon = (args.horizon || "W4").trim();

  const end = args.end ? new Date(args.end + "T00:00:00Z") : new Date();
  const start = args.start
    ? new Date(args.start + "T00:00:00Z")
    : new Date(end.getTime() - 180 * 24 * 3600 * 1000);

  const startDate = toIsoDate(start);
  const endDate = toIsoDate(end);

  console.log(`[canonical] markets=${markets.join(",")} horizon=${horizon} range=${startDate}..${endDate}`);

  const observations = await fetchObservationsForMarkets(markets, startDate, endDate);
  console.log(`[canonical] fetched observations: ${observations.length}`);

  const cfg = defaultConfig();

  for (const m of markets) {
    const points = buildCanonicalDaily(m, startDate, endDate, observations, cfg);

    // Write p50 daily series as actuals (EUR/ton)
    const outPath = `data/${m}/${horizon}/actuals.csv`;
    writeActualsCsv(outPath, points);

    const finiteCount = points.filter((p) => Number.isFinite(p.p50)).length;
    const lastFinite = [...points].reverse().find((p) => Number.isFinite(p.p50));

    console.log(
      `[canonical] wrote ${outPath} rows=${finiteCount}` +
        (lastFinite
          ? ` last=${lastFinite.date} p50=${lastFinite.p50.toFixed(2)} ageDays=${lastFinite.ageDays} conf=${lastFinite.confidence.toFixed(3)}`
          : "")
    );
  }

  console.log("[canonical] done");
}

main().catch((e) => {
  console.error("[canonical] ERROR:", e?.stack || e);
  process.exit(1);
});
