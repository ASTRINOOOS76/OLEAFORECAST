import http from "http";
import { existsSync, readFileSync, readdirSync, statSync } from "fs";
import { extname, join, resolve } from "path";
import { URL } from "url";

type WinnerRow = {
  market: string;
  horizon: string;
  winnerModelId?: string;
};

type SummaryItem = {
  market: string;
  horizon: string;
  winnerModelId: string | null;
  latest: { date: string; value: number } | null;
};

const ROOT = resolve(__dirname, "..");
const DATA_DIR = join(ROOT, "data");
const FIXTURES_DIR = join(DATA_DIR, "_fixtures");
const REPORTS_DIR = join(ROOT, "reports");
const CLIENT_DIR = join(ROOT, "client");
const PORT = Number(process.env.PORT ?? 3000);

function readJson<T>(filePath: string): T {
  return JSON.parse(readFileSync(filePath, "utf-8")) as T;
}

function firstExisting(paths: string[]): string | null {
  for (const path of paths) {
    if (existsSync(path)) return path;
  }
  return null;
}

function listMarketHorizonPairs(): Array<{ market: string; horizon: string }> {
  const pairs = new Set<string>();
  for (const base of [DATA_DIR, FIXTURES_DIR]) {
    if (!existsSync(base)) continue;
    for (const market of readdirSync(base)) {
      if (market.startsWith("_")) continue;
      const marketPath = join(base, market);
      if (!statSync(marketPath).isDirectory()) continue;
      for (const horizon of readdirSync(marketPath)) {
        const horizonPath = join(marketPath, horizon);
        if (!statSync(horizonPath).isDirectory()) continue;
        if (!existsSync(join(horizonPath, "actuals.csv"))) continue;
        pairs.add(`${market}/${horizon}`);
      }
    }
  }
  return [...pairs].map((value) => {
    const [market, horizon] = value.split("/");
    return { market, horizon };
  });
}

function loadWinners(): WinnerRow[] {
  const path = firstExisting([
    join(REPORTS_DIR, "winners.json"),
    join(FIXTURES_DIR, "winners.json")
  ]);
  if (!path) return [];
  const payload = readJson<{ results?: WinnerRow[] }>(path);
  return payload.results ?? [];
}

function loadActuals(market: string, horizon: string): Array<{ date: string; value: number }> {
  const path = firstExisting([
    join(DATA_DIR, market, horizon, "actuals.csv"),
    join(FIXTURES_DIR, market, horizon, "actuals.csv")
  ]);
  if (!path) return [];
  const raw = readFileSync(path, "utf-8").trim();
  const lines = raw.split(/\r?\n/).filter(Boolean);
  if (lines.length <= 1) return [];
  const rows = lines.slice(1).map((line) => line.split(","));
  return rows
    .map(([date, value]) => ({ date, value: Number(value) }))
    .filter((row) => row.date && Number.isFinite(row.value));
}

function loadForecast(market: string, horizon: string, modelId: string) {
  const path = firstExisting([
    join(DATA_DIR, market, horizon, `${modelId}.json`),
    join(FIXTURES_DIR, market, horizon, `${modelId}.json`)
  ]);
  if (!path) return null;
  return readJson<{ series?: Array<Record<string, number | string>> }>(path);
}

function latestActual(actuals: Array<{ date: string; value: number }>) {
  if (actuals.length === 0) return null;
  const sorted = [...actuals].sort((a, b) => a.date.localeCompare(b.date));
  return sorted[sorted.length - 1];
}

function makeSummary(): { generatedAt: string; items: SummaryItem[]; markets: string[]; horizons: string[] } {
  const winners = loadWinners();
  const pairs = listMarketHorizonPairs();
  const markets = [...new Set(pairs.map((p) => p.market))].sort();
  const horizons = [...new Set(pairs.map((p) => p.horizon))].sort();
  const items = pairs.map((pair) => {
    const winner = winners.find(
      (row) => row.market === pair.market && row.horizon === pair.horizon
    );
    const actuals = loadActuals(pair.market, pair.horizon);
    return {
      market: pair.market,
      horizon: pair.horizon,
      winnerModelId: winner?.winnerModelId ?? null,
      latest: latestActual(actuals)
    };
  });
  return {
    generatedAt: new Date().toISOString(),
    items,
    markets,
    horizons
  };
}

function sendJson(res: http.ServerResponse, status: number, payload: unknown) {
  const body = JSON.stringify(payload, null, 2);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store"
  });
  res.end(body);
}

function sendText(res: http.ServerResponse, status: number, body: string, type = "text/plain") {
  res.writeHead(status, {
    "Content-Type": `${type}; charset=utf-8",
    "Cache-Control": "no-store"
  });
  res.end(body);
}

function serveStatic(res: http.ServerResponse, filePath: string) {
  if (!existsSync(filePath)) {
    sendText(res, 404, "Not found");
    return;
  }
  const content = readFileSync(filePath);
  const ext = extname(filePath);
  const type = ext === ".html" ? "text/html" : "application/octet-stream";
  res.writeHead(200, {
    "Content-Type": `${type}; charset=utf-8",
    "Cache-Control": "no-store"
  });
  res.end(content);
}

const server = http.createServer((req, res) => {
  if (!req.url) {
    sendText(res, 400, "Bad request");
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host ?? "localhost"}`);
  const path = url.pathname;

  if (path === "/api/summary") {
    sendJson(res, 200, makeSummary());
    return;
  }

  if (path === "/api/actuals") {
    const market = url.searchParams.get("market") ?? "";
    const horizon = url.searchParams.get("horizon") ?? "";
    if (!market || !horizon) {
      sendJson(res, 400, { error: "market and horizon are required" });
      return;
    }
    sendJson(res, 200, { market, horizon, series: loadActuals(market, horizon) });
    return;
  }

  if (path === "/api/winner") {
    const market = url.searchParams.get("market") ?? "";
    const horizon = url.searchParams.get("horizon") ?? "";
    if (!market || !horizon) {
      sendJson(res, 400, { error: "market and horizon are required" });
      return;
    }
    const winners = loadWinners();
    const winner = winners.find(
      (row) => row.market === market && row.horizon === horizon
    );
    sendJson(res, 200, { market, horizon, winnerModelId: winner?.winnerModelId ?? null });
    return;
  }

  if (path === "/api/forecast") {
    const market = url.searchParams.get("market") ?? "";
    const horizon = url.searchParams.get("horizon") ?? "";
    const model = url.searchParams.get("model") ?? "";
    if (!market || !horizon || !model) {
      sendJson(res, 400, { error: "market, horizon, and model are required" });
      return;
    }
    const payload = loadForecast(market, horizon, model);
    if (!payload) {
      sendJson(res, 404, { error: "forecast not found" });
      return;
    }
    sendJson(res, 200, { market, horizon, model, ...payload });
    return;
  }

  if (path === "/" || path === "/index.html") {
    serveStatic(res, join(CLIENT_DIR, "index.html"));
    return;
  }

  sendText(res, 404, "Not found");
});

server.listen(PORT, () => {
  console.log(`OleaForecast app running on http://localhost:${PORT}`);
});