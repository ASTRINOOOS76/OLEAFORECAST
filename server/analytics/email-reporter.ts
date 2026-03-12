/**
 * OLEA Intelligence Email Reporter
 *
 * Two distinct report paths (Issue B):
 *   generatePreviewReport()  — read-only, no writes, fast
 *   generatePublishReport()  — full pipeline, writes weekly JSON, logs delivery
 *
 * PDF export uses real Playwright PDF rendering (Issue A).
 */

// ─── Static imports (Issue K: no dynamic imports for internal modules) ────────
import { createHmac, randomBytes } from "crypto";
import { existsSync, readFileSync } from "fs";
import { join } from "path";

import { loadActualsCsv } from "./load-actuals";
import type { ActualPoint } from "./metrics";
import { writeJson } from "./write-report";
import type { MarketCode, Horizon } from "./types";

// ─── Feedback signing secret (Issue H) ────────────────────────────────────────
// Use FEEDBACK_SECRET — never EMAIL_PASS or any SMTP credential.
let _feedbackSecret: string | undefined;
function getFeedbackSecret(): string {
  if (_feedbackSecret) return _feedbackSecret;
  const envSecret = process.env.FEEDBACK_SECRET;
  if (!envSecret) {
    console.warn(
      "[email-reporter] FEEDBACK_SECRET env var is not set. " +
        "Generating an ephemeral signing key. Set FEEDBACK_SECRET in production."
    );
    _feedbackSecret = randomBytes(32).toString("hex");
  } else {
    _feedbackSecret = envSecret;
  }
  return _feedbackSecret;
}

// ─── Path helpers ──────────────────────────────────────────────────────────────
const ROOT = join(__dirname, "..", "..");
const DATA_DIR = join(ROOT, "data");
const REPORTS_DIR = join(ROOT, "reports");

function firstExisting(paths: string[]): string | null {
  for (const p of paths) {
    if (existsSync(p)) return p;
  }
  return null;
}

// ─── Unit conversion ──────────────────────────────────────────────────────────
/**
 * Forecast and actuals CSVs store prices in EUR/ton.
 * Canonical CSVs store prices in EUR/kg.
 * All display values are normalised to EUR/kg.
 */
function eurTonToKg(v: number): number {
  return v / 1000;
}

// ─── Local types ──────────────────────────────────────────────────────────────
interface WinnersFile {
  generatedAt: string;
  results: Array<{
    market: string;
    horizon: string;
    winnerModelId?: string;
    top3?: string[];
    ineligible?: Array<{ modelId: string; reasons: string[] }>;
    actualsQuality?: {
      count: number;
      duplicates: unknown[];
      nonFinite: number;
      isSorted: boolean;
    };
  }>;
}

interface ForecastFile {
  modelId: string;
  market: string;
  horizon: string;
  generatedAt: string;
  series: Array<{ date: string; p10: number; p50: number; p90: number }>;
}

export interface AccuracyRecord {
  countryCode: string;
  mape: number;
  hitRate: number;
  n: number;
}

interface AccuracyScorecard {
  byCountry: AccuracyRecord[];
}

interface RetrospectiveRow {
  countryCode: string;
  predictedValue: number; // EUR/kg
  actualValue: number; // EUR/kg
  priorActual: number | null; // EUR/kg
  withinBounds: boolean;
  direction: "correct" | "wrong" | "unknown";
}

interface ForecastMatrixRow {
  market: MarketCode;
  horizon: Horizon;
  currentPriceEurKg: number | null;
  forecastP50EurKg: number | null;
  forecastP10EurKg: number | null;
  forecastP90EurKg: number | null;
  modelId: string;
  trend: "up" | "down" | "flat";
}

interface MarketPulse {
  grEvoo: number | null; // EUR/kg
  esEvoo: number | null; // EUR/kg
  itEvoo: number | null; // EUR/kg
  spreadGrEs: number | null; // EUR/kg — Issue D: no /1000
  narrativeHtml: string;
}

export interface LiveSnapshot {
  generatedAt: string;
  forecastMatrix: ForecastMatrixRow[];
  retrospective: RetrospectiveRow[];
  accuracyScorecard: AccuracyScorecard | null;
  marketPulse: MarketPulse;
  garchVolatility: number | null;
}

export interface PublishResult {
  html: string;
  snapshot: LiveSnapshot;
  sentAt: string;
}

/** Vegetable oil complex entry (Issue L: no .date field). */
export interface VegOilCommodity {
  name: string;
  priceEurKg: number | null;
  changeWeekPct: number | null;
}

// ─── Supported report scope ────────────────────────────────────────────────────
const REPORT_MARKETS: MarketCode[] = ["GR", "ES", "IT", "PT", "TN", "TR"];
const REPORT_HORIZON: Horizon = "W4";

const MARKET_LABELS: Record<string, string> = {
  GR: "Ελλάδα",
  ES: "Ισπανία",
  IT: "Ιταλία",
  PT: "Πορτογαλία",
  TN: "Τυνησία",
  TR: "Τουρκία",
};

const TREND_SYMBOL: Record<string, string> = {
  up: "▲",
  down: "▼",
  flat: "→",
};

// ─── Data loaders ──────────────────────────────────────────────────────────────
function loadWinnersFile(): WinnersFile {
  const path = firstExisting([
    join(REPORTS_DIR, "winners.json"),
    join(DATA_DIR, "_fixtures", "winners.json"),
  ]);
  if (!path) return { generatedAt: "", results: [] };
  try {
    return JSON.parse(readFileSync(path, "utf-8")) as WinnersFile;
  } catch {
    return { generatedAt: "", results: [] };
  }
}

/**
 * Returns the latest canonical price in EUR/kg for a market.
 * Canonical CSVs: date,value_eur_kg,staleDays,sources
 */
function loadCanonicalLatestEurKg(market: MarketCode): number | null {
  const path = join(DATA_DIR, "canonical", `${market}_EVOO.csv`);
  if (!existsSync(path)) return null;
  try {
    const lines = readFileSync(path, "utf-8")
      .trim()
      .split(/\r?\n/)
      .filter(Boolean);
    if (lines.length < 2) return null;
    const last = lines[lines.length - 1].split(",");
    if (last.length < 2) return null;
    const val = parseFloat(last[1]);
    return Number.isFinite(val) ? val : null;
  } catch {
    return null;
  }
}

/**
 * Returns the latest actual price in EUR/kg, converted from EUR/ton.
 * Falls back when canonical data is unavailable.
 */
function loadLatestActualEurKg(market: MarketCode, horizon: Horizon): number | null {
  const path = firstExisting([
    join(DATA_DIR, market, horizon, "actuals.csv"),
    join(DATA_DIR, "_fixtures", market, horizon, "actuals.csv"),
  ]);
  if (!path) return null;
  try {
    const actuals = loadActualsCsv(path);
    if (actuals.length === 0) return null;
    const last = actuals[actuals.length - 1];
    if (!Number.isFinite(last.value) || last.value === 0) return null;
    // Heuristic: values > 100 are EUR/ton, otherwise EUR/kg
    return last.value > 100 ? eurTonToKg(last.value) : last.value;
  } catch {
    return null;
  }
}

function loadForecastFile(
  market: MarketCode,
  horizon: Horizon,
  modelId: string
): ForecastFile | null {
  const path = firstExisting([
    join(DATA_DIR, market, horizon, `${modelId}.json`),
    join(DATA_DIR, "_fixtures", market, horizon, `${modelId}.json`),
  ]);
  if (!path) return null;
  try {
    const raw = JSON.parse(readFileSync(path, "utf-8")) as ForecastFile;
    if (!Array.isArray(raw.series)) return null;
    return raw;
  } catch {
    return null;
  }
}

// ─── HTML sanitization (Issue M) ──────────────────────────────────────────────
/**
 * Strip <script> blocks and inline event handlers from AI-generated narratives
 * before injecting into HTML. Does not touch <strong>, <em>, <br>.
 */
export function sanitizeNarrativeHtml(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, "")
    .replace(/(<[^>]+?)\s+on\w+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]*)/gi, "$1");
}

// ─── Direction computation (Issue F) ──────────────────────────────────────────
/**
 * Determines directional correctness by comparing predicted vs actual movement
 * relative to the prior period's actual value.
 *
 * - If no prior actual is available, falls back to withinBounds.
 * - Returns "unknown" rather than fabricating a result.
 */
function computeDirection(
  predictedValue: number,
  actualValue: number,
  priorActual: number | null,
  withinBounds: boolean
): "correct" | "wrong" | "unknown" {
  if (priorActual === null) {
    if (withinBounds) return "correct";
    return "unknown";
  }
  const predictedDir = predictedValue > priorActual ? "up" : "down";
  const actualDir = actualValue > priorActual ? "up" : "down";
  return predictedDir === actualDir ? "correct" : "wrong";
}

// ─── Market pulse (Issue D) ───────────────────────────────────────────────────
/**
 * Prices are in EUR/kg. Spread is computed directly without dividing by 1000.
 * Issue D fix: spreadPerKg = grEvoo - esEvoo  (was wrongly / 1000)
 */
function buildMarketPulse(snapshot: Omit<LiveSnapshot, "marketPulse">): MarketPulse {
  const grRow = snapshot.forecastMatrix.find((r) => r.market === "GR");
  const esRow = snapshot.forecastMatrix.find((r) => r.market === "ES");
  const itRow = snapshot.forecastMatrix.find((r) => r.market === "IT");

  const grEvoo = grRow?.currentPriceEurKg ?? null;
  const esEvoo = esRow?.currentPriceEurKg ?? null;
  const itEvoo = itRow?.currentPriceEurKg ?? null;

  // Issue D: prices already in EUR/kg — no division by 1000
  const spreadGrEs =
    grEvoo !== null && esEvoo !== null ? grEvoo - esEvoo : null;

  let narrativeHtml = "";
  if (grEvoo !== null && esEvoo !== null && spreadGrEs !== null) {
    const direction = spreadGrEs > 0 ? "υψηλότερα" : spreadGrEs < 0 ? "χαμηλότερα" : "στο ίδιο επίπεδο";
    const absSpread = Math.abs(spreadGrEs).toFixed(2);
    narrativeHtml = `<p>Η ελληνική EVOO διαπραγματεύεται <strong>${absSpread} EUR/kg ${direction}</strong> από την ισπανική EVOO.</p>`;
  }

  return { grEvoo, esEvoo, itEvoo, spreadGrEs, narrativeHtml };
}

// ─── Accuracy scorecard (Issue E) ─────────────────────────────────────────────
/**
 * Derives real accuracy metrics from actual forecast vs actuals data.
 * No hardcoded numbers. Returns null when no data is available.
 */
function deriveAccuracyScorecard(): AccuracyScorecard | null {
  const winnersFile = loadWinnersFile();
  if (!winnersFile.results.length) return null;

  const byCountry: AccuracyRecord[] = [];

  for (const market of REPORT_MARKETS) {
    const winnerRecord = winnersFile.results.find(
      (w) => w.market === market && w.horizon === REPORT_HORIZON
    );
    if (!winnerRecord?.winnerModelId) continue;

    const actualsPath = firstExisting([
      join(DATA_DIR, market, REPORT_HORIZON, "actuals.csv"),
      join(DATA_DIR, "_fixtures", market, REPORT_HORIZON, "actuals.csv"),
    ]);
    if (!actualsPath) continue;

    let actuals: ActualPoint[];
    try {
      actuals = loadActualsCsv(actualsPath);
    } catch {
      continue;
    }
    if (actuals.length === 0) continue;

    const forecastFile = loadForecastFile(
      market as MarketCode,
      REPORT_HORIZON,
      winnerRecord.winnerModelId
    );
    if (!forecastFile) continue;

    const actualMap = new Map(actuals.map((a) => [a.date, a.value]));

    let mapeSum = 0;
    let mapeCount = 0;
    let hitCount = 0;
    let matchCount = 0;

    for (const f of forecastFile.series) {
      const actual = actualMap.get(f.date);
      if (actual === undefined) continue;

      matchCount++;

      // Hit rate: actual within [p10, p90]
      if (actual >= f.p10 && actual <= f.p90) hitCount++;

      // MAPE: skip zero-actual rows to avoid division by zero
      if (actual !== 0) {
        mapeSum += Math.abs(f.p50 - actual) / Math.abs(actual);
        mapeCount++;
      }
    }

    if (matchCount === 0) continue;

    byCountry.push({
      countryCode: market,
      mape: mapeCount > 0 ? (mapeSum / mapeCount) * 100 : 0,
      hitRate: hitCount / matchCount,
      n: matchCount,
    });
  }

  return byCountry.length > 0 ? { byCountry } : null;
}

// ─── AccuracyRecord[] derivation from scorecard (Issue E helper) ──────────────
/**
 * Maps AccuracyScorecard.byCountry → AccuracyRecord[].
 * If scorecard is null or empty, returns []. Never invents numbers.
 */
export function deriveAccuracyRecords(
  scorecard: AccuracyScorecard | null
): AccuracyRecord[] {
  if (!scorecard || scorecard.byCountry.length === 0) return [];
  return scorecard.byCountry.map((entry) => ({
    countryCode: entry.countryCode,
    mape: entry.mape,
    hitRate: entry.hitRate,
    n: entry.n,
  }));
}

// ─── GARCH volatility with 8s timeout (Issue J) ───────────────────────────────
async function fetchGarchVolatility(): Promise<number | null> {
  const url = process.env.GARCH_SERVICE_URL;
  if (!url) return null;

  const timeoutGuard = new Promise<null>((resolve) =>
    setTimeout(() => resolve(null), 8_000)
  );

  const fetchAttempt = fetch(url)
    .then((r) => r.json() as Promise<unknown>)
    .then((data) => {
      if (data && typeof data === "object" && "volatility" in data) {
        const v = (data as Record<string, unknown>)["volatility"];
        return typeof v === "number" ? v : null;
      }
      return null;
    })
    .catch(() => null);

  const result = await Promise.race([fetchAttempt, timeoutGuard]);
  if (result === null) {
    console.warn(
      "[email-reporter] GARCH service timed out or unavailable – continuing without volatility"
    );
  }
  return result;
}

// ─── Forecast matrix builder (Issue I: parallelised with Promise.allSettled) ──
async function buildForecastMatrix(): Promise<ForecastMatrixRow[]> {
  const winnersFile = loadWinnersFile();

  const settled = await Promise.allSettled(
    REPORT_MARKETS.map(async (market): Promise<ForecastMatrixRow> => {
      const winnerRecord = winnersFile.results.find(
        (w) => w.market === market && w.horizon === REPORT_HORIZON
      );

      // Parallel: canonical price fetch + forecast data read (Issue I)
      const [currentPriceEurKg, forecastFile] = await Promise.all([
        Promise.resolve(
          loadCanonicalLatestEurKg(market) ??
            loadLatestActualEurKg(market, REPORT_HORIZON)
        ),
        Promise.resolve(
          winnerRecord?.winnerModelId
            ? loadForecastFile(market, REPORT_HORIZON, winnerRecord.winnerModelId)
            : null
        ),
      ]);

      if (!winnerRecord?.winnerModelId || !forecastFile) {
        return {
          market,
          horizon: REPORT_HORIZON,
          currentPriceEurKg,
          forecastP50EurKg: null,
          forecastP10EurKg: null,
          forecastP90EurKg: null,
          modelId: winnerRecord?.winnerModelId ?? "N/A",
          trend: "flat",
        };
      }

      const lastEntry =
        forecastFile.series[forecastFile.series.length - 1] ?? null;

      // Convert EUR/ton → EUR/kg for display
      const forecastP50EurKg = lastEntry ? eurTonToKg(lastEntry.p50) : null;
      const forecastP10EurKg = lastEntry ? eurTonToKg(lastEntry.p10) : null;
      const forecastP90EurKg = lastEntry ? eurTonToKg(lastEntry.p90) : null;

      let trend: "up" | "down" | "flat" = "flat";
      if (
        forecastP50EurKg !== null &&
        currentPriceEurKg !== null &&
        currentPriceEurKg > 0
      ) {
        const delta = (forecastP50EurKg - currentPriceEurKg) / currentPriceEurKg;
        if (delta > 0.005) trend = "up";
        else if (delta < -0.005) trend = "down";
      }

      return {
        market,
        horizon: REPORT_HORIZON,
        currentPriceEurKg,
        forecastP50EurKg,
        forecastP10EurKg,
        forecastP90EurKg,
        modelId: winnerRecord.winnerModelId,
        trend,
      };
    })
  );

  return settled
    .filter(
      (r): r is PromiseFulfilledResult<ForecastMatrixRow> =>
        r.status === "fulfilled"
    )
    .map((r) => r.value);
}

// ─── Retrospective builder (Issue I: parallelised + Issue F: correct direction) ─
async function buildRetrospective(): Promise<RetrospectiveRow[]> {
  const winnersFile = loadWinnersFile();

  const settled = await Promise.allSettled(
    REPORT_MARKETS.map(async (market): Promise<RetrospectiveRow[]> => {
      const winnerRecord = winnersFile.results.find(
        (w) => w.market === market && w.horizon === REPORT_HORIZON
      );
      if (!winnerRecord?.winnerModelId) return [];

      // Parallel: actuals + forecast data (Issue I YoY-style lookback pattern)
      const [actuals, forecastFile] = await Promise.all([
        (async (): Promise<ActualPoint[]> => {
          const path = firstExisting([
            join(DATA_DIR, market, REPORT_HORIZON, "actuals.csv"),
            join(DATA_DIR, "_fixtures", market, REPORT_HORIZON, "actuals.csv"),
          ]);
          if (!path) return [];
          try {
            return loadActualsCsv(path);
          } catch {
            return [];
          }
        })(),
        Promise.resolve(
          loadForecastFile(market, REPORT_HORIZON, winnerRecord.winnerModelId)
        ),
      ]);

      if (actuals.length === 0 || !forecastFile) return [];

      type ForecastEntry = ForecastFile["series"][number];
      const forecastMap = new Map<string, ForecastEntry>(
        forecastFile.series.map((s): [string, ForecastEntry] => [s.date, s])
      );

      const rows: RetrospectiveRow[] = [];
      for (let i = 0; i < actuals.length; i++) {
        const actual = actuals[i];
        const forecast = forecastMap.get(actual.date);
        if (!forecast) continue;

        // priorActual: previous period in EUR/ton (same scale as forecast for comparison)
        const priorActualRaw = i > 0 ? actuals[i - 1].value : null;
        const withinBounds =
          actual.value >= forecast.p10 && actual.value <= forecast.p90;

        // Issue F: compute real directional correctness in EUR/ton space
        const direction = computeDirection(
          forecast.p50,
          actual.value,
          priorActualRaw,
          withinBounds
        );

        // Store display values in EUR/kg
        const priorEurKg =
          priorActualRaw !== null
            ? priorActualRaw > 100
              ? eurTonToKg(priorActualRaw)
              : priorActualRaw
            : null;
        const actualEurKg =
          actual.value > 100 ? eurTonToKg(actual.value) : actual.value;
        const predictedEurKg = eurTonToKg(forecast.p50);

        rows.push({
          countryCode: market,
          predictedValue: predictedEurKg,
          actualValue: actualEurKg,
          priorActual: priorEurKg,
          withinBounds,
          direction,
        });
      }

      return rows;
    })
  );

  const allRows: RetrospectiveRow[] = [];
  for (const r of settled) {
    if (r.status === "fulfilled") allRows.push(...r.value);
  }
  return allRows;
}

// ─── Vegetable oil complex section (Issue L: no commodity.date access) ────────
export function buildVegOilComplexHtml(commodities: VegOilCommodity[]): string {
  if (commodities.length === 0) return "";

  const rows = commodities
    .map((c) => {
      const price =
        c.priceEurKg !== null ? `${c.priceEurKg.toFixed(2)} EUR/kg` : "—";
      const changeSign =
        c.changeWeekPct !== null && c.changeWeekPct >= 0 ? "+" : "";
      const change =
        c.changeWeekPct !== null
          ? `${changeSign}${c.changeWeekPct.toFixed(1)}%`
          : "—";
      return `<tr><td>${c.name}</td><td>${price}</td><td>${change}</td></tr>`;
    })
    .join("\n    ");

  return `
  <h2>Σύμπλεγμα Φυτικών Ελαίων</h2>
  <table>
    <thead>
      <tr><th>Εμπόρευμα</th><th>Τιμή</th><th>Εβδ. Μεταβολή</th></tr>
    </thead>
    <tbody>
    ${rows}
    </tbody>
  </table>`;
}

// ─── HTML report builder ──────────────────────────────────────────────────────
function buildReportHtml(snapshot: LiveSnapshot, isPreview: boolean): string {
  const dateStr = new Date().toLocaleDateString("el-GR", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const accuracyRecords = deriveAccuracyRecords(snapshot.accuracyScorecard);

  const forecastRows = snapshot.forecastMatrix
    .map(
      (row) => `
      <tr>
        <td>${MARKET_LABELS[row.market] ?? row.market}</td>
        <td>${row.currentPriceEurKg !== null ? row.currentPriceEurKg.toFixed(2) : "—"} EUR/kg</td>
        <td>${row.forecastP50EurKg !== null ? row.forecastP50EurKg.toFixed(2) : "—"} EUR/kg</td>
        <td>${
          row.forecastP10EurKg !== null && row.forecastP90EurKg !== null
            ? `[${row.forecastP10EurKg.toFixed(2)}, ${row.forecastP90EurKg.toFixed(2)}]`
            : "—"
        }</td>
        <td>${TREND_SYMBOL[row.trend] ?? "→"}</td>
        <td>${row.modelId}</td>
      </tr>`
    )
    .join("");

  const retroRows = snapshot.retrospective
    .map((row) => {
      const dirClass =
        row.direction === "correct"
          ? "dir-correct"
          : row.direction === "wrong"
          ? "dir-wrong"
          : "dir-unknown";
      const dirSymbol =
        row.direction === "correct" ? "✓" : row.direction === "wrong" ? "✗" : "–";
      const boundsSymbol = row.withinBounds ? "✓" : "✗";
      return `
      <tr>
        <td>${MARKET_LABELS[row.countryCode] ?? row.countryCode}</td>
        <td>${row.predictedValue.toFixed(2)}</td>
        <td>${row.actualValue.toFixed(2)}</td>
        <td class="${dirClass}">${dirSymbol}</td>
        <td>${boundsSymbol}</td>
      </tr>`;
    })
    .join("");

  const accuracyRows = accuracyRecords
    .map(
      (r) => `
      <tr>
        <td>${MARKET_LABELS[r.countryCode] ?? r.countryCode}</td>
        <td>${r.mape.toFixed(1)}%</td>
        <td>${(r.hitRate * 100).toFixed(0)}%</td>
        <td>${r.n}</td>
      </tr>`
    )
    .join("");

  const pulseHtml = sanitizeNarrativeHtml(snapshot.marketPulse.narrativeHtml);
  const volHtml =
    snapshot.garchVolatility !== null
      ? `<p>Τεκμαρτή Μεταβλητότητα (GARCH): <strong>${(
          snapshot.garchVolatility * 100
        ).toFixed(1)}%</strong></p>`
      : "";

  const previewBanner = isPreview
    ? `<div class="preview-banner">⚠️ ΠΡΟΕΠΙΣΚΟΠΗΣΗ — αυτή η αναφορά δεν έχει αποσταλεί</div>`
    : "";

  const accuracySection =
    accuracyRecords.length > 0
      ? `
  <h2>Ακρίβεια Προβλέψεων</h2>
  <table>
    <thead>
      <tr>
        <th>Αγορά</th><th>MAPE</th><th>Hit Rate</th><th>Σύνολο</th>
      </tr>
    </thead>
    <tbody>${accuracyRows}</tbody>
  </table>`
      : "";

  return `<!DOCTYPE html>
<html lang="el">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OLEA Intelligence — Εβδομαδιαία Έκθεση</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: Arial, Helvetica, sans-serif; font-size: 13px; color: #222; background: #fff; padding: 24px; }
    h1 { font-size: 20px; color: #2c5f2e; margin-bottom: 4px; }
    h2 { font-size: 13px; color: #2c5f2e; margin: 20px 0 8px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #e0e0e0; padding-bottom: 4px; }
    .meta { font-size: 11px; color: #666; margin-bottom: 20px; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 12px; }
    th { background: #2c5f2e; color: #fff; padding: 6px 10px; text-align: left; font-size: 11px; font-weight: 600; }
    td { padding: 5px 10px; border-bottom: 1px solid #eee; }
    tr:hover td { background: #f7faf7; }
    .dir-correct { color: #28a745; font-weight: bold; }
    .dir-wrong { color: #dc3545; font-weight: bold; }
    .dir-unknown { color: #6c757d; }
    .pulse { background: #f8faf8; border: 1px solid #d4e6d4; border-radius: 4px; padding: 12px 16px; margin-bottom: 8px; }
    .pulse p { line-height: 1.5; }
    .preview-banner { background: #fff3cd; padding: 8px 16px; border-left: 4px solid #ffc107; margin-bottom: 20px; font-size: 12px; border-radius: 2px; }
    .footer { font-size: 10px; color: #aaa; margin-top: 24px; border-top: 1px solid #eee; padding-top: 8px; }
  </style>
</head>
<body>
  ${previewBanner}
  <h1>OLEA Intelligence</h1>
  <p class="meta">Εβδομαδιαία Έκθεση Αγοράς Ελαιόλαδου &mdash; ${dateStr}</p>

  <h2>Παλμός Αγοράς</h2>
  <div class="pulse">
    ${pulseHtml || "<p>Δεν υπάρχουν διαθέσιμα δεδομένα παλμού.</p>"}
    ${volHtml}
  </div>

  <h2>Πίνακας Προβλέψεων (${REPORT_HORIZON})</h2>
  <table>
    <thead>
      <tr>
        <th>Αγορά</th>
        <th>Τρέχουσα Τιμή</th>
        <th>Πρόβλεψη P50</th>
        <th>Διάστημα [P10, P90]</th>
        <th>Τάση</th>
        <th>Μοντέλο</th>
      </tr>
    </thead>
    <tbody>
      ${forecastRows || "<tr><td colspan='6' style='text-align:center;color:#888'>Δεν υπάρχουν διαθέσιμα δεδομένα</td></tr>"}
    </tbody>
  </table>

  <h2>Αναδρομική Αξιολόγηση</h2>
  <table>
    <thead>
      <tr>
        <th>Αγορά</th>
        <th>Πρόβλεψη (EUR/kg)</th>
        <th>Πραγματική (EUR/kg)</th>
        <th>Κατεύθυνση</th>
        <th>Εντός Ορίων</th>
      </tr>
    </thead>
    <tbody>
      ${retroRows || "<tr><td colspan='5' style='text-align:center;color:#888'>Δεν υπάρχουν δεδομένα αξιολόγησης</td></tr>"}
    </tbody>
  </table>

  ${accuracySection}

  <div class="footer">
    <p>OLEA Intelligence Platform &mdash; Εμπιστευτικό</p>
    <p>Δημιουργήθηκε: ${new Date().toISOString()}</p>
  </div>
</body>
</html>`;
}

// ─── Snapshot gatherer (read-only, no side effects) ──────────────────────────
async function gatherLiveSnapshot(): Promise<LiveSnapshot> {
  // Three independent async operations run in parallel
  const [forecastMatrix, retrospective, garchVolatility] = await Promise.all([
    buildForecastMatrix(),
    buildRetrospective(),
    fetchGarchVolatility(),
  ]);

  const accuracyScorecard = deriveAccuracyScorecard();

  const partial = {
    generatedAt: new Date().toISOString(),
    forecastMatrix,
    retrospective,
    accuracyScorecard,
    garchVolatility,
  };

  const marketPulse = buildMarketPulse(partial);

  return { ...partial, marketPulse };
}

// ─── Issue B: Preview path — read-only, no writes, no delivery logging ────────
/**
 * Generates a preview HTML report. Safe to call repeatedly.
 * Does NOT write to disk, does NOT log delivery, does NOT freeze snapshot.
 */
export async function generatePreviewReport(): Promise<string> {
  const snapshot = await gatherLiveSnapshot();
  return buildReportHtml(snapshot, true);
}

// ─── Issue B: Publish path — full pipeline with side effects ─────────────────
/**
 * Generates the full weekly report, persists a snapshot JSON, and logs delivery.
 * Use only for actual send operations — not for preview.
 */
export async function generatePublishReport(): Promise<PublishResult> {
  const snapshot = await gatherLiveSnapshot();
  const html = buildReportHtml(snapshot, false);

  // Persist weekly snapshot (publish path only)
  const weeklyPath = join(
    REPORTS_DIR,
    `weekly-${new Date().toISOString().slice(0, 10)}.json`
  );
  writeJson(weeklyPath, {
    generatedAt: snapshot.generatedAt,
    forecastMatrix: snapshot.forecastMatrix,
    accuracyScorecard: snapshot.accuracyScorecard,
  });

  const sentAt = new Date().toISOString();
  console.log(`[email-reporter] Report published at ${sentAt}`);

  return { html, snapshot, sentAt };
}

// ─── Issue A: Real PDF via Playwright ─────────────────────────────────────────
/**
 * Renders `html` through headless Chromium and returns a real PDF Buffer.
 * Dynamic import is intentional: Playwright is only loaded when PDF is needed.
 */
export async function generatePdfReport(html: string): Promise<Buffer> {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const { chromium } = await import("playwright");
  const browser = await chromium.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });
  try {
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: "networkidle" });
    const pdfUint8 = await page.pdf({
      format: "A4",
      printBackground: true,
      margin: { top: "15mm", bottom: "15mm", left: "12mm", right: "12mm" },
    });
    return Buffer.from(pdfUint8);
  } finally {
    await browser.close();
  }
}

// ─── Issue H: HMAC feedback signing (FEEDBACK_SECRET only) ───────────────────
export function signFeedbackToken(payload: string): string {
  return createHmac("sha256", getFeedbackSecret()).update(payload).digest("hex");
}

export function verifyFeedbackToken(payload: string, token: string): boolean {
  const expected = signFeedbackToken(payload);
  if (expected.length !== token.length) return false;
  // Constant-time comparison
  let diff = 0;
  for (let i = 0; i < expected.length; i++) {
    diff |= expected.charCodeAt(i) ^ token.charCodeAt(i);
  }
  return diff === 0;
}
