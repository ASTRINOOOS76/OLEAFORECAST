import * as cheerio from "cheerio";
import { CatalogSource, ObservationRaw } from "./types";
import { fetchText, isoNow, toYmd } from "./http";

function parseGreekNumber(s: string): number | null {
  // "5,00" -> 5.00
  const cleaned = s.replace(/[^\d.,-]/g, "").trim();
  if (!cleaned) return null;
  const normalized = cleaned.replace(/\./g, "").replace(",", ".");
  const v = Number(normalized);
  return Number.isFinite(v) ? v : null;
}

export async function ingestSedik(src: CatalogSource): Promise<ObservationRaw[]> {
  const html = await fetchText(src.url);
  const $ = cheerio.load(html);

  const rows: ObservationRaw[] = [];

  // Table has columns including date of auction and "Timh katakyrosis"
  $("tr").each((_i, tr) => {
    const tds = $(tr).find("td").toArray().map(x => $(x).text().replace(/\s+/g, " ").trim());
    if (tds.length < 6) return;

    const dateCell = tds[1]; // "30/12/2025"
    const priceCell = tds[5]; // "5,00" or "Agonos"
    if (!dateCell || !/\d{2}\/\d{2}\/\d{4}/.test(dateCell)) return;
    if (/agono[s]?/i.test(priceCell)) return;

    const v = parseGreekNumber(priceCell);
    if (v === null) return;

    const [dd, mm, yyyy] = dateCell.split("/");
    const ymd = `${yyyy}-${mm}-${dd}`;

    // This is usually EUR/kg in Greek tenders (often stated as EUR/kg). If uncertain, mark meta.
    rows.push({
      sourceId: src.sourceId,
      market: src.market,
      grade: "EVOO",
      date: ymd,
      value: v,
      unit: "EUR_KG",
      currency: "EUR",
      region: "GR",
      url: src.url,
      fetchedAt: isoNow(),
      meta: { kind: "auction", rawRow: tds }
    });
  });

  // Keep only latest 1 event (or you can keep all; we keep all and dedupe on (date,value))
  const seen = new Set<string>();
  const out: ObservationRaw[] = [];
  for (const r of rows) {
    const k = `${r.date}|${r.value}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(r);
  }
  if (out.length) return out;
  const fallback: ObservationRaw = {
    sourceId: src.sourceId,
    market: src.market,
    grade: "EVOO",
    date: toYmd(new Date()),
    value: NaN,
    unit: "EUR_KG",
    currency: "EUR",
    region: "GR",
    url: src.url,
    fetchedAt: isoNow(),
    meta: { note: "No parseable rows" }
  };
  return [fallback].filter(o => Number.isFinite(o.value));
}
