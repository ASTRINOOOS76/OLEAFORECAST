import * as cheerio from "cheerio";
import { CatalogSource, ObservationRaw } from "./types";
import { fetchText, isoNow, toYmd } from "./http";

function parseEuroNumber(s: string): number | null {
  // "4.175 EUR" or "3,967 EUR"
  const cleaned = s.replace(/[^\d.,-]/g, "").trim();
  if (!cleaned) return null;
  // try comma as decimal
  const c = cleaned.includes(",") && !cleaned.includes(".");
  const normalized = c ? cleaned.replace(/\./g, "").replace(",", ".") : cleaned.replace(/,/g, "");
  const v = Number(normalized);
  return Number.isFinite(v) ? v : null;
}

export async function ingestInfaoliva(src: CatalogSource): Promise<ObservationRaw[]> {
  const html = await fetchText(src.url);
  const $ = cheerio.load(html);

  // Find a date like "05/02/2026" near the observatorio section
  const bodyText = $("body").text();
  const m = bodyText.match(/(\d{2}\/\d{2}\/\d{4})/);
  const dateYmd = m ? (() => {
    const [dd, mm, yyyy] = m[1].split("/");
    return `${yyyy}-${mm}-${dd}`;
  })() : toYmd(new Date());

  const out: ObservationRaw[] = [];

  // Find rows that include categories
  // We do a robust scan: any table row containing "AOVE" or "Virgen" or "lampante"
  $("tr").each((_i, tr) => {
    const tds = $(tr).find("td,th").toArray().map(x => $(x).text().replace(/\s+/g, " ").trim());
    if (tds.length < 2) return;
    const row = tds.join(" | ").toLowerCase();

    const priceCell = tds[tds.length - 1];
    const v = parseEuroNumber(priceCell);
    if (v === null) return;

    const mk = (grade: string, fieldMatch: RegExp) => {
      if (!fieldMatch.test(row)) return;
      out.push({
        sourceId: src.sourceId,
        market: src.market,
        grade,
        date: dateYmd,
        value: v,
        unit: "EUR_KG",
        currency: "EUR",
        region: "ES",
        url: src.url,
        fetchedAt: isoNow(),
        meta: { rawRow: tds }
      });
    };

    mk("EVOO", /aove|virgen extra/);
    mk("VOO", /aceite de oliva virgen(?! extra)/);
    mk("LAMPANTE", /lampante/);
  });

  // Dedup per grade
  const byKey = new Map<string, ObservationRaw>();
  for (const o of out) byKey.set(`${o.grade}`, o);

  return Array.from(byKey.values());
}
