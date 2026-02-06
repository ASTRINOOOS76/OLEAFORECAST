import * as cheerio from "cheerio";
import { CatalogSource, ObservationRaw } from "./types";
import { fetchText, isoNow, eurTtoEurKg, toYmd } from "./http";

function parseEuroNumber(s: string): number | null {
  // "4.233,19" or "4233,19" or "4233.19"
  const cleaned = s.replace(/[^\d.,-]/g, "").trim();
  if (!cleaned) return null;
  const normalized = cleaned.includes(",") ? cleaned.replace(/\./g, "").replace(",", ".") : cleaned;
  const v = Number(normalized);
  return Number.isFinite(v) ? v : null;
}

export async function ingestCentrolivaPoolred(src: CatalogSource): Promise<ObservationRaw[]> {
  const html = await fetchText(src.url);
  const $ = cheerio.load(html);

  // Extract week period "Desde dd/mm/yy ... Hasta dd/mm/yy" if present; else today
  const text = $("body").text();
  const m = text.match(/Hasta\s+(\d{2}\/\d{2}\/\d{2,4})/i);
  const dateYmd = m ? (() => {
    const parts = m[1].split("/");
    const dd = parts[0], mm = parts[1];
    const yy = parts[2].length === 2 ? `20${parts[2]}` : parts[2];
    return `${yy}-${mm}-${dd}`;
  })() : toYmd(new Date());

  const out: ObservationRaw[] = [];

  // Look for rows with "Aceite de Oliva Virgen Extra", "Virgen", "Lampante"
  $("tr").each((_i, tr) => {
    const cells = $(tr).find("td,th").toArray().map(x => $(x).text().replace(/\s+/g, " ").trim());
    if (cells.length < 3) return;

    const label = cells[0].toLowerCase();
    const priceCell = cells.find(c => c.includes("EUR") || /[\d]/.test(c)) ?? cells[cells.length - 2] ?? "";
    const vEurT = parseEuroNumber(priceCell);
    if (vEurT === null) return;

    const push = (grade: string, match: RegExp) => {
      if (!match.test(label)) return;
      out.push({
        sourceId: src.sourceId,
        market: src.market,
        grade,
        date: dateYmd,
        value: eurTtoEurKg(vEurT),   // normalize to EUR/kg already
        unit: "EUR_KG",
        currency: "EUR",
        region: "ES",
        url: src.url,
        fetchedAt: isoNow(),
        meta: { eur_t: vEurT, rawRow: cells }
      });
    };

    push("EVOO", /virgen extra/);
    push("VOO", /^aceite de oliva virgen\b(?! extra)/);
    push("LAMPANTE", /lampante/);
  });

  // Dedup per grade
  const byKey = new Map<string, ObservationRaw>();
  for (const o of out) byKey.set(`${o.grade}`, o);

  return Array.from(byKey.values());
}
