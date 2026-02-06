import { readFileSync, mkdirSync, appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { SourcesCatalog, CatalogSource, ObservationRaw } from "./sources/types";
import { toYmd, isoNow } from "./sources/http";
import { ingestInfaoliva } from "./sources/parse_infaoliva";
import { ingestCentrolivaPoolred } from "./sources/parse_centroliva_poolred";
import { ingestSedik } from "./sources/parse_sedik";

function ensureDir(path: string) {
  mkdirSync(path, { recursive: true });
}

function loadCatalog(path: string): SourcesCatalog {
  return JSON.parse(readFileSync(path, "utf8"));
}

async function ingestOne(src: CatalogSource): Promise<ObservationRaw[]> {
  if (src.parser === "infaoliva") return ingestInfaoliva(src);
  if (src.parser === "centroliva_poolred") return ingestCentrolivaPoolred(src);
  if (src.parser === "sedik") return ingestSedik(src);
  return [];
}

function writeNdjson(path: string, rows: ObservationRaw[]) {
  ensureDir(dirname(path));
  const lines = rows.map(r => JSON.stringify(r)).join("\n") + "\n";
  appendFileSync(path, lines, "utf8");
}

async function main() {
  const catalogPath = process.argv[2] || "server/ingestion/catalog/sources.catalog.json";
  const cat = loadCatalog(catalogPath);

  const today = toYmd(new Date());
  const fetchedAt = isoNow();

  console.log(`[ingest] catalog=${catalogPath} fetchedAt=${fetchedAt}`);

  for (const src of cat.sources) {
    if (src.type === "stub") {
      console.log(`[ingest] skip stub ${src.sourceId}`);
      continue;
    }

    console.log(`[ingest] source=${src.sourceId} url=${src.url}`);
    const rows = await ingestOne(src);

    const outPath = `data/observations/raw/${src.sourceId}/${today}.ndjson`;
    writeNdjson(outPath, rows);

    console.log(`[ingest] wrote ${rows.length} rows -> ${outPath}`);
  }
}

main().catch((e) => {
  console.error("[ingest] ERROR:", e?.stack || e);
  process.exit(1);
});
