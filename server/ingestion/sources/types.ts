export type ObservationRaw = {
  sourceId: string;
  market: string;          // ES/GR/IT/...
  grade: string;           // EVOO/VOO/LAMPANTE
  date: string;            // YYYY-MM-DD (as published or inferred)
  value: number;
  unit: "EUR_KG" | "EUR_T";
  currency: "EUR";
  region?: string | null;
  url: string;
  fetchedAt: string;       // ISO
  meta?: Record<string, any>;
};

export type CatalogSource = {
  sourceId: string;
  market: string;
  type: "html_table" | "api" | "pdf" | "stub";
  url: string;
  parser: "infaoliva" | "centroliva_poolred" | "sedik" | "stub";
  frequency: "daily" | "weekly" | "event" | "unknown";
  outputs: Array<{ grade: string; unit: "EUR_KG" | "EUR_T"; field: string }>;
  notes?: string;
};

export type SourcesCatalog = {
  version: number;
  sources: CatalogSource[];
};
