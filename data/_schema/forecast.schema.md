# Forecast payload JSON schema

Path:
  data/<MARKET>/<HORIZON>/<MODEL>.json

Required keys:
- modelId (string)
- market (string)
- horizon (string)
- generatedAt (ISO datetime string)
- series: [{ date, p10, p50, p90 }...]

Rules:
- series must be non-empty
- p10 <= p50 <= p90 for each point
