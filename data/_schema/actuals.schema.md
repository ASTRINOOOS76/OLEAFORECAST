# Actuals CSV schema

Path:
  data/<MARKET>/<HORIZON>/actuals.csv

Format (comma-separated, header required):
  date,value

Examples:
  2026-01-05,4210
  2026-01-12,4195

Notes:
- `date` must be ISO-like sortable string (YYYY-MM-DD recommended).
- `value` numeric (EUR/ton).
