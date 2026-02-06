# OleaForecast

Forecasting & decision-support system for olive oil markets.

## What this repo is
OleaForecast is a production-minded forecasting application that:
- ingests market data (spot/weekly/monthly time series)
- runs multiple forecasting experts
- applies serving gates (availability / bias / MASE / interval coverage)
- selects a champion forecast and exposes outputs for dashboards/API

## Repo principles
- No secrets in git (use `.env` locally; commit only `.env.example`)
- Deterministic builds (pin deps; CI runs on every push)
- Guardrails first (gates + audit trail are non-negotiable)

## Quick start (conceptual)
- Configure environment variables (see `.env.example`)
- Install dependencies
- Run dev server

## Web app (rebuilt)
- Install: `npm install`
- Run: `npm run dev`
- Open: `http://localhost:3000`

## Streamlit UI (local)
- Install: `pip install -r requirements.txt`
- Run: `streamlit run streamlit_app.py`

## Streamlit Cloud deploy
- App entry: `streamlit_app.py`
- Python deps: `requirements.txt`
- Secrets: add key/value pairs in Streamlit Cloud (see `.streamlit/secrets.toml.example`)
- Update data: `npm run -s eval:batch -- data` (regenerates `reports/winners.json`)
- Scheduled refresh: GitHub Actions workflow `refresh-data` updates `reports/winners.json` and `reports/run-meta.json`

## Structure (high level)
- `server/` backend (gates, scoring, APIs)
- `ml/`     forecasting engines / utilities
- `client/` (if present) UI / dashboard

## Notes
This repository is intentionally strict about:
- model eligibility (hard-fail gates)
- confidence / interval integrity
- auditability (why a model won)
