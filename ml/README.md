# ML (OleaForecast)

This folder contains forecasting utilities and baseline models.

## Goals
- Deterministic, testable forecasting functions
- Clear separation: data prep -> model -> intervals -> evaluation
- No "magic drift" hacks in production paths

## Running locally (conceptual)
- Create venv
- pip install -r requirements.txt
- Run scripts under `ml/`
## Baseline runner (JSON output)
Example:

```bash
echo '{"y":[4200,4210,4190,4220,4230,4240,4250,4260], "future_dates":["t+1","t+2","t+3","t+4"]}' \
| python ml/run_baseline.py E0 ES W4 4
