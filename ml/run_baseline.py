import json
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np

from forecast_baseline import forecast_with_intervals


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_payload(model_id: str, market: str, horizon: str, dates: List[str], y: np.ndarray, steps: int) -> Dict[str, Any]:
    fc = forecast_with_intervals(y=y, steps=steps, alpha=0.2)

    # Build forward dates (naive: reuse last date cadence if possible; otherwise just index)
    # For now: if you pass explicit future dates, use them; else create placeholders.
    series = []
    for i in range(steps):
        date = dates[i] if i < len(dates) else f"t+{i+1}"
        series.append({
            "date": date,
            "p10": float(fc["p10"][i]),
            "p50": float(fc["p50"][i]),
            "p90": float(fc["p90"][i]),
        })

    return {
        "modelId": model_id,
        "market": market,
        "horizon": horizon,
        "generatedAt": _iso_now(),
        "series": series,
        "diagnostics": {
            "method": "naive_last_value + empirical_residual_intervals",
            "alpha": 0.2,
            "steps": steps
        }
    }


def main():
    """
    Usage:
      python ml/run_baseline.py <modelId> <market> <horizon> <steps>

    Reads JSON from stdin:
      { "y": [..], "future_dates": ["2026-02-01", ...] }

    Writes ModelForecastPayload JSON to stdout.
    """
    if len(sys.argv) < 5:
        print("Usage: python ml/run_baseline.py <modelId> <market> <horizon> <steps>", file=sys.stderr)
        sys.exit(2)

    model_id = sys.argv[1]
    market = sys.argv[2]
    horizon = sys.argv[3]
    steps = int(sys.argv[4])

    raw = sys.stdin.read().strip()
    if not raw:
        print("Expected JSON input on stdin", file=sys.stderr)
        sys.exit(2)

    data = json.loads(raw)
    y = np.asarray(data.get("y", []), dtype=float)
    future_dates = data.get("future_dates", [])

    payload = make_payload(model_id=model_id, market=market, horizon=horizon, dates=future_dates, y=y, steps=steps)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
