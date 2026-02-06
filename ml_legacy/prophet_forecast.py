#!/usr/bin/env python3
"""Simple ETS forecasting for OleaForecast (using statsmodels instead of Prophet)."""

import sys
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')


def forecast_ets(prices: list, horizon: int) -> dict:
    """Run ETS (Holt-Winters) forecast."""
    if len(prices) < 10:
        return {"error": "Need at least 10 data points"}

    y = np.array(prices, dtype=float)

    try:
        model = ExponentialSmoothing(
            y,
            seasonal_periods=min(12, len(prices) // 2),
            trend='add',
            seasonal='add' if len(prices) >= 24 else None,
            damped_trend=True
        )
        fitted = model.fit(optimized=True)

        forecast_values = fitted.forecast(horizon)
        residuals = y - fitted.fittedvalues
        std_error = float(np.std(residuals))

        predictions = []
        for h in range(horizon):
            h_factor = np.sqrt(h + 1)
            pred = float(forecast_values[h])
            predictions.append({
                "horizon": h + 1,
                "p10": round(pred - 1.28 * std_error * h_factor, 2),
                "p50": round(pred, 2),
                "p90": round(pred + 1.28 * std_error * h_factor, 2)
            })

        return {
            "model": "ETS (Holt-Winters)",
            "predictions": predictions,
            "std_error": round(std_error, 2),
            "aic": round(float(fitted.aic), 2) if hasattr(fitted, 'aic') else None
        }
    except Exception as e:
        return {"error": f"ETS failed: {str(e)}"}


if __name__ == "__main__":
    try:
        input_data = json.loads(sys.argv[1]) if len(sys.argv) > 1 else json.load(sys.stdin)
        prices = input_data.get("prices", [])
        horizon = input_data.get("horizon", 12)

        result = forecast_ets(prices, horizon)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
