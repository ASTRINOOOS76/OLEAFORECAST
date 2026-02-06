#!/usr/bin/env python3
"""Regime-Aware ETS forecasting for OleaForecast.

Improvements over standard ETS:
1. Regime detection: switches damping off during volatile periods
2. Momentum tracking: follows recent price direction
3. Ensemble: combines ETS with trend extrapolation
"""

import sys
import json
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')


def detect_regime(prices: np.ndarray, window: int = 12) -> dict:
    """Detect market regime based on recent volatility and trend."""
    if len(prices) < window:
        return {"regime": "stable", "volatility": 0.0, "momentum": 0.0}

    recent = prices[-window:]
    mean_price = np.mean(recent)

    volatility = np.std(recent) / mean_price if mean_price > 0 else 0.0

    if len(prices) >= 4:
        momentum = (prices[-1] - prices[-4]) / prices[-4] if prices[-4] > 0 else 0.0
    else:
        momentum = 0.0

    if volatility > 0.08:
        regime = "volatile"
    elif abs(momentum) > 0.05:
        regime = "trending"
    else:
        regime = "stable"

    return {
        "regime": regime,
        "volatility": round(volatility, 4),
        "momentum": round(momentum, 4),
        "mean_price": round(mean_price, 2)
    }


def forecast_regime_aware_ets(prices: list, horizon: int) -> dict:
    """Run regime-aware ETS forecast."""
    if len(prices) < 10:
        return {"error": "Need at least 10 data points"}

    y = np.array(prices, dtype=float)
    regime_info = detect_regime(y)
    regime = regime_info["regime"]
    momentum = regime_info["momentum"]
    volatility = regime_info["volatility"]

    use_damping = regime == "stable"

    try:
        model = ExponentialSmoothing(
            y,
            seasonal_periods=min(12, len(prices) // 2),
            trend='add',
            seasonal='add' if len(prices) >= 24 else None,
            damped_trend=use_damping
        )
        fitted = model.fit(optimized=True)

        ets_forecast = fitted.forecast(horizon)
        residuals = y - fitted.fittedvalues
        std_error = float(np.std(residuals))

        if regime in ["volatile", "trending"] and abs(momentum) > 0.02:
            weekly_change = (y[-1] - y[-4]) / 4 if len(y) >= 4 else 0
            trend_adjustment = weekly_change * np.arange(1, horizon + 1)
            trend_decay = np.exp(-0.1 * np.arange(0, horizon))
            trend_adjustment = trend_adjustment * trend_decay
        else:
            trend_adjustment = np.zeros(horizon)

        if regime == "stable":
            ets_weight = 0.9
            trend_weight = 0.1
        elif regime == "trending":
            ets_weight = 0.6
            trend_weight = 0.4
        else:
            ets_weight = 0.7
            trend_weight = 0.3

        final_forecast = ets_weight * ets_forecast + trend_weight * (ets_forecast + trend_adjustment)

        if regime == "volatile":
            uncertainty_mult = 1.5
        elif regime == "trending":
            uncertainty_mult = 1.2
        else:
            uncertainty_mult = 1.0

        predictions = []
        for h in range(horizon):
            h_factor = np.sqrt(h + 1) * uncertainty_mult
            pred = float(final_forecast[h])
            width = 1.28 * std_error * h_factor

            predictions.append({
                "horizon": h + 1,
                "p10": round(pred - width, 2),
                "p50": round(pred, 2),
                "p90": round(pred + width, 2),
                "ets_raw": round(float(ets_forecast[h]), 2),
                "trend_adj": round(float(trend_adjustment[h]) if h < len(trend_adjustment) else 0, 2)
            })

        return {
            "model": "Regime-Aware ETS",
            "regime": regime_info,
            "use_damping": use_damping,
            "weights": {"ets": ets_weight, "trend": trend_weight},
            "predictions": predictions,
            "std_error": round(std_error, 2),
            "aic": round(float(fitted.aic), 2) if hasattr(fitted, 'aic') else None
        }

    except Exception as e:
        return {"error": f"Regime-Aware ETS failed: {str(e)}"}


if __name__ == "__main__":
    try:
        input_data = json.loads(sys.argv[1]) if len(sys.argv) > 1 else json.load(sys.stdin)
        prices = input_data.get("prices", [])
        horizon = input_data.get("horizon", 12)

        result = forecast_regime_aware_ets(prices, horizon)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
