import numpy as np

def naive_last_value_forecast(y: np.ndarray, steps: int) -> np.ndarray:
    """Baseline: forecast equals last observed value."""
    if y.size < 2:
        raise ValueError("Need at least 2 observations")
    last = float(y[-1])
    return np.full((steps,), last, dtype=float)

def empirical_intervals(residuals: np.ndarray, alpha: float = 0.2) -> tuple[float, float]:
    """
    Empirical central (1-alpha) interval from residual distribution.
    For p10/p90 -> alpha=0.2 yields 10th and 90th percentiles.
    """
    if residuals.size < 20:
        # small sample: fallback to normal-ish
        s = float(np.std(residuals, ddof=1)) if residuals.size > 1 else 0.0
        return (-1.2816 * s, 1.2816 * s)
    lo = float(np.quantile(residuals, alpha/2))
    hi = float(np.quantile(residuals, 1 - alpha/2))
    return lo, hi

def forecast_with_intervals(y: np.ndarray, steps: int, alpha: float = 0.2) -> dict:
    """
    Returns p10/p50/p90 arrays using naive baseline + empirical residual intervals.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 8:
        raise ValueError("Need at least 8 observations for residual-based intervals")

    # one-step residuals for naive model
    residuals = y[1:] - y[:-1]
    lo_r, hi_r = empirical_intervals(residuals, alpha=alpha)

    p50 = naive_last_value_forecast(y, steps)
    # Horizon scaling: sqrt(h) is a heuristic; later we calibrate per-horizon.
    hs = np.sqrt(np.arange(1, steps + 1, dtype=float))
    p10 = p50 + lo_r * hs
    p90 = p50 + hi_r * hs

    return {"p10": p10, "p50": p50, "p90": p90}
