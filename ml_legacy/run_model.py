#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_actuals_csv(path: str) -> List[Dict[str, Any]]:
    # expects header: date,value
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) < 2:
        return out
    hdr = lines[0].lower()
    if hdr != "date,value":
        raise ValueError(f"Bad actuals header: {lines[0]} (expected date,value)")
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 2:
            continue
        d, v = parts
        try:
            val = float(v)
        except Exception:
            continue
        out.append({"date": d, "value": val})
    return out


def naive_last_value_forecast(
    actuals: List[Dict[str, Any]],
    horizon_days: int = 28,
) -> List[Dict[str, Any]]:
    """
    Baseline: p50=last actual, band expands with time.
    Output dates: same cadence as actuals (if weekly) OR daily (if dense).
    For now: create 4 weekly points (W4) starting from last date + 7d steps.
    """
    if not actuals:
        return []

    # take last value
    last = actuals[-1]
    last_value = float(last["value"])
    last_date = last["date"]

    def parse_date(value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                return None

    last_dt = parse_date(last_date)

    # produce 4 weekly dates (W4)
    # In production, TS will pass explicit target dates to python.
    series: List[Dict[str, Any]] = []
    for k in range(4):
        if last_dt:
            date_out = (last_dt + timedelta(days=7 * (k + 1))).date().isoformat()
        else:
            date_out = f"{last_date}_p{k + 1}"
        sigma = 120 + 40 * (k ** 0.5)  # EUR/ton-ish scale; ok after normalization upstream
        p50 = last_value
        p10 = p50 - 1.28155 * sigma
        p90 = p50 + 1.28155 * sigma
        series.append({"date": date_out, "p10": p10, "p50": p50, "p90": p90})
    return series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--horizon", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--actuals", required=True, help="Path to actuals.csv (date,value)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument(
        "--engine",
        required=False,
        default="naive",
        choices=["naive", "xgb", "prophet", "ets"],
    )
    args = ap.parse_args()

    market = args.market.strip()
    horizon = args.horizon.strip()
    model_id = args.model_id.strip()
    actuals_path = args.actuals
    out_path = args.out

    if not os.path.exists(actuals_path):
        raise FileNotFoundError(f"actuals not found: {actuals_path}")

    actuals = read_actuals_csv(actuals_path)

    if args.engine == "naive":
        series = naive_last_value_forecast(actuals)
    else:
        # v1: keep it simple; wire legacy modules in next commit
        # You will replace this block to call predict_xgb/prophet_forecast/regime_aware_ets
        series = naive_last_value_forecast(actuals)

    payload = {
        "modelId": model_id,
        "market": market,
        "horizon": horizon,
        "generatedAt": iso_now(),
        "series": series,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
