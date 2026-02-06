#!/usr/bin/env python3
import argparse
import json
import os
import math
import sys
from datetime import datetime, timezone, date
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

Z = 1.281551565545  # p10/p90 z-score (Normal)

# Ensure repo root is on sys.path when executed as a script.
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def horizon_to_weeks(h: str) -> int:
    h = h.upper().strip()
    if h.startswith("W"):
        return int(h[1:])
    raise ValueError(f"Unsupported horizon format: {h} (expected W4/W13/W26/W52)")


def parse_yyyy_mm_dd(s: str) -> date:
    y, m, d = s.split("-")
    return date(int(y), int(m), int(d))


def read_actuals_csv(path: str) -> List[Dict[str, Any]]:
    out = []
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
        except:
            continue
        out.append({"date": d, "value": val})
    out.sort(key=lambda x: x["date"])
    return out


def make_target_dates(actuals: List[Dict[str, Any]], horizon_weeks: int) -> List[str]:
    last_d = parse_yyyy_mm_dd(actuals[-1]["date"])
    out = []
    for k in range(1, horizon_weeks + 1):
        dd = last_d.toordinal() + 7 * k
        out.append(date.fromordinal(dd).isoformat())
    return out


def build_feature_dict_from_actuals(actuals: List[Dict[str, Any]]) -> Dict[str, float]:
    if not actuals:
        return {"baseValue": 4000.0}

    values = [float(a["value"]) for a in actuals]
    last = values[-1]
    feats: Dict[str, float] = {"baseValue": float(last)}

    def lag(k: int) -> Optional[float]:
        idx = len(values) - 1 - k
        if idx < 0:
            return None
        return float(values[idx])

    for k in [1,2,4,8,12,26,52]:
        v = lag(k)
        feats[f"lag_{k}w"] = float(v) if v is not None else 0.0

    def rolling_mean(w: int) -> float:
        if len(values) < w:
            return float(np.mean(values)) if values else 0.0
        return float(np.mean(values[-w:]))

    def rolling_std(w: int) -> float:
        if len(values) < w:
            return float(np.std(values)) if values else 0.0
        return float(np.std(values[-w:]))

    for w in [4,8,13,26]:
        feats[f"rolling_mean_{w}w"] = rolling_mean(w)
        feats[f"rolling_std_{w}w"]  = rolling_std(w)
        m = feats[f"rolling_mean_{w}w"]
        s = feats[f"rolling_std_{w}w"]
        feats[f"rolling_vol_{w}w"]  = float(s / m) if m and m != 0 else 0.0

    def ret(k: int) -> float:
        v0 = last
        vk = lag(k)
        if vk is None or vk == 0:
            return 0.0
        return float((v0 / vk) - 1.0)

    feats["ret_1w"]  = ret(1)
    feats["ret_4w"]  = ret(4)
    feats["ret_13w"] = ret(13)
    feats["ret_26w"] = ret(26)

    d = parse_yyyy_mm_dd(actuals[-1]["date"])
    week = int(d.isocalendar()[1])
    month = d.month

    feats["week_sin"] = float(math.sin(2 * math.pi * week / 52.0))
    feats["week_cos"] = float(math.cos(2 * math.pi * week / 52.0))
    feats["month_sin"] = float(math.sin(2 * math.pi * month / 12.0))
    feats["month_cos"] = float(math.cos(2 * math.pi * month / 12.0))

    crop_start = date(d.year if d.month >= 10 else d.year - 1, 10, 1)
    delta_days = (d - crop_start).days
    feats["crop_year_progress"] = float(max(0.0, min(1.0, delta_days / 365.0)))

    feats["is_harvest_season"] = 1.0 if month in [10,11,12,1,2] else 0.0
    return feats


def s3_download_artifact(series_key: str, horizon_weeks: int, dst_dir: str) -> Tuple[str, str]:
    import boto3
    from botocore.config import Config

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError("Missing env S3_BUCKET")

    prefix = os.environ.get("S3_PREFIX", "ml/artifacts").strip("/")
    endpoint = os.environ.get("S3_ENDPOINT")
    region = os.environ.get("AWS_REGION", "us-east-1")

    session = boto3.session.Session()
    s3 = session.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4"),
    )

    os.makedirs(dst_dir, exist_ok=True)

    base = f"{series_key}_h{horizon_weeks}w"
    key_pkl = f"{prefix}/{base}.pkl"
    key_meta = f"{prefix}/{base}_meta.json"

    pkl_path = os.path.join(dst_dir, f"{base}.pkl")
    meta_path = os.path.join(dst_dir, f"{base}_meta.json")

    s3.download_file(bucket, key_pkl, pkl_path)
    s3.download_file(bucket, key_meta, meta_path)

    return pkl_path, meta_path


def xgb_artifact_forecast(series_key: str, horizon_weeks: int, features: Dict[str, float]) -> Dict[str, Any]:
    from ml_legacy import predict_xgb

    cache_dir = os.path.join("/tmp", "oleaforecast_artifacts")
    _pkl_path, meta_path = s3_download_artifact(series_key, horizon_weeks, cache_dir)
    predict_xgb.ARTIFACTS_DIR = cache_dir

    out = predict_xgb.predict_all_horizons(series_key, features)
    preds = out.get("predictions", [])
    row = next((p for p in preds if int(p.get("horizon", -1)) == int(horizon_weeks)), None)
    if not row:
        raise RuntimeError(f"No prediction for horizon {horizon_weeks} in predict_xgb output")

    p50 = float(row.get("predictedPrice", features.get("baseValue", 0.0)))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    rmse = float(meta.get("metrics", {}).get("rmse", 0.03))
    sigma = rmse * math.sqrt(max(1.0, horizon_weeks / 4.0))
    p10 = p50 * math.exp(-Z * sigma)
    p90 = p50 * math.exp(+Z * sigma)

    return {"p10": p10, "p50": p50, "p90": p90, "meta": {"rmse": rmse, "sigma": sigma}}


def ets_forecast(prices: List[float], horizon_weeks: int, mode: str) -> List[Dict[str, Any]]:
    if mode == "prophet":
        from ml_legacy import prophet_forecast as pf
        res = pf.forecast_ets(prices, horizon_weeks)
        if "error" in res:
            raise RuntimeError(res["error"])
        return res["predictions"]
    if mode == "ets":
        from ml_legacy import regime_aware_ets as rae
        res = rae.forecast_regime_aware_ets(prices, horizon_weeks)
        if "error" in res:
            raise RuntimeError(res["error"])
        return res["predictions"]
    raise ValueError(f"Unknown ETS mode: {mode}")


def naive_forecast(actuals: List[Dict[str, Any]], target_dates: List[str], variant: str) -> List[Dict[str, Any]]:
    y = [float(a["value"]) for a in actuals]
    last = y[-1]
    variant = (variant or "last").lower()

    # baseline volatility proxy
    if len(y) >= 13:
        vol = float(np.std(y[-13:]))
    else:
        vol = float(np.std(y)) if len(y) > 2 else max(1.0, 0.02 * last)

    out = []

    if variant == "last":
        for i, d in enumerate(target_dates):
            w = max(1.0, vol) * math.sqrt(i + 1)
            out.append({"date": d, "p10": last - Z*w, "p50": last, "p90": last + Z*w})
        return out

    if variant == "drift":
        # drift from last 8 weeks (or available)
        n = min(8, len(y) - 1)
        slope = (y[-1] - y[-1-n]) / max(1, n)
        for i, d in enumerate(target_dates):
            p50 = last + slope * (i + 1)
            w = max(1.0, vol) * math.sqrt(i + 1)
            out.append({"date": d, "p10": p50 - Z*w, "p50": p50, "p90": p50 + Z*w})
        return out

    if variant == "seasonal52":
        # seasonal naive (requires 52w)
        for i, d in enumerate(target_dates):
            idx = len(y) - 52 + i
            p50 = y[idx] if 0 <= idx < len(y) else last
            w = max(1.0, vol) * math.sqrt(i + 1)
            out.append({"date": d, "p10": p50 - Z*w, "p50": p50, "p90": p50 + Z*w})
        return out

    raise ValueError(f"Unknown naive variant: {variant}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True)
    ap.add_argument("--horizon", required=True)  # W4/W13/W26/W52
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--actuals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", required=True, choices=["naive", "xgb", "prophet", "ets"])
    ap.add_argument("--grade", required=False, default="EVOO")
    ap.add_argument("--variant", required=False, default="last")  # for naive
    ap.add_argument("--series-key", required=False, default="")
    args = ap.parse_args()

    market = args.market.strip()
    horizon = args.horizon.strip().upper()
    model_id = args.model_id.strip()
    engine = args.engine.strip()
    grade = args.grade.strip().upper()
    variant = args.variant.strip() if args.variant else "last"
    series_key = args.series_key.strip() or f"{grade}_{market}"

    actuals = read_actuals_csv(args.actuals)
    if not actuals:
        raise RuntimeError("No actuals rows")

    horizon_weeks = horizon_to_weeks(horizon)
    target_dates = make_target_dates(actuals, horizon_weeks)

    if engine == "naive":
        payload_series = naive_forecast(actuals, target_dates, variant)

    elif engine == "xgb":
        feats = build_feature_dict_from_actuals(actuals)
        one = xgb_artifact_forecast(series_key, horizon_weeks, feats)
        payload_series = [{"date": d, "p10": one["p10"], "p50": one["p50"], "p90": one["p90"]} for d in target_dates]

    elif engine in ["prophet", "ets"]:
        prices = [float(a["value"]) for a in actuals]
        preds = ets_forecast(prices, horizon_weeks, mode=engine)
        payload_series = []
        for i, d in enumerate(target_dates):
            p = preds[i]
            payload_series.append({"date": d, "p10": float(p["p10"]), "p50": float(p["p50"]), "p90": float(p["p90"])})

    payload = {
        "modelId": model_id,
        "market": market,
        "horizon": horizon,
        "generatedAt": iso_now(),
        "series": payload_series
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
