from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIXTURES_DIR = DATA_DIR / "_fixtures"
REPORTS_DIR = ROOT / "reports"
RUN_META_PATH = REPORTS_DIR / "run-meta.json"


@st.cache_data(show_spinner=False)
def load_winners() -> list[dict[str, Any]]:
    candidates = [
        REPORTS_DIR / "winners.json",
        FIXTURES_DIR / "winners.json",
    ]
    for path in candidates:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("results", [])
    return []


@st.cache_data(show_spinner=False)
def load_run_meta() -> dict[str, Any] | None:
    if not RUN_META_PATH.exists():
        return None
    return json.loads(RUN_META_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def list_market_horizon_pairs() -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for base in [DATA_DIR, FIXTURES_DIR]:
        if not base.exists():
            continue
        for market_dir in base.iterdir():
            if not market_dir.is_dir() or market_dir.name.startswith("_"):
                continue
            for horizon_dir in market_dir.iterdir():
                if not horizon_dir.is_dir():
                    continue
                if (horizon_dir / "actuals.csv").exists():
                    pairs.add((market_dir.name, horizon_dir.name))
    return sorted(pairs)


def load_actuals(market: str, horizon: str) -> tuple[pd.DataFrame, str]:
    paths = [
        DATA_DIR / market / horizon / "actuals.csv",
        FIXTURES_DIR / market / horizon / "actuals.csv",
    ]
    for path in paths:
        if path.exists():
            df = pd.read_csv(path)
            if list(df.columns) != ["date", "value"]:
                raise ValueError(f"Bad actuals header in {path}")
            return df, str(path.relative_to(ROOT))
    raise FileNotFoundError(f"actuals not found for {market}/{horizon}")


def load_forecast(market: str, horizon: str, model_id: str) -> tuple[pd.DataFrame, dict[str, Any], str]:
    paths = [
        DATA_DIR / market / horizon / f"{model_id}.json",
        FIXTURES_DIR / market / horizon / f"{model_id}.json",
    ]
    for path in paths:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(payload.get("series", []))
            return df, payload, str(path.relative_to(ROOT))
    raise FileNotFoundError(f"forecast not found for {market}/{horizon}/{model_id}")


def build_chart(
    actuals: pd.DataFrame,
    forecast: pd.DataFrame,
    compare: pd.DataFrame | None = None,
    show_actuals: bool = True,
    show_band: bool = True,
) -> alt.Chart:
    layers: list[alt.Chart] = []

    if show_band and not forecast.empty:
        band = (
            alt.Chart(forecast)
            .mark_area(color="#111", opacity=0.08)
            .encode(
                x="date:T",
                y="p10:Q",
                y2="p90:Q",
                tooltip=["date:T", "p10:Q", "p50:Q", "p90:Q"],
            )
        )
        layers.append(band)

    if not forecast.empty:
        p50_line = (
            alt.Chart(forecast)
            .mark_line(color="#111", strokeWidth=2)
            .encode(x="date:T", y="p50:Q", tooltip=["date:T", "p50:Q"])
        )
        layers.append(p50_line)

    if show_actuals and not actuals.empty:
        actuals_line = (
            alt.Chart(actuals)
            .mark_line(color="#555", strokeWidth=2)
            .encode(x="date:T", y="value:Q", tooltip=["date:T", "value:Q"])
        )
        layers.append(actuals_line)

    if compare is not None and not compare.empty:
        compare_line = (
            alt.Chart(compare)
            .mark_line(color="#1f77b4", strokeWidth=2, strokeDash=[4, 2])
            .encode(x="date:T", y="p50:Q", tooltip=["date:T", "p50:Q"])
        )
        layers.append(compare_line)

    chart = alt.layer(*layers).properties(height=280)
    return chart.interactive()


def latest_actual(actuals: pd.DataFrame) -> dict[str, Any] | None:
    if actuals.empty:
        return None
    ordered = actuals.sort_values("date")
    row = ordered.iloc[-1]
    return {"date": row["date"], "value": float(row["value"])}


def main() -> None:
    st.set_page_config(page_title="OleaForecast | Live Prices", layout="wide")

    st.title("OleaForecast")
    st.caption("Live prices and horizon forecasts by country.")

    run_meta = load_run_meta()
    if run_meta:
        st.caption(
            "runMeta: "
            f"generatedAt={run_meta.get('generatedAt', 'n/a')} · "
            f"gitSha={run_meta.get('gitSha', 'n/a')} · "
            f"dataHash={run_meta.get('dataHash', 'n/a')}"
        )

    winners = load_winners()
    pairs = list_market_horizon_pairs()
    markets = sorted({market for market, _ in pairs})
    horizons = sorted({horizon for _, horizon in pairs})

    st.subheader("Market radar")
    st.write(
        "Track live olive oil prices and horizon forecasts by country. "
        "Pick a market, compare the latest actuals with the winning model, "
        "and stay ahead of price movement."
    )
    st.caption("Data source: local reports/ and data/ snapshots with fixtures fallback.")

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        st.metric("Markets tracked", value=len(markets))
    with kpi_cols[1]:
        st.metric("Horizons available", value=len(horizons))
    with kpi_cols[2]:
        st.metric(
            "Winner models",
            value=sum(1 for row in winners if row.get("winnerModelId")),
        )

    st.subheader("Live Prices")
    if not pairs:
        st.info("No markets found. Check data/ and fixtures for actuals.csv files.")
        return

    region_map = {
        "Southern EU": ["ES", "IT", "GR", "PT"],
        "North Africa": ["TN"],
        "Anatolia": ["TR"],
    }

    grouped: dict[str, list[tuple[str, str]]] = {key: [] for key in region_map}
    grouped["Other"] = []
    for market, horizon in pairs:
        placed = False
        for region, markets_list in region_map.items():
            if market in markets_list:
                grouped[region].append((market, horizon))
                placed = True
                break
        if not placed:
            grouped["Other"].append((market, horizon))

    for region, items in grouped.items():
        if not items:
            continue
        st.markdown(f"**{region}**")
        items = sorted(items, key=lambda row: (row[0], row[1]))
        weighted_sum = 0.0
        weighted_count = 0
        for market, horizon in items:
            actuals_df, _ = load_actuals(market, horizon)
            latest = latest_actual(actuals_df)
            if latest:
                count = len(actuals_df)
                if count > 0:
                    weighted_sum += latest["value"] * count
                    weighted_count += count

        region_cols = st.columns(2)
        avg_value = (weighted_sum / weighted_count) if weighted_count else None
        with region_cols[0]:
            st.metric(
                "Region avg (weighted)",
                value="--" if avg_value is None else f"€{avg_value:,.0f}",
            )
        with region_cols[1]:
            st.metric("Markets", value=len(items))

        price_cols = st.columns(2)
        for idx, (market, horizon) in enumerate(items):
            actuals_df, _ = load_actuals(market, horizon)
            latest = latest_actual(actuals_df)
            label = f"{market} {horizon}"
            value = latest["value"] if latest else None
            date_label = latest["date"] if latest else "No actuals"
            with price_cols[idx % 2]:
                st.metric(label, value="--" if value is None else f"€{value:,.0f}", delta=date_label)

    st.subheader("Forecast Explorer")
    market_choice = st.selectbox("Market", options=markets)
    horizon_choice = st.selectbox("Horizon", options=horizons)

    winner_row = next(
        (
            row
            for row in winners
            if row.get("market") == market_choice and row.get("horizon") == horizon_choice
        ),
        None,
    )
    winner_model = winner_row.get("winnerModelId") if winner_row else None
    st.write(f"Winner model: {winner_model or 'none'}")

    try:
        actuals_df, actuals_path = load_actuals(market_choice, horizon_choice)
    except Exception as exc:
        st.error(f"Failed to load actuals: {exc}")
        return

    forecast_df = pd.DataFrame()
    if winner_model:
        try:
            forecast_df, forecast_json, forecast_path = load_forecast(
                market_choice, horizon_choice, winner_model
            )
            st.caption(f"forecast: {forecast_path}")
        except Exception as exc:
            st.warning(f"Failed to load forecast: {exc}")

    st.caption(f"actuals: {actuals_path}")

    chart = build_chart(actuals_df, forecast_df, compare=None, show_actuals=True, show_band=True)
    st.altair_chart(chart, use_container_width=True)

    latest = latest_actual(actuals_df)
    if latest:
        st.caption(f"Latest actual: {latest['date']} at €{latest['value']:,.0f}")
    if not winner_model:
        st.info("No winner model recorded for this market/horizon.")


if __name__ == "__main__":
    main()
