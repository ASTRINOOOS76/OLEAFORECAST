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
DIST_HISTORY_DIR = ROOT / "dist" / "history"


@st.cache_data(show_spinner=False)
def load_latest_winners() -> dict[str, Any]:
    candidates = [
        (REPORTS_DIR / "winners.json", "reports/winners.json"),
        (FIXTURES_DIR / "winners.json", "data/_fixtures/winners.json"),
    ]
    for path, label in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "winners.json not found. Expected reports/winners.json (commit it) "
        "or data/_fixtures/winners.json"
    )


@st.cache_data(show_spinner=False)
def load_history_index() -> dict[str, Any] | None:
    index_path = DIST_HISTORY_DIR / "index.json"
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_history_run(run_key: str) -> dict[str, Any]:
    run_path = DIST_HISTORY_DIR / run_key / "winners.json"
    if not run_path.exists():
        raise FileNotFoundError(f"dist/history/{run_key}/winners.json not found")
    return json.loads(run_path.read_text(encoding="utf-8"))


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


def build_chart(actuals: pd.DataFrame, forecast: pd.DataFrame) -> alt.Chart:
    actuals_line = (
        alt.Chart(actuals)
        .mark_line(color="#555", strokeWidth=2)
        .encode(x="date:T", y="value:Q")
    )

    band = (
        alt.Chart(forecast)
        .mark_area(color="#111", opacity=0.08)
        .encode(x="date:T", y="p10:Q", y2="p90:Q")
    )

    p50_line = (
        alt.Chart(forecast)
        .mark_line(color="#111", strokeWidth=2)
        .encode(x="date:T", y="p50:Q")
    )

    return (band + p50_line + actuals_line).properties(height=280)


def main() -> None:
    st.set_page_config(page_title="OleaForecast Dashboard", layout="wide")

    st.title("OleaForecast")
    st.caption("Streamlit drilldown using local data/ and reports/ with fixtures fallback.")

    history_index = load_history_index()
    run_options = ["latest"]
    if history_index:
        run_options += history_index.get("runs", [])

    run_key = st.selectbox("Run", options=run_options, index=0)
    try:
        if run_key == "latest":
            data = load_latest_winners()
        else:
            data = load_history_run(run_key)
    except Exception as exc:
        st.error(f"Failed to load winners data: {exc}")
        st.stop()

    results = data.get("results", [])
    st.caption(f"generatedAt: {data.get('generatedAt', 'n/a')} · rows: {len(results)}")

    markets = sorted({r.get("market", "") for r in results if r.get("market")})
    market_filter = st.selectbox("Market", options=["All"] + markets)
    query = st.text_input("Search", placeholder="ES / IT / W4 / modelId…")

    def match_row(row: dict[str, Any]) -> bool:
        if market_filter != "All" and row.get("market") != market_filter:
            return False
        if not query:
            return True
        blob = " ".join(
            [
                str(row.get("market", "")),
                str(row.get("horizon", "")),
                str(row.get("winnerModelId", "")),
                " ".join(row.get("top3", []) or []),
                " ".join([x.get("modelId", "") for x in row.get("ineligible", []) or []]),
            ]
        ).lower()
        return query.lower() in blob

    filtered = [r for r in results if match_row(r)]

    st.subheader("Winners")
    st.dataframe(
        [
            {
                "Market": r.get("market"),
                "Horizon": r.get("horizon"),
                "Winner": r.get("winnerModelId"),
                "Top-3": ", ".join(r.get("top3", []) or []),
                "Ineligible": len(r.get("ineligible", []) or []),
                "Actuals N": (r.get("actualsQuality", {}) or {}).get("count"),
            }
            for r in filtered
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Drilldown")
    if not filtered:
        st.info("No rows match the current filter.")
        return

    options = [
        f"{r.get('market')} / {r.get('horizon')} — {r.get('winnerModelId')}"
        for r in filtered
    ]
    selected_label = st.selectbox("Select row", options=options)
    selected_index = options.index(selected_label)
    selected = filtered[selected_index]

    market = selected.get("market")
    horizon = selected.get("horizon")
    winner = selected.get("winnerModelId")

    try:
        actuals_df, actuals_path = load_actuals(market, horizon)
        forecast_df, forecast_json, forecast_path = load_forecast(market, horizon, winner)
    except Exception as exc:
        st.error(f"Load failed: {exc}")
        return

    meta_cols = st.columns(2)
    with meta_cols[0]:
        st.caption(f"actuals: {actuals_path}")
    with meta_cols[1]:
        st.caption(f"forecast: {forecast_path}")

    st.write(
        {
            "generatedAt": forecast_json.get("generatedAt", "n/a"),
            "modelId": forecast_json.get("modelId", winner),
            "market": forecast_json.get("market", market),
            "horizon": forecast_json.get("horizon", horizon),
            "actuals_rows": len(actuals_df),
            "forecast_points": len(forecast_df),
        }
    )

    tail_cols = st.columns(2)
    with tail_cols[0]:
        st.markdown("**Actuals (tail)**")
        st.dataframe(actuals_df.tail(12), use_container_width=True, hide_index=True)

    with tail_cols[1]:
        st.markdown("**Winner forecast (series)**")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    st.markdown("**Overlap check**")
    overlap = sorted(set(actuals_df["date"]).intersection(set(forecast_df["date"])))
    miss = sorted(set(forecast_df["date"]) - set(actuals_df["date"]))
    st.code(
        "\n".join(
            [
                f"overlap_count={len(overlap)} / forecast_points={len(forecast_df)}",
                f"overlap_dates={', '.join(overlap) if overlap else '(none)'}",
                f"missing_in_actuals={', '.join(miss) if miss else '(none)'}",
            ]
        )
    )

    st.markdown("**Chart**")
    chart = build_chart(actuals_df, forecast_df)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Ineligible details")
    for row in filtered:
        ineligible = row.get("ineligible", []) or []
        if not ineligible:
            continue
        with st.expander(f"{row.get('market')} / {row.get('horizon')} — ineligible: {len(ineligible)}"):
            st.write(
                [
                    {
                        "modelId": item.get("modelId"),
                        "reasons": ", ".join(item.get("reasons", []) or []),
                    }
                    for item in ineligible
                ]
            )


if __name__ == "__main__":
    main()
