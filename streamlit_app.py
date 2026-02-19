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


def latest_actual(df: pd.DataFrame) -> dict[str, Any] | None:
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def build_chart(
    actuals_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    compare: pd.DataFrame = None,
    show_actuals: bool = True,
    show_band: bool = True
):
    base = alt.Chart().encode(
        x=alt.X("date:T", title="Date")
    )
    layers = []
    if show_actuals and not actuals_df.empty:
        layers.append(
            base.mark_line(color="#222", strokeWidth=2).encode(
                y=alt.Y("value:Q", title="Price (€)", scale=alt.Scale(zero=False)),
                tooltip=["date:T", alt.Tooltip("value:Q", title="Actual")],
            ).transform_calculate(series='"Actuals"').properties(data=actuals_df)
        )
    if not forecast_df.empty:
        layers.append(
            base.mark_line(color="#1f77b4", strokeDash=[4,2]).encode(
                y=alt.Y("value:Q"),
                tooltip=["date:T", alt.Tooltip("value:Q", title="Forecast")],
            ).transform_calculate(series='"Forecast"').properties(data=forecast_df)
        )
        if show_band and "lower" in forecast_df.columns and "upper" in forecast_df.columns:
            layers.append(
                base.mark_area(opacity=0.15, color="#1f77b4").encode(
                    y="lower:Q",
                    y2="upper:Q",
                    tooltip=["date:T", "lower:Q", "upper:Q"],
                ).properties(data=forecast_df)
            )
    if compare is not None and not compare.empty:
        layers.append(
            base.mark_line(color="#e377c2", strokeDash=[2,2]).encode(
                y=alt.Y("value:Q"),
                tooltip=["date:T", alt.Tooltip("value:Q", title="Compare")],
            ).transform_calculate(series='"Compare"').properties(data=compare)
        )
    chart = alt.layer(*layers).resolve_scale(y="shared").properties(height=350)
    return chart


def main() -> None:
    st.set_page_config(page_title="OleaForecast | Advanced Dashboard", layout="wide")
    st.title("OleaForecast Advanced")
    st.caption("Advanced dashboard: prices, forecasts, model audit, and data explorer.")

    use_demo = st.sidebar.checkbox(
        "Use demo/sample data", value=True, help="Toggle to use generated demo data for all markets."
    )

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

    tab1, tab2, tab3, tab4 = st.tabs(["Live Prices", "Forecasts", "Models", "Data Explorer"])

    with tab1:
        st.subheader("Live Prices by Region")
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
                actuals_df, _ = load_actuals(market, horizon, use_demo=use_demo)
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
                actuals_df, _ = load_actuals(market, horizon, use_demo=use_demo)
                latest = latest_actual(actuals_df)
                label = f"{market} {horizon}"
                value = latest["value"] if latest else None
                date_label = latest["date"] if latest else "No actuals"
                with price_cols[idx % 2]:
                    st.metric(label, value="--" if value is None else f"€{value:,.0f}", delta=date_label)

    with tab2:
        st.subheader("Forecast Explorer")
        market_choice = st.selectbox("Market", options=markets, key="f_market")
        horizon_choice = st.selectbox("Horizon", options=horizons, key="f_horizon")
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

        all_models = sorted({row.get("winnerModelId") for row in winners if row.get("market") == market_choice and row.get("horizon") == horizon_choice and row.get("winnerModelId")})
        model_choice = st.selectbox("Model", options=all_models if all_models else [winner_model or "none"], key="f_model")

        try:
            actuals_df, actuals_path = load_actuals(market_choice, horizon_choice, use_demo=use_demo)
        except Exception as exc:
            st.error(f"Failed to load actuals: {exc}")
            return

        forecast_df = pd.DataFrame()
        if model_choice and model_choice != "none":
            try:
                forecast_df, forecast_json, forecast_path = load_forecast(
                    market_choice, horizon_choice, model_choice, use_demo=use_demo
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

    with tab3:
        st.subheader("Model Audit & Comparison")
        st.write("Compare models, see eligibility, and audit reasons.")
        market_choice = st.selectbox("Market", options=markets, key="m_market")
        horizon_choice = st.selectbox("Horizon", options=horizons, key="m_horizon")
        rows = [row for row in winners if row.get("market") == market_choice and row.get("horizon") == horizon_choice]
        if not rows:
            st.info("No models found for this selection.")
        else:
            st.dataframe([
                {
                    "Model": row.get("winnerModelId"),
                    "Top-3": ", ".join(row.get("top3", []) or []),
                    "Ineligible": len(row.get("ineligible", []) or []),
                    "Reasons": "; ".join(
                        ", ".join(item.get("reasons", [])) for item in row.get("ineligible", []) or []
                    ),
                }
                for row in rows
            ], use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Data Explorer")
        st.write("Browse and download raw actuals and forecasts.")
        market_choice = st.selectbox("Market", options=markets, key="d_market")
        horizon_choice = st.selectbox("Horizon", options=horizons, key="d_horizon")
        try:
            actuals_df, actuals_path = load_actuals(market_choice, horizon_choice, use_demo=use_demo)
            st.markdown("**Actuals (tail)**")
            st.dataframe(actuals_df.tail(20), use_container_width=True, hide_index=True)
            st.download_button(
                "Download actuals.csv",
                actuals_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{market_choice}_{horizon_choice}_actuals.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.warning(f"Failed to load actuals: {exc}")
        winner_row = next(
            (
                row
                for row in winners
                if row.get("market") == market_choice and row.get("horizon") == horizon_choice
            ),
            None,
        )
        winner_model = winner_row.get("winnerModelId") if winner_row else None
        if winner_model:
            try:
                forecast_df, forecast_json, forecast_path = load_forecast(
                    market_choice, horizon_choice, winner_model, use_demo=use_demo
                )
                st.markdown("**Winner forecast (series)**")
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download forecast.json",
                    json.dumps(forecast_json, indent=2).encode("utf-8"),
                    file_name=f"{market_choice}_{horizon_choice}_{winner_model}.json",
                    mime="application/json",
                )
                st.download_button(
                    "Download forecast.csv",
                    forecast_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{market_choice}_{horizon_choice}_{winner_model}.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.warning(f"Failed to load forecast: {exc}")


if __name__ == "__main__":
    main()
