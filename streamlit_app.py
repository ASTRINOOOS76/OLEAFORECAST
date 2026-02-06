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
RUN_META_PATH = REPORTS_DIR / "run-meta.json"


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


@st.cache_data(show_spinner=False)
def load_run_meta() -> dict[str, Any] | None:
    if not RUN_META_PATH.exists():
        return None
    return json.loads(RUN_META_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def list_models(market: str, horizon: str) -> list[str]:
    models: set[str] = set()
    for base in [DATA_DIR, FIXTURES_DIR]:
        folder = base / market / horizon
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            if path.name == "winners.json":
                continue
            models.add(path.stem)
    return sorted(models)


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
    run_meta = load_run_meta()
    st.caption(f"generatedAt: {data.get('generatedAt', 'n/a')} · rows: {len(results)}")
    if run_meta:
        st.caption(
            "runMeta: "
            f"generatedAt={run_meta.get('generatedAt', 'n/a')} · "
            f"gitSha={run_meta.get('gitSha', 'n/a')} · "
            f"dataHash={run_meta.get('dataHash', 'n/a')}"
        )

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

    all_models = list_models(market, horizon)
    if winner not in all_models:
        all_models = [winner] + [m for m in all_models if m != winner]

    model_cols = st.columns(3)
    with model_cols[0]:
        model_choice = st.selectbox("Model", options=all_models, index=0)
    with model_cols[1]:
        compare_choice = st.selectbox("Compare", options=["None"] + all_models, index=0)
    with model_cols[2]:
        last_n = st.slider("Chart window (points)", min_value=0, max_value=200, value=0, step=5)

    compare_df = None
    try:
        actuals_df, actuals_path = load_actuals(market, horizon)
        forecast_df, forecast_json, forecast_path = load_forecast(market, horizon, model_choice)
        if compare_choice != "None" and compare_choice != model_choice:
            try:
                compare_df, _, _ = load_forecast(market, horizon, compare_choice)
            except Exception as exc:
                st.warning(f"Compare load failed: {exc}")
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
            "modelId": forecast_json.get("modelId", model_choice),
            "market": forecast_json.get("market", market),
            "horizon": forecast_json.get("horizon", horizon),
            "actuals_rows": len(actuals_df),
            "forecast_points": len(forecast_df),
        }
    )

    download_cols = st.columns(3)
    with download_cols[0]:
        st.download_button(
            "Download actuals.csv",
            actuals_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{market}_{horizon}_actuals.csv",
            mime="text/csv",
        )
    with download_cols[1]:
        st.download_button(
            "Download forecast.json",
            json.dumps(forecast_json, indent=2).encode("utf-8"),
            file_name=f"{market}_{horizon}_{model_choice}.json",
            mime="application/json",
        )
    with download_cols[2]:
        st.download_button(
            "Download forecast.csv",
            forecast_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{market}_{horizon}_{model_choice}.csv",
            mime="text/csv",
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
    coverage = (len(overlap) / len(forecast_df)) if len(forecast_df) else 0.0
    st.metric("Coverage", f"{coverage:.0%}")
    if coverage < 0.6:
        st.warning("Low overlap between actuals and forecast points.")
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
    show_actuals = st.checkbox("Show actuals line", value=True)
    show_band = st.checkbox("Show p10/p90 band", value=True)

    chart_actuals = actuals_df.copy()
    chart_forecast = forecast_df.copy()
    chart_compare = compare_df.copy() if compare_df is not None else None

    if last_n > 0:
        chart_actuals = chart_actuals.tail(last_n)
        chart_forecast = chart_forecast.tail(last_n)
        if chart_compare is not None:
            chart_compare = chart_compare.tail(last_n)

    chart = build_chart(
        chart_actuals,
        chart_forecast,
        compare=chart_compare,
        show_actuals=show_actuals,
        show_band=show_band,
    )
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
