"""
AI Paid Media Intelligence & Optimization
Streamlit prototype. Run with: streamlit run app.py

Tabs map directly to the project's minimum bar:
  1. Ingestion & Reconciliation (M1)
  2. Forecast (M2)
  3. Budget Allocation (M3)
  4. Marketer Approval View (M3)
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent / "src"))
from ingest_reconcile import unify
from optimize_allocation import fit_efficiency_curves, recommend_allocation, hill
from budget_utilization import flag_waste_and_underspend, summarize as bu_summarize
from llm_narrator import generate_recommendation_narrative
from nc_cpa import run_all_channels as run_nc_cpa, NC_CPA_TARGET
from business_impact import estimate_dio_impact, estimate_fulfillment_impact

DATA = Path(__file__).parent / "data"

st.set_page_config(page_title="Paid Media Intelligence", layout="wide")

# Load the API key from .streamlit/secrets.toml if present (env var takes precedence).
try:
    _secret_key = st.secrets.get("ANTHROPIC_API_KEY", "")
except Exception:
    _secret_key = ""
if _secret_key and not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = _secret_key

with st.sidebar:
    st.markdown("### LLM Configuration")
    st.caption("The recommendation narrator (tab 4) tries, in order: "
               "**1) a local Ollama model** (free, no key, no data leaves your machine) → "
               "**2) the Claude API** (if you set a key below) → "
               "**3) a deterministic template** (zero hallucination risk, always works).")
    api_key_input = st.text_input("Anthropic API key (optional — only used if Ollama isn't running)",
                                    type="password",
                                    help="Leave blank to rely on Ollama or the template fallback. "
                                         "The app works fully with no key at all.")
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
        st.success("API key set for this session (used only as a fallback if Ollama isn't reachable).")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        st.success("API key loaded from environment / .streamlit/secrets.toml.")
    else:
        st.info("No key set — narrator will use Ollama if running locally, otherwise the template.")

st.title("🧵 AI Paid Media Intelligence & Optimization")
st.caption("Prototype · Meta + Google + Amazon (synthetic, benchmark-anchored data) · "
           "Forecast model selected per-channel by 4-fold rolling cross-validation · "
           "Allocation optimized for blended ROAS subject to constraints")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Ingestion & Reconciliation", "2️⃣ Forecast", "3️⃣ Budget Allocation",
    "4️⃣ Marketer Approval", "5️⃣ Budget Utilization", "6️⃣ NC-CPA & Business Impact"
])

# ---------------------------------------------------------------- TAB 1: M1
with tab1:
    st.subheader("Cross-platform ingestion")
    st.write("Raw exports → reconciliation against canonical SKU master → unified schema.")

    meta_raw = pd.read_csv(DATA / "meta_ads_export.csv")
    google_raw = pd.read_csv(DATA / "google_ads_export.csv")
    amazon_raw = pd.read_csv(DATA / "amazon_ads_export.csv")
    report, unified = unify()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Meta rows ingested", report["meta_total_rows"])
        st.metric("Meta reconciled", report["meta_reconciled"],
                   delta=f"-{report['meta_unreconciled']} failed" if report['meta_unreconciled'] else "0 failed")
    with c2:
        st.metric("Google rows ingested", report["google_total_rows"])
        st.metric("Google reconciled", report["google_reconciled"],
                   delta=f"-{report['google_unreconciled']} failed", delta_color="inverse")
    with c3:
        st.metric("Amazon rows ingested", report["amazon_total_rows"])
        st.metric("Amazon reconciled", report["amazon_reconciled"],
                   delta=f"-{report['amazon_unreconciled']} failed", delta_color="inverse")

    if report["google_unreconciled"] > 0:
        st.warning(
            f"⚠️ {report['google_unreconciled']} Google rows failed SKU reconciliation. "
            f"Root cause: the category 'T-Shirt' (and similarly 'Sponsored' multi-word categories) "
            f"contains a hyphen, which collides with the '-' delimiter used in the platform key "
            f"(goog::t-shirt-black-men), breaking the 3-way split on category/color/gender. "
            f"In production this map would come from a PIM feed with stable platform catalog "
            f"IDs, not string parsing."
        )
    if report["amazon_unreconciled"] > 0:
        st.warning(
            f"⚠️ {report['amazon_unreconciled']} Amazon rows failed SKU reconciliation. "
            f"Root cause: a listing was re-platformed under a new ASIN ('B0GHOST99X') that "
            f"hasn't propagated into the Seller Central SKU↔ASIN mapping table yet — a real "
            f"catalog-drift failure mode, not a parsing bug like the Google case above."
        )

    st.markdown(f"**Missing revenue rows imputed (channel×SKU median):** "
                f"Meta {report['meta_missing_revenue']} · Google {report['google_missing_revenue']} · "
                f"Amazon {report['amazon_missing_revenue']}")

    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Raw Meta export (sample)**")
        st.dataframe(meta_raw.head(8), use_container_width=True, height=250)
    with colB:
        st.markdown("**Raw Google export (sample)**")
        st.dataframe(google_raw.head(8), use_container_width=True, height=250)
    with colC:
        st.markdown("**Raw Amazon export (sample)**")
        st.dataframe(amazon_raw.head(8), use_container_width=True, height=250)

    st.markdown("**Unified, reconciled performance table**")
    st.dataframe(unified.head(15), use_container_width=True)

    spend_by_channel = unified.groupby("channel")["spend"].sum().reset_index()
    fig = px.pie(spend_by_channel, names="channel", values="spend", title="Spend share by channel")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- TAB 2: M2
with tab2:
    st.subheader("AI Performance Forecasting Engine")
    st.write("Four candidate models were evaluated per channel using **4-fold rolling-window "
             "cross-validation** (not a single holdout) — Prophet, XGBoost, LightGBM, and a "
             "Ridge log-log power-law model, against a naive baseline. The best model is selected "
             "**per channel** by lowest mean CV MAPE.")

    results_df = pd.read_csv(DATA / "model_comparison_results.csv")
    with open(DATA / "best_forecasts.json") as f:
        forecasts = json.load(f)

    st.markdown("**Model comparison (mean ± std MAPE across CV folds)**")
    def highlight_best(row):
        is_best = row["MAPE_mean_%"] == results_df[results_df.channel == row.channel]["MAPE_mean_%"].min()
        return ["background-color: #d4f4dd" if is_best else "" for _ in row]
    st.dataframe(results_df.style.apply(highlight_best, axis=1), use_container_width=True)

    sel_channel = st.selectbox("Channel", list(forecasts.keys()))
    fc = forecasts[sel_channel]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fc["dates"], y=fc["actual"], name="Actual", mode="lines+markers"))
    fig2.add_trace(go.Scatter(x=fc["dates"], y=fc["best_pred"], name=f"Forecast ({fc['best_model']})",
                               mode="lines+markers", line=dict(dash="dash")))
    if fc["ci_lower"]:
        fig2.add_trace(go.Scatter(x=fc["dates"] + fc["dates"][::-1],
                                   y=fc["ci_upper"] + fc["ci_lower"][::-1],
                                   fill="toself", fillcolor="rgba(99,110,250,0.15)",
                                   line=dict(width=0), name="Confidence interval"))
    fig2.update_layout(title=f"{sel_channel} — actual vs forecast (winning model: {fc['best_model']})",
                        xaxis_title="Date", yaxis_title="Revenue ($)")
    st.plotly_chart(fig2, use_container_width=True)
    st.info(f"**Selected model for {sel_channel}: {fc['best_model']}** — chosen automatically "
            f"because it had the lowest backtest MAPE among all candidates for this channel.")

# ---------------------------------------------------------------- TAB 3: M3
with tab3:
    st.subheader("Budget Allocation & Diminishing Returns")
    curves, daily = fit_efficiency_curves()

    st.markdown("**Fitted efficiency curves** (Hill/saturation function fit to observed spend→revenue)")
    spend_range = np.linspace(1, daily["spend"].max() * 1.3, 100)
    fig3 = go.Figure()
    for ch, p in curves.items():
        fig3.add_trace(go.Scatter(x=spend_range, y=hill(spend_range, p["R_max"], p["S50"], p["k"]),
                                   name=f"{ch} efficiency curve", mode="lines"))
        fig3.add_trace(go.Scatter(x=[p["current_daily_spend"]],
                                   y=[hill(p["current_daily_spend"], p["R_max"], p["S50"], p["k"])],
                                   mode="markers", marker=dict(size=12),
                                   name=f"{ch} current spend", showlegend=False))
    fig3.update_layout(title="Spend → Revenue efficiency curves (diminishing returns visible as flattening)",
                        xaxis_title="Daily spend ($)", yaxis_title="Daily revenue ($)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Constraints**")
    c1, c2 = st.columns(2)
    min_pct = c1.slider("Min spend (% of current)", 40, 100, 60) / 100
    max_pct = c2.slider("Max spend (% of current)", 100, 250, 160) / 100

    rec_df, summary = recommend_allocation(curves, min_pct=min_pct, max_pct=max_pct)
    st.session_state["rec_df"] = rec_df
    st.session_state["summary"] = summary

    m1, m2, m3 = st.columns(3)
    m1.metric("Current blended ROAS", f"{summary['current_blended_roas']}x")
    m2.metric("Projected blended ROAS", f"{summary['projected_blended_roas']}x",
               delta=f"{summary['projected_blended_roas'] - summary['current_blended_roas']:+.2f}x")
    m3.metric("Projected daily revenue lift", f"${summary['projected_revenue_lift']:,.0f}")

    st.dataframe(rec_df, use_container_width=True)

    for _, r in rec_df.iterrows():
        if r["diminishing_returns_flag"]:
            st.warning(f"📉 **{r['channel']}** is showing diminishing returns "
                       f"(marginal ROAS {r['current_marginal_roas']}x) — recommend capping further increases.")

    TARGET_ROAS = 4.0
    if summary["projected_blended_roas"] < TARGET_ROAS:
        gap = TARGET_ROAS - summary["projected_blended_roas"]
        avg_marginal = rec_df["projected_marginal_roas"].mean()
        st.error(
            f"⚠️ **Reallocation alone can't hit the {TARGET_ROAS}x blended ROAS target.** "
            f"Even at the optimal split, projected blended ROAS is {summary['projected_blended_roas']}x "
            f"— {gap:.2f}x short. All three channels have converged to similar marginal ROAS "
            f"(~{avg_marginal:.2f}x), meaning the constraint isn't *which* channel gets the next "
            f"dollar — it's that total spend now exceeds efficient auction capacity everywhere. "
            f"The lever that actually closes this gap is **total budget reduction** or "
            f"**SKU/creative-level efficiency work**, not further reshuffling."
        )

# ---------------------------------------------------------------- TAB 4: M3 approval
with tab4:
    st.subheader("Marketer-Facing Approval View")
    st.write("Humans review, adjust, and approve before anything executes. **No API calls fire from this prototype** — "
             "the 'Approve' action below would, in production, call the Meta/Google Ads API to push the new budget.")

    if "rec_df" not in st.session_state:
        st.info("Visit the Budget Allocation tab first to generate a recommendation.")
    else:
        rec_df = st.session_state["rec_df"].copy()
        summary = st.session_state["summary"]
        decisions = {}
        for i, r in rec_df.iterrows():
            with st.container(border=True):
                narrative = generate_recommendation_narrative(r.to_dict(), summary)
                badge = {"ollama": "🖥️ Local model (Ollama)", "anthropic": "☁️ Claude API",
                         "template": "📋 Template (no LLM)"}[narrative.provider]
                st.caption(badge)
                st.write(narrative.text)
                if narrative.fallback_reason:
                    with st.expander("Why template, not a live model?"):
                        st.write(narrative.fallback_reason)
                if not narrative.groundedness_check_passed:
                    st.warning("Groundedness check failed on the model's first attempt — "
                               "rejected and replaced with the template above rather than risk "
                               "showing a fabricated number.")

                cols = st.columns([2, 2, 2, 2, 2])
                cols[0].markdown(f"**{r['channel']}**")
                cols[1].write(f"Current: ${r['current_daily_spend']:,.0f}/day")
                new_val = cols[2].number_input(
                    f"Recommended spend — {r['channel']}", value=float(r["recommended_daily_spend"]),
                    key=f"spend_{r['channel']}", label_visibility="collapsed")
                cols[3].write(f"ROAS: {r['current_roas']}x → {r['projected_roas']}x")
                decision = cols[4].radio("Decision", ["Approve", "Reject", "Adjust"],
                                          key=f"dec_{r['channel']}", horizontal=True, label_visibility="collapsed")
                decisions[r['channel']] = {"approved_spend": new_val, "decision": decision}

        if st.button("✅ Submit decisions", type="primary"):
            st.success("Decisions recorded. In production this triggers stubbed Meta/Google Ads API "
                       "budget-update calls; nothing executes automatically without this step.")
            st.json(decisions)

# ---------------------------------------------------------------- TAB 5: Budget Utilization
with tab5:
    st.subheader("Budget Waste & Underspend Detector")
    st.write("Addresses the project's **tertiary** success metric directly: efficient campaigns that "
             "hit their daily ceiling are leaving revenue on the table; inefficient campaigns burning "
             "full budgets are wasting cash. **Honest limitation**: this dataset has no hourly pacing "
             "data, so true 'hit the cap by noon' detection isn't possible — what's shown below is a "
             "daily proxy: each SKU-channel's own historical 90th-percentile spend stands in for its "
             "budget cap.")

    unified_df = pd.read_csv(DATA / "unified_performance.csv", parse_dates=["date"])
    flagged = flag_waste_and_underspend(unified_df)
    bu_summary = bu_summarize(flagged)

    c1, c2, c3 = st.columns(3)
    c1.metric("Budget waste flagged", f"{flagged['flag_budget_waste'].mean():.1%}",
               help="Near-cap spend + below-median ROAS")
    c2.metric("Underspend opportunity", f"{flagged['flag_underspend_opportunity'].mean():.1%}",
               help="Low spend vs. ceiling + above-median ROAS")
    c3.metric("Efficient utilization", f"{flagged['flag_efficient_utilization'].mean():.1%}",
               help="Near-cap spend + above-median ROAS — the healthy case")

    sub1, sub2 = st.columns(2)
    with sub1:
        st.markdown("**Top SKU-channel combos by waste rate**")
        st.dataframe(bu_summary.sort_values("waste_rate", ascending=False).head(10)[
            ["channel", "sku", "waste_rate", "avg_roas", "avg_spend", "spend_ceiling_p90"]
        ], use_container_width=True)
    with sub2:
        st.markdown("**Top SKU-channel combos by underspend opportunity**")
        st.dataframe(bu_summary.sort_values("underspend_rate", ascending=False).head(10)[
            ["channel", "sku", "underspend_rate", "avg_roas", "avg_spend", "spend_ceiling_p90"]
        ], use_container_width=True)

    fig5 = px.scatter(bu_summary, x="avg_spend", y="avg_roas", size="days_observed",
                       color="channel", hover_data=["sku"],
                       title="SKU-channel avg ROAS vs avg spend — bubble size = days observed")
    st.plotly_chart(fig5, use_container_width=True)

# ---------------------------------------------------------------- TAB 6: NC-CPA & Business Impact
with tab6:
    st.subheader("Secondary Success Metric: New Customer Acquisition Cost")
    st.write(f"Target: **NC-CPA ≤ ${NC_CPA_TARGET}**. Computed from raw exports (not the aggregated "
             "unified table) because the prospecting-vs-retargeting split the requirements ask for only exists "
             "at that grain — spend is proportionally allocated by each row's new-customer conversion share.")

    nc_cpa_df = run_nc_cpa()
    channel_view = nc_cpa_df[nc_cpa_df["cut"] == "channel_blended"]
    funnel_view = nc_cpa_df[nc_cpa_df["cut"] == "prospecting_vs_retargeting"]

    cols = st.columns(3)
    for i, (_, row) in enumerate(channel_view.iterrows()):
        status = "✅" if row["within_target"] else "⚠️"
        cols[i].metric(f"{status} {row['channel']} NC-CPA", f"${row['nc_cpa']:.2f}",
                        delta=f"{row['nc_cpa_vs_target']:+.2f} vs ${NC_CPA_TARGET} target",
                        delta_color="inverse")

    st.markdown("**Prospecting vs. retargeting split** — the metric that actually matters, since a "
                "blended channel average can hide a starved top-of-funnel:")
    st.dataframe(funnel_view[["channel", "segment", "nc_cpa", "blended_cpa", "within_target",
                               "new_customer_conversions"]], use_container_width=True)

    st.divider()
    st.subheader("Business Judgment: DIO, Stockout Risk & Fulfillment Cost")
    st.caption("Every technical decision below traces back to the budget waste/underspend flags already "
               "computed in tab 5 — this section translates them into inventory-economics language, "
               "grounded in cited 2026 DTC apparel benchmarks. Explicitly directional, not a precision "
               "financial model — stated plainly, not hidden.")

    from business_impact import load_flags as _load_flags
    flags = _load_flags()
    dio_impact = estimate_dio_impact(flags)
    fulfillment_impact = estimate_fulfillment_impact(flags)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**📦 Inventory Velocity (DIO)**")
        st.metric("Hero-SKU unused efficient ad capacity",
                   f"${dio_impact['hero_unused_efficient_capacity_dollars']:,.0f}",
                   help=f"{dio_impact['headroom_as_pct_of_hero_spend']}% of hero spend, across "
                        f"{dio_impact['hero_underspend_flagged_days']:,} flagged days")
        st.write(dio_impact["directional_read"])
    with d2:
        st.markdown("**🚚 Fulfillment / Working Capital**")
        st.metric("Flagged budget waste",
                   f"${fulfillment_impact['total_flagged_waste_dollars']:,.0f}",
                   help=f"{fulfillment_impact['waste_as_pct_of_total_spend']}% of total spend")
        st.write(fulfillment_impact["directional_read"])
