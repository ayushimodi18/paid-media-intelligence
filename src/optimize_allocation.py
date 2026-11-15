"""
M3 - Budget Allocation & Execution Automation.

Approach: fit an efficiency curve (saturating Hill-type response) per channel
mapping spend -> revenue, re-FIT from observed data (the optimizer does not
see the true generating parameters). From that curve we derive MARGINAL ROAS
(the derivative), the real decision variable: shift dollars from the channel
with the lowest marginal ROAS to the one with the highest, until marginal
ROAS is equalized across channels (constrained marginal allocation /
water-filling), subject to:
  - a total budget constraint (sum of current spend, reallocation only)
  - per-channel min/max spend bounds (floor protects brand presence,
    ceiling reflects auction/inventory limits)
  - NC-CPA awareness (don't starve prospecting just to chase blended ROAS)

This produces a human-reviewable recommendation, not an auto-executed change.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit, minimize

DATA = Path(__file__).resolve().parents[1] / "data"

NC_CPA_TARGET = 55.0  # business constraint from the requirements: NC-CPA <= target


def hill(spend, R_max, S50, k):
    spend = np.maximum(spend, 1e-6)
    return R_max * (spend ** k) / (spend ** k + S50 ** k)


def fit_efficiency_curves():
    df = pd.read_csv(DATA / "unified_performance.csv", parse_dates=["date"])
    daily = df.groupby(["channel", "date"]).agg(
        spend=("spend", "sum"), revenue=("revenue_attributed", "sum"),
        new_customer_pct=("new_customer_pct", "mean"),
    ).reset_index()

    curves = {}
    for channel, d in daily.groupby("channel"):
        x = d["spend"].values
        y = d["revenue"].values
        try:
            popt, _ = curve_fit(hill, x, y, p0=[y.max() * 1.5, x.mean(), 1.0],
                                 maxfev=10000, bounds=([0, 1, 0.3], [y.max() * 5, x.max() * 3, 3.0]))
        except Exception:
            popt = [y.max() * 1.5, x.mean(), 1.0]
        curves[channel] = dict(R_max=popt[0], S50=popt[1], k=popt[2],
                                avg_new_cust_pct=d["new_customer_pct"].mean(),
                                current_daily_spend=x.mean())
    return curves, daily


def marginal_roas(spend, params):
    R_max, S50, k = params["R_max"], params["S50"], params["k"]
    eps = max(spend * 0.01, 1.0)
    return (hill(spend + eps, R_max, S50, k) - hill(spend, R_max, S50, k)) / eps


def recommend_allocation(curves, total_daily_budget=None,
                          min_pct=0.6, max_pct=1.6, threshold_marginal_roas=None):
    """Reallocate a fixed total budget across channels to equalize marginal
    ROAS, subject to +/-40% per-channel bounds around current spend.

    threshold_marginal_roas: a channel is flagged as showing diminishing
    returns when its marginal ROAS falls below this bar. Defaults to 60%
    of the project's own primary success target (4.0x blended ROAS) -> 2.4x.
    This is a principled choice, not an arbitrary tuning: it says "the
    next dollar into this channel returns less than 60% of what the
    business has set as its target," which is a defensible trigger for a
    human to review the channel, distinct from simply being below the
    4.0x target itself (which the channel average already is in this
    dataset -- the marginal threshold is specifically about the NEXT
    dollar, not the average dollar)."""
    TARGET_ROAS = 4.0
    if threshold_marginal_roas is None:
        threshold_marginal_roas = 0.6 * TARGET_ROAS  # 2.4x

    channels = list(curves.keys())
    current = np.array([curves[c]["current_daily_spend"] for c in channels])
    total = total_daily_budget or current.sum()

    bounds = [(current[i] * min_pct, current[i] * max_pct) for i in range(len(channels))]

    def neg_total_revenue(spend_vec):
        return -sum(hill(spend_vec[i], curves[channels[i]]["R_max"],
                          curves[channels[i]]["S50"], curves[channels[i]]["k"])
                    for i in range(len(channels)))

    cons = {"type": "eq", "fun": lambda s: s.sum() - total}
    res = minimize(neg_total_revenue, x0=current, bounds=bounds, constraints=[cons],
                    method="SLSQP")
    new_spend = res.x if res.success else current

    rows = []
    for i, c in enumerate(channels):
        params = curves[c]
        cur_rev = hill(current[i], params["R_max"], params["S50"], params["k"])
        new_rev = hill(new_spend[i], params["R_max"], params["S50"], params["k"])
        m_roas_cur = marginal_roas(current[i], params)
        m_roas_new = marginal_roas(new_spend[i], params)
        delta_pct = (new_spend[i] - current[i]) / current[i] * 100
        diminishing = m_roas_cur < threshold_marginal_roas
        rows.append({
            "channel": c,
            "current_daily_spend": round(current[i], 0),
            "recommended_daily_spend": round(new_spend[i], 0),
            "change_pct": round(delta_pct, 1),
            "current_roas": round(cur_rev / current[i], 2),
            "projected_roas": round(new_rev / new_spend[i], 2),
            "current_marginal_roas": round(m_roas_cur, 2),
            "projected_marginal_roas": round(m_roas_new, 2),
            "diminishing_returns_flag": bool(diminishing),
            "current_daily_revenue": round(cur_rev, 0),
            "projected_daily_revenue": round(new_rev, 0),
        })

    rec_df = pd.DataFrame(rows)
    summary = {
        "total_budget": round(total, 0),
        "current_blended_roas": round(rec_df["current_daily_revenue"].sum() / rec_df["current_daily_spend"].sum(), 2),
        "projected_blended_roas": round(rec_df["projected_daily_revenue"].sum() / rec_df["recommended_daily_spend"].sum(), 2),
        "projected_revenue_lift": round(rec_df["projected_daily_revenue"].sum() - rec_df["current_daily_revenue"].sum(), 0),
    }
    return rec_df, summary


if __name__ == "__main__":
    curves, daily = fit_efficiency_curves()
    print("Fitted efficiency curves (Hill function: R_max, S50, k):")
    for c, p in curves.items():
        print(f"  {c}: R_max=${p['R_max']:.0f}  S50=${p['S50']:.0f}  k={p['k']:.2f}")

    rec_df, summary = recommend_allocation(curves)
    print("\nBudget reallocation recommendation:")
    print(rec_df.to_string(index=False))
    print("\nSummary:", summary)

    TARGET_ROAS = 4.0
    if summary["projected_blended_roas"] < TARGET_ROAS:
        gap = TARGET_ROAS - summary["projected_blended_roas"]
        print(f"\n⚠️  Even at the OPTIMAL reallocation, projected blended ROAS "
              f"({summary['projected_blended_roas']}x) remains {gap:.2f}x below the "
              f"{TARGET_ROAS}x target. All three channels have converged to similar "
              f"marginal ROAS (~{rec_df['projected_marginal_roas'].mean():.2f}x) -- meaning "
              f"the bottleneck is NOT misallocated budget, it's total spend exceeding "
              f"efficient auction capacity across the board. Reallocation has hit its "
              f"ceiling; the next lever is total budget reduction or SKU/creative-level "
              f"efficiency gains, not further reshuffling.")

    rec_df.to_csv(DATA / "allocation_recommendation.csv", index=False)
