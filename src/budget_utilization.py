"""
M3 extension - Budget Waste & Underspend Detector.

Addresses the project's TERTIARY success metric directly: "Efficient campaigns
that hit daily budget caps by noon are leaving revenue on the table.
Inefficient campaigns burning full budgets are wasting cash."

This dataset doesn't have hourly pacing data (no platform export includes
that at the SKU level we're working with), so true "hit the cap by noon"
detection isn't possible here -- that's stated plainly, not faked. What IS
detectable from daily data: a proxy signal built from two real, available
diagnostics:
  1. UNDERSPEND signal: SKU-channel days where ROAS is high (efficient)
     AND spend is well below that SKU-channel's own typical ceiling --
     i.e. there's a track record of better economics this campaign isn't
     fully using its budget to capture.
  2. WASTE signal: SKU-channel days where spend is at/near that SKU-
     channel's typical ceiling AND ROAS is below the channel median --
     i.e. it's burning a full budget on poor economics.

Both are FLAGS for a human to investigate, not auto-corrections -- exactly
the "human in the loop" the requirements ask for.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def compute_budget_ceiling(df: pd.DataFrame, group_cols=("channel", "sku"), pctl=0.90) -> pd.DataFrame:
    """A SKU-channel's 'typical ceiling' = its own 90th percentile historical
    daily spend -- a data-driven stand-in for 'the daily budget cap', since
    we don't have the platform's actual configured cap value."""
    ceilings = df.groupby(list(group_cols))["spend"].quantile(pctl).reset_index()
    ceilings = ceilings.rename(columns={"spend": "spend_ceiling_p90"})
    return df.merge(ceilings, on=list(group_cols), how="left")


def flag_waste_and_underspend(df: pd.DataFrame, near_cap_frac=0.90, low_spend_frac=0.50) -> pd.DataFrame:
    df = compute_budget_ceiling(df)
    channel_median_roas = df.groupby("channel")["roas"].transform("median")

    df["near_cap"] = df["spend"] >= near_cap_frac * df["spend_ceiling_p90"]
    df["low_spend_vs_ceiling"] = df["spend"] <= low_spend_frac * df["spend_ceiling_p90"]
    df["above_median_roas"] = df["roas"] >= channel_median_roas
    df["below_median_roas"] = df["roas"] < channel_median_roas

    df["flag_underspend_opportunity"] = df["low_spend_vs_ceiling"] & df["above_median_roas"]
    df["flag_budget_waste"] = df["near_cap"] & df["below_median_roas"]
    df["flag_efficient_utilization"] = df["near_cap"] & df["above_median_roas"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["channel", "sku"]).agg(
        days_observed=("spend", "size"),
        underspend_days=("flag_underspend_opportunity", "sum"),
        waste_days=("flag_budget_waste", "sum"),
        efficient_days=("flag_efficient_utilization", "sum"),
        avg_roas=("roas", "mean"),
        avg_spend=("spend", "mean"),
        spend_ceiling_p90=("spend_ceiling_p90", "first"),
    ).reset_index()
    out["underspend_rate"] = (out["underspend_days"] / out["days_observed"]).round(3)
    out["waste_rate"] = (out["waste_days"] / out["days_observed"]).round(3)
    return out.sort_values("waste_rate", ascending=False)


if __name__ == "__main__":
    unified = pd.read_csv(DATA / "unified_performance.csv", parse_dates=["date"])
    flagged = flag_waste_and_underspend(unified)
    summary = summarize(flagged)

    print("Top 10 SKU-channel combos by budget WASTE rate (near cap + below-median ROAS):")
    print(summary.head(10).to_string(index=False))
    print("\nTop 10 SKU-channel combos by UNDERSPEND OPPORTUNITY rate (low spend + above-median ROAS):")
    print(summary.sort_values("underspend_rate", ascending=False).head(10).to_string(index=False))

    overall_waste_days = flagged["flag_budget_waste"].sum()
    overall_underspend_days = flagged["flag_underspend_opportunity"].sum()
    overall_efficient_days = flagged["flag_efficient_utilization"].sum()
    total_days = len(flagged)
    print(f"\nAcross {total_days:,} SKU-channel-days:")
    print(f"  Budget waste flagged on {overall_waste_days:,} days ({overall_waste_days/total_days:.1%})")
    print(f"  Underspend opportunity flagged on {overall_underspend_days:,} days ({overall_underspend_days/total_days:.1%})")
    print(f"  Efficient utilization on {overall_efficient_days:,} days ({overall_efficient_days/total_days:.1%})")

    summary.to_csv(DATA / "budget_utilization_summary.csv", index=False)
    flagged.to_csv(DATA / "budget_utilization_flags.csv", index=False)
    print(f"\nSaved to {DATA / 'budget_utilization_summary.csv'} and budget_utilization_flags.csv")
