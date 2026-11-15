"""
Secondary success metric - New Customer Acquisition Cost (NC-CPA).

Requirement: "Your system must distinguish prospecting from retargeting
spend and keep new-customer CPA within ceiling -- without starving the
top of funnel."

This is computed from the RAW exports, not the unified table, because the
unified table aggregates away campaign_objective/campaign_type/ad_type --
exactly the dimension needed to separate prospecting (new-customer-heavy)
from retargeting (existing-customer-heavy) spend. NC-CPA without that
split would just be blended CPA wearing a different name.

Methodology: for each row, new_customer_pct tells us what share of that
row's conversions came from new customers. We apply that same share to
spend as a proportional allocation (spend isn't independently split by
customer type in any platform export -- this is the standard approximation
used when a platform doesn't report new-customer spend directly, only
new-customer conversion share):

    new_customer_spend       = spend * new_customer_pct
    new_customer_conversions = conversions * new_customer_pct
    NC-CPA                   = sum(new_customer_spend) / sum(new_customer_conversions)

Target ceiling: $45, anchored to the cited 2026 benchmark range (Meta
apparel CPA ~$31, ecommerce ~$30-38, Google ecommerce CPA ~$53 cross-
industry but lower for retail) -- $45 sits inside that range as a
blended cross-channel ceiling. Documented here, not asserted silently.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
NC_CPA_TARGET = 45.0

CONVERSION_COL = {
    "Meta": "purchases_7d_click_1d_view",
    "Google": "conversions_ddattr",
    "Amazon": "orders",
}
TYPE_COL = {
    "Meta": "campaign_objective",
    "Google": "campaign_type",
    "Amazon": "ad_type",
}
# which type values are "prospecting-like" (acquiring new customers) per channel
PROSPECTING_VALUES = {
    "Meta": {"Prospecting", "DPA"},
    "Google": {"Search", "PMax", "Demand Gen"},
    "Amazon": {"Sponsored Products", "Sponsored Brands"},
}


def load_raw(channel: str) -> pd.DataFrame:
    fname = {"Meta": "meta_ads_export.csv", "Google": "google_ads_export.csv",
             "Amazon": "amazon_ads_export.csv"}[channel]
    df = pd.read_csv(DATA / fname, parse_dates=["date"])
    df["channel"] = channel
    df["conversions"] = df[CONVERSION_COL[channel]]
    df["campaign_type_generic"] = df[TYPE_COL[channel]]
    df["is_prospecting"] = df["campaign_type_generic"].isin(PROSPECTING_VALUES[channel])
    return df[["date", "channel", "true_sku", "spend", "conversions",
               "new_customer_pct", "campaign_type_generic", "is_prospecting"]]


def compute_nc_cpa(df: pd.DataFrame, group_cols=("channel",)) -> pd.DataFrame:
    d = df.copy()
    d["new_customer_spend"] = d["spend"] * d["new_customer_pct"]
    d["new_customer_conversions"] = d["conversions"] * d["new_customer_pct"]

    grouped = d.groupby(list(group_cols)).agg(
        total_spend=("spend", "sum"),
        new_customer_spend=("new_customer_spend", "sum"),
        new_customer_conversions=("new_customer_conversions", "sum"),
        total_conversions=("conversions", "sum"),
    ).reset_index()
    grouped["nc_cpa"] = (grouped["new_customer_spend"] / grouped["new_customer_conversions"]).round(2)
    grouped["blended_cpa"] = (grouped["total_spend"] / grouped["total_conversions"]).round(2)
    grouped["nc_cpa_vs_target"] = grouped["nc_cpa"] - NC_CPA_TARGET
    grouped["within_target"] = grouped["nc_cpa"] <= NC_CPA_TARGET
    return grouped


def compute_nc_cpa_by_funnel_stage(df: pd.DataFrame) -> pd.DataFrame:
    """The actual requirement: split prospecting vs retargeting explicitly,
    not just blend by new_customer_pct -- this answers 'is prospecting
    spend specifically within NC-CPA ceiling' rather than a channel
    average that could hide a starved top-of-funnel."""
    return compute_nc_cpa(df, group_cols=("channel", "is_prospecting"))


def run_all_channels() -> pd.DataFrame:
    all_raw = pd.concat([load_raw(c) for c in ["Meta", "Google", "Amazon"]], ignore_index=True)
    by_channel = compute_nc_cpa(all_raw, group_cols=("channel",))
    by_channel["cut"] = "channel_blended"
    by_funnel = compute_nc_cpa_by_funnel_stage(all_raw)
    by_funnel["cut"] = "prospecting_vs_retargeting"
    by_funnel = by_funnel.rename(columns={"is_prospecting": "segment"})
    by_funnel["segment"] = by_funnel["segment"].map({True: "Prospecting-type", False: "Retargeting-type"})
    by_channel["segment"] = "All"

    cols = ["channel", "segment", "cut", "nc_cpa", "blended_cpa", "nc_cpa_vs_target", "within_target",
            "new_customer_conversions", "total_spend"]
    combined = pd.concat([by_channel[cols], by_funnel[cols]], ignore_index=True)
    return combined


if __name__ == "__main__":
    result = run_all_channels()
    print(f"NC-CPA target ceiling: ${NC_CPA_TARGET}\n")
    print(result.to_string(index=False))

    overall_fail = result[(result["cut"] == "channel_blended") & (~result["within_target"])]
    if len(overall_fail):
        print(f"\n⚠️  {len(overall_fail)} channel(s) exceed the NC-CPA ceiling: "
              f"{', '.join(overall_fail['channel'].tolist())}")
    else:
        print("\n✅ All channels within NC-CPA ceiling on a blended basis.")

    prospecting_fail = result[(result["cut"] == "prospecting_vs_retargeting") &
                               (result["segment"] == "Prospecting-type") & (~result["within_target"])]
    if len(prospecting_fail):
        print(f"⚠️  Prospecting-specific NC-CPA exceeds ceiling on: "
              f"{', '.join(prospecting_fail['channel'].tolist())} -- this is the metric that matters most, "
              f"since it's the one a blended channel average could hide.")

    result.to_csv(DATA / "nc_cpa_summary.csv", index=False)
    print(f"\nSaved to {DATA / 'nc_cpa_summary.csv'}")
