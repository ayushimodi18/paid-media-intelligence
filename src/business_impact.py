"""
Business impact translation layer -- DIO, stockout risk, fulfillment cost.

Requirement: "Every technical decision should connect back to DIO, stockout
losses, or fulfillment cost. We'll ask you to map it."

This module is deliberately NOT a new forecasting/optimization system --
it translates the REAL outputs already computed by budget_utilization.py
into inventory-economics language, grounded in cited 2026 benchmarks for
DTC apparel. It is explicitly directional/illustrative, not a precise
financial model -- that distinction is stated in every output, not
hidden behind false precision.

BENCHMARKS USED (compiled July 2026):
  - DTC apparel DIO: 50-70 days typical range (Descartes Finale 2026 DIO
    guide); broader DTC median across public 10-Ks is higher (133 days,
    Eightx 2026) but skews toward beauty/CPG categories with longer
    inbound supply chains -- apparel-specific range is the more relevant
    anchor here.
  - Apparel inventory turn target: 6-8x/year (GoBolt 2026 DTC fulfillment guide)
  - Apparel return rate: 20-30% of orders (GoBolt 2026)
  - DTC shipping cost: 7-12% of revenue (GoBolt 2026)
  - Returns processing cost lever: 24h vs 72h triage = 30-50% lower cost
    per return; ~$1.50-3.00/order savings at a 25% return rate, $60 AOV
    (fforder.com 2026 DTC fulfillment guide)
  - Real-world case reference: a retail apparel brand cut DIO from 130 to
    95 days via size-curation and demand-responsive markdowns (Verve AI
    2026 DIO benchmarks) -- cited as a directional reference point for
    "what a meaningful DIO improvement looks like," not as a prediction
    of this brand's result.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

APPAREL_DIO_RANGE = (50, 70)        # days
APPAREL_TURN_TARGET = (6, 8)        # turns/year
APPAREL_RETURN_RATE_RANGE = (0.20, 0.30)
SHIPPING_PCT_OF_REVENUE_RANGE = (0.07, 0.12)
CASE_STUDY_DIO_IMPROVEMENT_DAYS = 35  # 130 -> 95, cited reference point


def load_flags() -> pd.DataFrame:
    return pd.read_csv(DATA / "budget_utilization_flags.csv", parse_dates=["date"])


def estimate_dio_impact(flags: pd.DataFrame = None) -> dict:
    """Connects the underspend-opportunity flag to inventory velocity.
    Logic: underspend-flagged days are, by construction, days where a
    SKU had above-median ROAS but spend well below its own ceiling --
    i.e. demonstrated headroom to convert MORE of that SKU's inventory
    into revenue per dollar than is currently happening. Hero-tier SKUs
    carry the most weight here because they're the ones a DIO
    conversation is actually about (tail SKUs are typically a smaller
    share of inventory dollars to begin with)."""
    flags = flags if flags is not None else load_flags()

    hero_skus = pd.read_csv(DATA / "meta_ads_export.csv")[["true_sku", "sku_tier"]].drop_duplicates()
    hero_skus = hero_skus.rename(columns={"true_sku": "sku"})
    f = flags.merge(hero_skus, on="sku", how="left")

    hero_underspend = f[(f["sku_tier"] == "hero") & (f["flag_underspend_opportunity"])]
    hero_headroom_dollars = (hero_underspend["spend_ceiling_p90"] - hero_underspend["spend"]).sum()
    hero_total_spend = f[f["sku_tier"] == "hero"]["spend"].sum()
    headroom_pct_of_hero_spend = hero_headroom_dollars / hero_total_spend if hero_total_spend else 0

    return {
        "hero_underspend_flagged_days": int(len(hero_underspend)),
        "hero_unused_efficient_capacity_dollars": round(hero_headroom_dollars, 0),
        "headroom_as_pct_of_hero_spend": round(headroom_pct_of_hero_spend * 100, 1),
        "apparel_dio_range_days": APPAREL_DIO_RANGE,
        "apparel_turn_target_per_year": APPAREL_TURN_TARGET,
        "directional_read": (
            f"Hero-tier SKUs show unused efficient ad capacity of "
            f"${hero_headroom_dollars:,.0f} across {len(hero_underspend):,} flagged days "
            f"({headroom_pct_of_hero_spend*100:.1f}% of hero spend). Hero SKUs are, by definition, "
            f"the fastest-turning inventory in the catalog -- under-marketing them on days with "
            f"demonstrated above-median ROAS leaves sell-through velocity on the table. Apparel "
            f"DIO typically runs {APPAREL_DIO_RANGE[0]}-{APPAREL_DIO_RANGE[1]} days against a "
            f"{APPAREL_TURN_TARGET[0]}-{APPAREL_TURN_TARGET[1]}x/year turn target -- closing even a "
            f"modest share of this headroom is the kind of lever that has driven double-digit-day "
            f"DIO improvements elsewhere in apparel (one cited case: 130 -> 95 days via demand-"
            f"responsive merchandising). This is a DIRECTIONAL read, not a precise forecast -- "
            f"converting ad spend into a DIO-day estimate would need actual inventory and COGS data, "
            f"which this prototype does not have."
        ),
    }


def estimate_fulfillment_impact(flags: pd.DataFrame = None) -> dict:
    """Connects the budget-waste flag to fulfillment/working-capital
    opportunity cost. Logic: dollars spent at near-cap with below-median
    ROAS are dollars NOT available for fulfillment investment (returns
    processing SLA, inventory accuracy, shipping cost negotiation) --
    framed as opportunity cost, not a literal 1:1 reallocation claim."""
    flags = flags if flags is not None else load_flags()
    waste_dollars = flags.loc[flags["flag_budget_waste"], "spend"].sum()
    total_spend = flags["spend"].sum()
    waste_pct = waste_dollars / total_spend if total_spend else 0

    return {
        "total_flagged_waste_dollars": round(waste_dollars, 0),
        "waste_as_pct_of_total_spend": round(waste_pct * 100, 2),
        "apparel_return_rate_range": APPAREL_RETURN_RATE_RANGE,
        "shipping_pct_of_revenue_range": SHIPPING_PCT_OF_REVENUE_RANGE,
        "directional_read": (
            f"${waste_dollars:,.0f} ({waste_pct*100:.1f}% of total spend) is flagged as budget waste "
            f"(near-cap spend with below-median ROAS). Apparel DTC returns run "
            f"{APPAREL_RETURN_RATE_RANGE[0]*100:.0f}-{APPAREL_RETURN_RATE_RANGE[1]*100:.0f}% of orders "
            f"and shipping costs run {SHIPPING_PCT_OF_REVENUE_RANGE[0]*100:.0f}-"
            f"{SHIPPING_PCT_OF_REVENUE_RANGE[1]*100:.0f}% of revenue -- this flagged waste is framed as "
            f"opportunity cost: cash currently buying inefficient ad placements that could instead "
            f"fund fulfillment-side improvements (faster returns triage has been shown to cut "
            f"per-return cost 30-50%) or simply reduce reliance on supplier payment terms. This is "
            f"an opportunity-cost framing, not a claim that this exact dollar amount would convert "
            f"1:1 into fulfillment savings."
        ),
    }


if __name__ == "__main__":
    flags = load_flags()
    dio = estimate_dio_impact(flags)
    fulfillment = estimate_fulfillment_impact(flags)

    print("=== DIO / Inventory Velocity Impact ===")
    for k, v in dio.items():
        if k != "directional_read":
            print(f"  {k}: {v}")
    print(f"\n{dio['directional_read']}\n")

    print("=== Fulfillment Cost / Working Capital Impact ===")
    for k, v in fulfillment.items():
        if k != "directional_read":
            print(f"  {k}: {v}")
    print(f"\n{fulfillment['directional_read']}")
