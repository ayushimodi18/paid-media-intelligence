"""
Test suite. Run with: pytest tests/ -v

Covers the logic most likely to silently break and corrupt downstream
numbers if it regresses: reconciliation correctness, feature-engineering
leakage, validation catching bad data, and optimizer constraint compliance.
Uses small synthetic fixtures built in-test, NOT the full generated
dataset -- fast, deterministic, and isolates exactly what's being tested.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import ingest_reconcile as ir
import feature_engineering as fe
import validation as val
import optimize_allocation as oa
import budget_utilization as bu
import nc_cpa
import llm_narrator
import business_impact as bi


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def tiny_meta_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "platform_sku_id": ["FB_SKU_TEE_BLK_M", "FB_SKU_UNKNOWN_SKU"],
        "true_sku": ["SKU-TEE-BLK-M", "SKU-UNKNOWN-SKU"],
        "channel": ["Meta", "Meta"],
        "spend": [100.0, 50.0],
        "impressions": [5000, 2000],
        "clicks": [100, 40],
        "revenue_attributed": [400.0, np.nan],
        "new_customer_pct": [0.4, 0.5],
        "sku_tier": ["hero", "tail"],
        "campaign_objective": ["Prospecting", "Retargeting"],
        "purchases_7d_click_1d_view": [10, 5],
        "is_preliminary": [False, False],
    })


# ---------------------------------------------------------------- M1: reconciliation
class TestReconciliation:
    def test_meta_known_sku_reconciles(self, tiny_meta_df):
        out = ir.reconcile_meta(tiny_meta_df)
        assert out.loc[0, "reconciled"] == True
        assert out.loc[0, "sku"] == "SKU-TEE-BLK-M"

    def test_google_hyphen_collision_is_reproduced(self):
        """The T-Shirt hyphen-collision bug is an intentional, documented
        demo artifact -- this test locks in that it still happens, so a
        future refactor doesn't accidentally 'fix' it without us noticing
        (which would silently change the EDA/README narrative)."""
        df = pd.DataFrame({
            "platform_sku_id": ["goog::t-shirt-black-men", "goog::polo-navy-men"],
        })
        out = ir.reconcile_google(df)
        assert out.loc[0, "reconciled"] == False  # T-Shirt hyphen breaks this one
        assert out.loc[1, "reconciled"] == True    # Polo (no hyphen) works fine
        assert out.loc[1, "sku"] == "SKU-POLO-NVY-M"

    def test_unify_produces_no_null_revenue(self):
        """After unify(), revenue_attributed must have zero nulls -- the
        whole point of the imputation step."""
        report, unified = ir.unify()
        assert unified["revenue_attributed"].isna().sum() == 0

    def test_unify_only_contains_canonical_skus(self):
        report, unified = ir.unify()
        assert set(unified["sku"]).issubset(set(ir.CANON.keys()))


# ---------------------------------------------------------------- feature engineering
class TestFeatureEngineering:
    def test_lag_features_dont_leak_future(self):
        """A lag-1 feature on day N must equal the raw value from day N-1,
        not day N or day N+1 -- this is the single most important thing to
        get right in time-series feature engineering, since a leak here
        would make every backtest number in the project meaningless."""
        df = pd.DataFrame({
            "channel": ["Meta"] * 5,
            "date": pd.date_range("2026-01-01", periods=5),
            "spend": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        out = fe.add_lag_and_rolling_features(df, group_cols=["channel"], value_col="spend", lags=(1,), windows=())
        # row for day 3 (index 2, value 30) should have lag1 == day 2's value (20)
        assert out.loc[2, "spend_lag1"] == 20.0
        assert out.loc[3, "spend_lag1"] == 30.0

    def test_rolling_mean_excludes_current_day(self):
        """rolling().mean() must be computed on shift(1) -- i.e. the
        current day's own spend must NOT be included in its own rolling
        average, or the model would be training on a feature that
        partially encodes the target."""
        df = pd.DataFrame({
            "channel": ["Meta"] * 4,
            "date": pd.date_range("2026-01-01", periods=4),
            "spend": [10.0, 10.0, 10.0, 100.0],  # big spike on day 4
        })
        out = fe.add_lag_and_rolling_features(df, group_cols=["channel"], value_col="spend",
                                                lags=(), windows=(3,))
        # day 4's roll3 mean should be the mean of days 1-3 (10,10,10=10), NOT including day 4's 100
        assert out.loc[3, "spend_roll3_mean"] == pytest.approx(10.0)

    def test_outlier_capping_preserves_row_count(self):
        df = pd.DataFrame({"spend": list(range(100)) + [10000]})  # one extreme outlier
        capped_df, n_capped = fe.cap_outliers(df, "spend")
        assert len(capped_df) == len(df)  # no rows dropped
        assert n_capped >= 1
        assert capped_df["spend_capped"].max() < 10000  # the outlier was capped, not just flagged


# ---------------------------------------------------------------- validation
class TestValidation:
    def test_negative_spend_fails_validation(self, tiny_meta_df):
        bad = tiny_meta_df.copy()
        bad.loc[0, "spend"] = -50.0
        report = val.validate_raw_export(bad, "test_negative_spend")
        assert report.passed == False

    def test_duplicate_rows_flagged(self, tiny_meta_df):
        dup = pd.concat([tiny_meta_df, tiny_meta_df.iloc[[0]]], ignore_index=True)
        report = val.validate_raw_export(dup, "test_dupes")
        assert report.passed == False
        assert any("duplicate" in e.lower() for e in report.errors)

    def test_clean_data_passes(self, tiny_meta_df):
        report = val.validate_raw_export(tiny_meta_df, "test_clean")
        assert report.passed == True


# ---------------------------------------------------------------- optimizer
class TestOptimizer:
    def test_recommended_spend_respects_bounds(self):
        curves = {
            "Meta": {"R_max": 9000, "S50": 1400, "k": 1.15, "avg_new_cust_pct": 0.4, "current_daily_spend": 1000},
            "Google": {"R_max": 7500, "S50": 1100, "k": 1.05, "avg_new_cust_pct": 0.3, "current_daily_spend": 800},
        }
        rec_df, summary = oa.recommend_allocation(curves, min_pct=0.6, max_pct=1.6)
        for _, row in rec_df.iterrows():
            current = curves[row["channel"]]["current_daily_spend"]
            assert row["recommended_daily_spend"] >= 0.6 * current - 1e-6
            assert row["recommended_daily_spend"] <= 1.6 * current + 1e-6

    def test_recommended_spend_conserves_total_budget(self):
        curves = {
            "Meta": {"R_max": 9000, "S50": 1400, "k": 1.15, "avg_new_cust_pct": 0.4, "current_daily_spend": 1000},
            "Google": {"R_max": 7500, "S50": 1100, "k": 1.05, "avg_new_cust_pct": 0.3, "current_daily_spend": 800},
        }
        rec_df, summary = oa.recommend_allocation(curves)
        total_current = sum(c["current_daily_spend"] for c in curves.values())
        assert rec_df["recommended_daily_spend"].sum() == pytest.approx(total_current, rel=1e-3)

    def test_hill_function_is_nondecreasing_in_spend(self):
        spends = np.linspace(1, 5000, 50)
        revenues = oa.hill(spends, R_max=9000, S50=1400, k=1.15)
        assert np.all(np.diff(revenues) >= 0)  # more spend never decreases predicted revenue

    def test_diminishing_returns_flag_actually_fires_at_realistic_marginal_roas(self):
        """Regression guard: an earlier version hardcoded the threshold at
        1.0x marginal ROAS, which never fired against this dataset's
        realistic ~2.2-2.5x marginal ROAS levels -- meaning the project's
        explicit 'show what your system does when marginal ROAS drops
        below threshold' requirement was never actually demonstrated live.
        This locks in that the default threshold (60% of the 4.0x target)
        does fire for a channel with marginal ROAS clearly below it."""
        curves = {
            "Amazon": {"R_max": 70000, "S50": 16000, "k": 1.05, "avg_new_cust_pct": 0.4, "current_daily_spend": 10000},
        }
        rec_df, _ = oa.recommend_allocation(curves)
        # this fixture's marginal ROAS at $10,000 spend should sit below the 2.4x default threshold
        assert rec_df.loc[0, "current_marginal_roas"] < 2.4
        assert rec_df.loc[0, "diminishing_returns_flag"] == True


# ---------------------------------------------------------------- budget utilization
class TestBudgetUtilization:
    def test_flags_are_mutually_sensible(self):
        """A row can't simultaneously be both 'budget waste' (near cap +
        below-median ROAS) and 'efficient utilization' (near cap +
        above-median ROAS) -- these are mutually exclusive by construction."""
        df = pd.DataFrame({
            "channel": ["Meta"] * 20,
            "sku": ["SKU-TEE-BLK-M"] * 20,
            "spend": list(np.linspace(50, 500, 20)),
            "roas": list(np.linspace(1.0, 6.0, 20)),
        })
        flagged = bu.flag_waste_and_underspend(df)
        overlap = flagged["flag_budget_waste"] & flagged["flag_efficient_utilization"]
        assert overlap.sum() == 0

    def test_ceiling_uses_p90_by_default(self):
        df = pd.DataFrame({
            "channel": ["Meta"] * 10,
            "sku": ["SKU-TEE-BLK-M"] * 10,
            "spend": list(range(1, 11)),  # 1..10
        })
        out = bu.compute_budget_ceiling(df)
        expected_p90 = pd.Series(range(1, 11)).quantile(0.90)
        assert out["spend_ceiling_p90"].iloc[0] == pytest.approx(expected_p90)


# ---------------------------------------------------------------- NC-CPA
class TestNcCpa:
    def test_nc_cpa_formula_is_correct(self):
        """NC-CPA = new-customer spend / new-customer conversions, where
        each row's spend/conversions are weighted by that row's own
        new_customer_pct -- verify against a hand-computed example."""
        df = pd.DataFrame({
            "channel": ["Meta", "Meta"],
            "spend": [100.0, 200.0],
            "conversions": [10, 20],
            "new_customer_pct": [0.5, 0.25],
        })
        out = nc_cpa.compute_nc_cpa(df, group_cols=("channel",))
        # new_customer_spend = 100*0.5 + 200*0.25 = 50 + 50 = 100
        # new_customer_conversions = 10*0.5 + 20*0.25 = 5 + 5 = 10
        # nc_cpa = 100 / 10 = 10.0
        assert out.loc[0, "nc_cpa"] == pytest.approx(10.0)

    def test_within_target_flag_is_consistent_with_value(self):
        df = pd.DataFrame({
            "channel": ["Meta"], "spend": [1000.0], "conversions": [10], "new_customer_pct": [1.0],
        })
        out = nc_cpa.compute_nc_cpa(df, group_cols=("channel",))
        expected_within = out.loc[0, "nc_cpa"] <= nc_cpa.NC_CPA_TARGET
        assert out.loc[0, "within_target"] == expected_within


# ---------------------------------------------------------------- LLM narrator
class TestLlmNarrator:
    def test_no_provider_available_uses_template_fallback(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Simulate "no local model" too, so the test doesn't depend on whether
        # Ollama happens to be running on this machine.
        monkeypatch.setattr(llm_narrator, "_try_ollama", lambda prompt: None)
        row = {"channel": "Meta", "current_daily_spend": 1000.0, "recommended_daily_spend": 1100.0,
               "change_pct": 10.0, "current_roas": 3.0, "projected_roas": 3.1,
               "current_marginal_roas": 2.5, "projected_marginal_roas": 2.4,
               "diminishing_returns_flag": False}
        summary = {"current_blended_roas": 2.9, "projected_blended_roas": 2.95}
        result = llm_narrator.generate_recommendation_narrative(row, summary)
        assert result.provider == "template"
        assert result.used_llm == False
        assert "1000" in result.text.replace(",", "")  # template references the real spend figure

    # Shared fixture for the groundedness tests -- deterministic, no live LLM call.
    ROW = {"channel": "Amazon", "current_daily_spend": 4974.0, "recommended_daily_spend": 4325.0,
           "change_pct": -13.04, "current_roas": 2.614, "projected_roas": 2.661,
           "current_marginal_roas": 2.23, "projected_marginal_roas": 2.34}
    SUMMARY = {"current_blended_roas": 2.91, "projected_blended_roas": 2.95}

    def _ungrounded(self, text):
        return llm_narrator._ungrounded_numbers(text, self.ROW, self.SUMMARY)

    def test_groundedness_check_rejects_fabricated_numbers(self):
        assert 99.9 in self._ungrounded("ROAS could reach 99.9x")
        assert 3.5 in self._ungrounded("lifting ROAS to 3.5x")

    def test_groundedness_rejects_near_miss_that_old_tolerance_allowed(self):
        """2.74 is within 0.15 of the real 2.661 -- the previous fuzzy check
        accepted it. Precision-aware matching must not."""
        assert 2.74 in self._ungrounded("projected ROAS of 2.74x")

    def test_groundedness_accepts_real_numbers_and_honest_rounding(self):
        text = ("Cut Amazon spend 13% from $4,974 to $4,325; ROAS moves from 2.61x to 2.66x "
                "(about 2.7x), with marginal ROAS 2.23x -> 2.34x and blended ROAS 2.95x.")
        assert self._ungrounded(text) == []

    def test_groundedness_accepts_derived_spend_change(self):
        assert self._ungrounded("a $649 reduction in daily spend") == []
        assert 650.0 in self._ungrounded("a $650 reduction in daily spend")

    def test_ungrounded_llm_output_falls_back_to_template(self, monkeypatch):
        monkeypatch.setattr(llm_narrator, "_try_ollama", lambda prompt: "ROAS will hit 9.9x.")
        result = llm_narrator.generate_recommendation_narrative(self.ROW, self.SUMMARY)
        assert result.provider == "template"
        assert result.groundedness_check_passed is False

    def test_grounded_llm_output_is_used(self, monkeypatch):
        monkeypatch.setattr(llm_narrator, "_try_ollama",
                            lambda prompt: "Reduce Amazon spend by 13.0% to $4,325 per day.")
        result = llm_narrator.generate_recommendation_narrative(self.ROW, self.SUMMARY)
        assert result.provider == "ollama" and result.used_llm


# ---------------------------------------------------------------- business impact
class TestBusinessImpact:
    def test_dio_impact_only_counts_hero_tier(self):
        flags = pd.DataFrame({
            "sku": ["SKU-TEE-BLK-M", "SKU-SHORT-KHK-M"],  # hero, tail
            "spend": [100.0, 50.0],
            "spend_ceiling_p90": [300.0, 150.0],
            "flag_underspend_opportunity": [True, True],
        })
        import unittest.mock as mock
        with mock.patch("pandas.read_csv") as mock_read:
            mock_read.return_value = pd.DataFrame({
                "true_sku": ["SKU-TEE-BLK-M", "SKU-SHORT-KHK-M"],
                "sku_tier": ["hero", "tail"],
            })
            result = bi.estimate_dio_impact(flags)
        # only the hero row's headroom (300-100=200) should count, not tail's (150-50=100)
        assert result["hero_unused_efficient_capacity_dollars"] == pytest.approx(200.0)

    def test_fulfillment_impact_waste_pct_is_bounded(self):
        flags = pd.DataFrame({
            "spend": [100.0, 100.0, 100.0],
            "flag_budget_waste": [True, False, False],
        })
        result = bi.estimate_fulfillment_impact(flags)
        assert 0 <= result["waste_as_pct_of_total_spend"] <= 100
        assert result["total_flagged_waste_dollars"] == pytest.approx(100.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
