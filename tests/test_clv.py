"""Tests for clv_model.py. Run with: python -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clv_model as clv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "transactions.csv"


def config(**changes) -> dict:
    settings = clv.load_config(None)
    settings.update(changes)
    return settings


def orders(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["customer_id", "order_date", "order_value"])
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    return frame


@pytest.fixture(scope="module")
def sample_orders() -> pd.DataFrame:
    return clv.clean_transactions(clv.read_transactions(SAMPLE), config())[0]


def test_bad_rows_are_rejected_with_a_reason_not_dropped():
    raw = pd.DataFrame({"order_id": ["1", "2", "2", "3", "4", "5", "6"],
                        "customer_id": ["A", "A", "A", "", "B", "B", "C"],
                        "order_date": ["2025-01-05", "2025-02-01", "2025-02-01", "2025-02-02", "no date",
                                       "2025-03-01", "2025-03-02"],
                        "order_value": ["Rs 1,200", "800", "800", "500", "400", "-250", "abc"]})
    clean, rejected, quality = clv.clean_transactions(raw, config())
    assert clean["order_value"].tolist() == [1200.0, 800.0]              # "Rs 1,200" is read as 1200
    assert rejected["reason"].tolist() == ["Duplicate row", "Missing customer id", "Unreadable order date",
                                           "Order value zero or negative (refund or cancellation)",
                                           "Order value is not a number"]
    assert len(clean) + len(rejected) == len(raw)                        # every row is accounted for
    assert quality.set_index("Check")["Rows"]["Clean orders used"] == 2


def test_features_match_a_hand_calculation():
    history = orders([("A", "2025-01-01", 1000), ("A", "2025-03-02", 2000), ("A", "2025-05-01", 3000),
                      ("B", "2023-06-01", 500)])
    row = clv.build_features(history, pd.Timestamp("2025-07-01")).loc["A"]
    assert row["frequency"] == 3 and row["monetary_total"] == 6000 and row["avg_order_value"] == 2000
    assert row["recency_days"] == 61            # 1 May to 1 July
    assert row["tenure_days"] == 181            # 1 Jan to 1 July
    assert row["avg_gap_days"] == 60            # 120 days between first and last order, 2 gaps
    assert row["silence_in_gaps"] == pytest.approx(61 / 60)
    assert row["spend_last_12m"] == 6000 and row["spend_last_3m"] == 3000 and row["spend_prev_12m"] == 0
    assert row["orders_per_year"] == pytest.approx(3 / 181 * 365)
    single = clv.build_features(history, pd.Timestamp("2025-07-01")).loc["B"]
    assert np.isnan(single["avg_gap_days"]) and single["spend_last_12m"] == 0 and single["value_trend"] == 1


def test_no_leakage_the_future_cannot_change_features_or_training_data(sample_orders):
    settings = config()
    cutoff = pd.Timestamp("2025-10-01")
    before = clv.build_features(sample_orders, cutoff)

    tampered = sample_orders.copy()
    future = tampered["order_date"] >= cutoff
    assert future.sum() > 1000
    tampered.loc[future, "order_value"] *= 100                           # rewrite every holdout order
    extra = orders([("C00001", "2025-10-01", 9e6), ("NEW", "2026-01-01", 9e6)])
    tampered = pd.concat([tampered, extra], ignore_index=True)
    pd.testing.assert_frame_equal(before, clv.build_features(tampered, cutoff))

    # the rows the holdout model learns from are also untouched by anything after the cut-off
    cutoffs = clv.training_cutoffs(sample_orders["order_date"].min(), cutoff - pd.DateOffset(months=12), settings)
    x_true, y_true = clv.training_set(sample_orders, cutoffs, cutoff, settings)
    x_tampered, y_tampered = clv.training_set(tampered, cutoffs, cutoff, settings)
    pd.testing.assert_frame_equal(x_true, x_tampered)
    pd.testing.assert_series_equal(y_true, y_tampered)

    # and a cut-off whose outcome window reaches past "today" is refused
    with pytest.raises(ValueError, match="leakage"):
        clv.training_set(sample_orders, [cutoff - pd.DateOffset(months=11)], cutoff, settings)


def test_target_window_includes_the_start_date_and_excludes_the_end_date():
    history = orders([("A", "2025-09-30", 100), ("A", "2025-10-01", 200), ("A", "2026-09-30", 400),
                      ("A", "2026-10-01", 800), ("B", "2024-01-01", 50)])
    spend = clv.actual_spend(history, pd.Index(["A", "B"]), pd.Timestamp("2025-10-01"), pd.Timestamp("2026-10-01"))
    assert spend.tolist() == [600.0, 0.0]


def test_simple_rule_matches_a_hand_calculation():
    features = pd.DataFrame({"avg_order_value": [1000.0, 1000.0, 500.0], "orders_per_year": [4.0, 4.0, 2.0],
                             "avg_gap_days": [30.0, 30.0, np.nan], "recency_days": [0, 60, 90]})
    value, active = clv.predict_rule(features, typical_gap=90, horizon_months=12, half_life_gaps=2.0)
    assert active.tolist() == pytest.approx([1.0, 0.5, 0.5 ** 0.5])      # silent for 0, 2 and 0.5 half-lives
    assert value.tolist() == pytest.approx([4000.0, 2000.0, 1000 * 0.5 ** 0.5])
    naive = clv.predict_naive(pd.DataFrame({"spend_last_12m": [1200.0]}), horizon_months=6)
    assert naive.tolist() == [600.0]


def test_ranking_measures_on_ten_customers():
    actual = pd.Series([500.0, 0, 300, 0, 100, 0, 0, 100, 0, 0])
    good = pd.Series([10.0, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    assert clv.share_captured(actual.to_numpy(), good.to_numpy(), 0.10) == pytest.approx(0.5)
    assert clv.share_captured(actual.to_numpy(), good.to_numpy(), 0.20) == pytest.approx(0.5)
    assert clv.share_captured(actual.to_numpy(), actual.to_numpy(), 0.20) == pytest.approx(0.8)   # the best possible
    table = clv.decile_table(actual, good, groups=5)
    assert table["Customers"].tolist() == [2] * 5
    assert table["Avg actual (Rs)"].tolist() == [250, 150, 50, 50, 0]
    assert table["Cumulative share %"].tolist() == pytest.approx([50, 80, 90, 100, 100])
    scores = clv.score_prediction("x", actual, actual * 2)
    assert scores["MAE (Rs)"] == 100 and scores["Spearman"] == pytest.approx(1.0)
    assert scores["Total predicted vs actual %"] == pytest.approx(100.0)


def test_tiers_follow_the_configured_shares():
    value = pd.Series(np.arange(100, 0, -1, dtype=float))                # 100 customers, best first
    tiers = clv.assign_tiers(value, {"Platinum": 5, "Gold": 15, "Silver": 30, "Bronze": 50})
    assert tiers.value_counts().to_dict() == {"Bronze": 50, "Silver": 30, "Gold": 15, "Platinum": 5}
    assert tiers.iloc[4] == "Platinum" and tiers.iloc[5] == "Gold" and tiers.iloc[-1] == "Bronze"


def test_end_to_end_on_the_sample_data(tmp_path):
    settings = config(gross_margin=0.4, payback_months=3)
    result = clv.run(settings, SAMPLE, tmp_path / "output", tmp_path / "images")

    comparison = result["comparison"].set_index("Approach")
    assert list(comparison.index[:3]) == [clv.NAIVE, clv.RULE, clv.ML]
    for approach in (clv.NAIVE, clv.RULE, clv.ML):                       # all far better than a random 20%
        assert comparison.loc[approach, "Revenue in predicted top 20% (%)"] > 50
        assert 0.3 < comparison.loc[approach, "Spearman"] < 0.95         # useful, and not suspiciously perfect
    assert all(cutoff + pd.DateOffset(months=12) <= result["cutoff"] for cutoff in result["train_cutoffs"])

    scores = result["scores"]
    assert len(scores) == result["orders"]["customer_id"].nunique() and scores["customer_id"].is_unique
    assert scores["prob_active"].between(0, 1).all() and (scores["predicted_value"] >= 0).all()
    assert scores["predicted_value"].is_monotonic_decreasing
    expected = scores["predicted_value"] * 0.4 * 3 / 12                  # the spend rule, within rounding
    assert (scores["max_spend_rule_of_thumb"] - expected).abs().max() <= 1
    assert len(result["rejected"]) == 110                                # 40 duplicates + 25 + 15 + 30 planted

    sheets = pd.ExcelFile(result["workbook"]).sheet_names
    assert sheets[:4] == ["Customer Scores", "Decile Table", "Model Comparison", "Tier Summary"]
    assert (tmp_path / "images" / "deciles.png").stat().st_size > 10_000
    assert (tmp_path / "images" / "gains.png").stat().st_size > 10_000


def test_other_column_names_and_too_little_history(tmp_path, capsys):
    raw = pd.read_csv(SAMPLE, dtype=str, keep_default_na=False)
    renamed = raw.rename(columns={"customer_id": "CustID", "order_date": "Order Date", "order_value": "Amount"})
    renamed.to_csv(tmp_path / "mine.csv", index=False)
    arguments = ["--input", str(tmp_path / "mine.csv"), "--customer-col", "CustID", "--date-col", "Order Date",
                 "--value-col", "Amount", "--output", str(tmp_path / "out"), "--images", str(tmp_path / "img")]
    assert clv.main(arguments) == 0
    assert "Model comparison on the holdout" in capsys.readouterr().out

    assert clv.main(["--input", str(tmp_path / "mine.csv"), "--output", str(tmp_path / "out")]) == 1
    assert "Column(s) not found" in capsys.readouterr().err

    short = raw[raw["order_date"] >= "2025-06-01"]                       # 16 months: no room for a holdout test
    short.to_csv(tmp_path / "short.csv", index=False)
    assert clv.main(["--input", str(tmp_path / "short.csv"), "--output", str(tmp_path / "out")]) == 1
    assert "Not enough history" in capsys.readouterr().err
