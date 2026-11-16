"""Customer Lifetime Value: predict what each customer will spend in the next 12 months, and prove it.

Input is a plain transaction file (customer id, order date, order value).

Steps:
    1. Clean     reject rows that cannot be used and report each one with a reason
    2. Validate  hide the latest 12 months (the "holdout"), predict them from the earlier
                 orders only, and compare three approaches against what really happened:
                     Last year's spend   next year = the customer's spend in the last 12 months
                     Simple rule         average order value x orders per year x chance of still being active
                     Gradient boosting   a two-part model: will they buy at all, then how much
    3. Score     refit on all the data and score every customer for the NEXT 12 months:
                 predicted value, chance of being active, value tier and a suggested spend ceiling
    4. Report    an Excel workbook and two charts

Usage:
    python clv_model.py
    python clv_model.py --input my_orders.csv --customer-col CustID --date-col "Order Date" --value-col Amount
    python clv_model.py --margin 0.4 --payback-months 9

Author : Swaminathan Narayanan
Repo   : https://github.com/Swami529796/customer-lifetime-value
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402

HERE = Path(__file__).parent
NAVY, BLUE, AMBER, GREY = "#1F4E78", "#9DB7D5", "#E08A00", "#595959"
NAIVE, RULE, ML = "Last year's spend", "Simple rule", "Gradient boosting"

DEFAULTS = {
    "input": "data/transactions.csv",
    "columns": {"customer_id": "customer_id", "order_date": "order_date", "order_value": "order_value",
                "order_id": "order_id"},
    "date_format": None,
    "day_first": False,
    "horizon_months": 12,
    "snapshot_step_months": 3,
    "min_history_months": 6,
    "heuristic": {"half_life_gaps": 2.0, "min_tenure_days": 180},
    "tiers": {"Platinum": 5, "Gold": 15, "Silver": 30, "Bronze": 50},
    "gross_margin": 0.35,
    "payback_months": 6,
    "random_seed": 7,
}

FEATURES = ["frequency", "monetary_total", "avg_order_value", "last_order_value", "recency_days", "tenure_days",
            "avg_gap_days", "silence_in_gaps", "orders_per_year", "orders_last_12m", "spend_last_12m",
            "spend_prev_12m", "spend_last_3m", "value_trend", "value_variability"]


# ------------------------------------------------------------------ settings
def load_config(path: Path | None) -> dict:
    """Defaults, overridden by the YAML file if there is one."""
    config = {key: (dict(value) if isinstance(value, dict) else value) for key, value in DEFAULTS.items()}
    if path is not None and path.exists():
        for key, value in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).items():
            if key not in DEFAULTS:
                raise ValueError(f"Unknown setting '{key}' in {path.name}")
            if isinstance(DEFAULTS[key], dict) and key != "tiers":
                config[key].update(value or {})
            else:
                config[key] = value
    if abs(sum(config["tiers"].values()) - 100) > 1e-6:
        raise ValueError("The tier shares in the settings must add up to 100.")
    return config


# ------------------------------------------------------------------ clean
def read_transactions(path: Path) -> pd.DataFrame:
    """Read the file as text, so nothing is changed on the way in."""
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean_transactions(raw: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (clean orders, rejected rows with a reason, data quality summary).

    Clean orders have three columns: customer_id, order_date, order_value.
    Nothing is dropped silently: every row that is not used is in `rejected`.
    """
    names = config["columns"]
    needed = [names[key] for key in ("customer_id", "order_date", "order_value")]
    missing = [name for name in needed if name not in raw.columns]
    if missing:
        raise ValueError(f"Column(s) not found in the input: {missing}. The file has: {list(raw.columns)}. "
                         "Set the names in config.yaml or with --customer-col / --date-col / --value-col.")

    data = raw.copy()
    data["_customer"] = data[names["customer_id"]].fillna("").astype(str).str.strip()
    with warnings.catch_warnings():              # pandas warns when it has to guess the format; bad dates are reported below
        warnings.simplefilter("ignore", UserWarning)
        data["_date"] = pd.to_datetime(data[names["order_date"]].astype(str).str.strip(), errors="coerce",
                                       format=config["date_format"],
                                       dayfirst=bool(config["day_first"]) if not config["date_format"] else False)
    value_text = data[names["order_value"]].astype(str).str.replace(r"(?i)rs\.?|inr|,|\s", "", regex=True)
    data["_value"] = pd.to_numeric(value_text, errors="coerce")

    order_id = names.get("order_id")
    key = [order_id] if order_id and order_id in data.columns else list(raw.columns)
    reason = pd.Series("", index=data.index)
    checks = [                                   # the first matching reason wins
        ("Duplicate row", data.duplicated(subset=key, keep="first")),
        ("Missing customer id", data["_customer"].isin(["", "nan", "None"])),
        ("Unreadable order date", data["_date"].isna()),
        ("Order value is not a number", data["_value"].isna()),
        ("Order value zero or negative (refund or cancellation)", data["_value"] <= 0),
    ]
    for label, mask in checks:
        reason = reason.mask((reason == "") & mask, label)

    rejected = raw[reason != ""].assign(reason=reason[reason != ""])
    good = data[reason == ""]
    orders = pd.DataFrame({"customer_id": good["_customer"], "order_date": good["_date"].dt.normalize(),
                           "order_value": good["_value"].astype(float)})
    orders = orders.sort_values(["customer_id", "order_date"], kind="stable").reset_index(drop=True)

    quality = [("Rows in the input file", len(raw), "")]
    quality += [(label, int((reason == label).sum()), "Removed" if label == "Duplicate row" else "Rejected")
                for label, _ in checks]
    quality.append(("Clean orders used", len(orders), ""))
    return orders, rejected, pd.DataFrame(quality, columns=["Check", "Rows", "Action"])


# ------------------------------------------------------------------ features and target
def build_features(orders: pd.DataFrame, cutoff: pd.Timestamp, min_tenure_days: int = 180) -> pd.DataFrame:
    """One row per customer, describing their history BEFORE the cut-off date.

    Orders dated on or after `cutoff` are discarded in the first line, so the
    caller cannot leak the future into the features by passing too much data.
    """
    past = orders[orders["order_date"] < cutoff].sort_values(["customer_id", "order_date"], kind="stable")
    groups = past.groupby("customer_id")
    days_ago = (cutoff - past["order_date"]).dt.days

    table = groups["order_value"].agg(frequency="count", monetary_total="sum", avg_order_value="mean",
                                      last_order_value="last", value_std="std")
    table["recency_days"] = days_ago.groupby(past["customer_id"]).min()       # days since the last order
    table["tenure_days"] = days_ago.groupby(past["customer_id"]).max()        # days since the first order
    repeat = table["frequency"] > 1
    table["avg_gap_days"] = ((table["tenure_days"] - table["recency_days"]) / (table["frequency"] - 1)).where(repeat)
    # silence measured in the customer's own usual gaps: 3.0 = "three times longer than usual"
    table["silence_in_gaps"] = table["recency_days"] / table["avg_gap_days"].clip(lower=1)
    table["orders_per_year"] = table["frequency"] / table["tenure_days"].clip(lower=min_tenure_days) * 365

    def window(first_day: int, last_day: int, how: str) -> pd.Series:
        inside = past[(days_ago > first_day) & (days_ago <= last_day)]
        return inside.groupby("customer_id")["order_value"].agg(how).reindex(table.index, fill_value=0)

    table["orders_last_12m"] = window(0, 365, "count")
    table["spend_last_12m"] = window(0, 365, "sum")
    table["spend_prev_12m"] = window(365, 730, "sum")
    table["spend_last_3m"] = window(0, 91, "sum")
    # trend: are the latest orders bigger or smaller than this customer's average? (1.0 = no change)
    latest = groups.tail(3).groupby("customer_id")["order_value"].mean()
    table["value_trend"] = latest / table["avg_order_value"]
    table["value_variability"] = (table["value_std"] / table["avg_order_value"]).fillna(0)
    table["last_order_date"] = groups["order_date"].max()
    return table[FEATURES + ["last_order_date"]]


def actual_spend(orders: pd.DataFrame, customers: pd.Index, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """What each customer really spent from `start` up to (not including) `end`."""
    inside = orders[(orders["order_date"] >= start) & (orders["order_date"] < end)]
    return inside.groupby("customer_id")["order_value"].sum().reindex(customers, fill_value=0.0)


def training_cutoffs(first_order: pd.Timestamp, latest: pd.Timestamp, config: dict) -> list[pd.Timestamp]:
    """Past cut-off dates to learn from: `latest`, then one every few months going back."""
    earliest = first_order + pd.DateOffset(months=config["min_history_months"])
    cutoffs, cutoff = [], latest
    while cutoff >= earliest:
        cutoffs.append(cutoff)
        cutoff = cutoff - pd.DateOffset(months=config["snapshot_step_months"])
    return cutoffs


def training_set(orders: pd.DataFrame, cutoffs: list[pd.Timestamp], known_until: pd.Timestamp,
                 config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Stack (features at a cut-off, spend in the following window) for several past cut-offs.

    `known_until` is "today" from the model's point of view. Orders from that
    date onward are removed before anything is computed, and a cut-off whose
    target window would run past it is refused.
    """
    visible = orders[orders["order_date"] < known_until]
    horizon = pd.DateOffset(months=config["horizon_months"])
    features, targets = [], []
    for cutoff in cutoffs:
        if cutoff + horizon > known_until:
            raise ValueError(f"Cut-off {cutoff.date()} needs outcomes after {known_until.date()}: that would be leakage.")
        table = build_features(visible, cutoff, config["heuristic"]["min_tenure_days"])
        features.append(table[FEATURES])
        targets.append(actual_spend(visible, table.index, cutoff, cutoff + horizon))
    return pd.concat(features), pd.concat(targets)


# ------------------------------------------------------------------ the three approaches
def predict_naive(features: pd.DataFrame, horizon_months: int = 12) -> pd.Series:
    """Baseline: the next period will repeat the last 12 months."""
    return features["spend_last_12m"] * horizon_months / 12


def predict_rule(features: pd.DataFrame, typical_gap: float, horizon_months: int = 12,
                 half_life_gaps: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Transparent rule: average order value x expected orders x chance of still being active.

    The chance of being active starts at 100% on the day of an order and halves
    every `half_life_gaps` usual gaps of silence. A customer who orders every
    30 days and has been silent for 60 is at 50% (with the default of 2). For a
    one-order customer the usual gap is not known, so `typical_gap` is used.
    Returns (predicted value, chance of being active).
    """
    gap = features["avg_gap_days"].fillna(typical_gap).clip(lower=1)
    active = 0.5 ** (features["recency_days"] / (half_life_gaps * gap))
    expected_orders = features["orders_per_year"] * horizon_months / 12
    return features["avg_order_value"] * expected_orders * active, active


class TwoPartModel:
    """Gradient boosting in two parts, because most of the difficulty is the zeros.

    Part 1 (classifier): the chance that the customer orders at all in the window.
    Part 2 (regressor):  how much they spend IF they order, learnt on a log scale
                         because spend is skewed.
    Prediction = chance of ordering x spend if they order.
    """

    def __init__(self, seed: int = 7):
        settings = {"max_iter": 150, "learning_rate": 0.05, "max_depth": 4, "min_samples_leaf": 40,
                    "l2_regularization": 1.0, "random_state": seed}
        self.buys = HistGradientBoostingClassifier(**settings)
        self.amount = HistGradientBoostingRegressor(**settings)
        self.scale = 1.0

    def fit(self, features: pd.DataFrame, spend: pd.Series) -> "TwoPartModel":
        bought = (spend > 0).to_numpy()
        self.buys.fit(features[FEATURES], bought)
        self.amount.fit(features[FEATURES][bought], np.log(spend[bought]))
        # averaging on a log scale under-states the average in rupees; scale back so the totals agree
        self.scale = float(spend[bought].mean() / np.exp(self.amount.predict(features[FEATURES][bought])).mean())
        return self

    def predict(self, features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Returns (predicted value, chance of ordering at least once)."""
        chance = pd.Series(self.buys.predict_proba(features[FEATURES])[:, 1], index=features.index)
        amount = np.exp(self.amount.predict(features[FEATURES])) * self.scale
        return chance * amount, chance


# ------------------------------------------------------------------ measuring a prediction
def share_captured(actual: np.ndarray, predicted: np.ndarray, top_share: float) -> float:
    """Share of all actual revenue that came from the customers ranked in the top X% by prediction."""
    order = np.argsort(-np.asarray(predicted, dtype=float), kind="stable")
    count = int(round(len(order) * top_share))
    total = float(np.sum(actual))
    return float(np.asarray(actual)[order][:count].sum() / total) if total else 0.0


def score_prediction(name: str, actual: pd.Series, predicted: pd.Series) -> dict:
    truth, guess = actual.to_numpy(dtype=float), predicted.to_numpy(dtype=float)
    return {"Approach": name,
            "MAE (Rs)": float(np.abs(truth - guess).mean()),
            "Total predicted vs actual %": float(100 * (guess.sum() / truth.sum() - 1)),
            "Spearman": float(spearmanr(truth, guess)[0]),
            "Revenue in predicted top 10% (%)": 100 * share_captured(truth, guess, 0.10),
            "Revenue in predicted top 20% (%)": 100 * share_captured(truth, guess, 0.20)}


def capture_gap(actual: pd.Series, first: pd.Series, second: pd.Series, top_share: float = 0.20,
                draws: int = 500, seed: int = 7) -> tuple[float, float, float]:
    """Is `first` really better than `second` at filling the top X%, or is the gap luck?

    Re-draws the customer list `draws` times (a bootstrap) and returns the gap in
    percentage points with a 90% range. A range that includes zero means the two
    rankings cannot be told apart on this data.
    """
    truth, one, two = (series.to_numpy(dtype=float) for series in (actual, first, second))
    rng = np.random.default_rng(seed)
    gaps = []
    for _ in range(draws):
        pick = rng.integers(0, len(truth), len(truth))
        gaps.append(share_captured(truth[pick], one[pick], top_share) - share_captured(truth[pick], two[pick], top_share))
    gap = share_captured(truth, one, top_share) - share_captured(truth, two, top_share)
    low, high = np.percentile(gaps, [5, 95])
    return 100 * gap, 100 * float(low), 100 * float(high)


def decile_table(actual: pd.Series, predicted: pd.Series, groups: int = 10) -> pd.DataFrame:
    """Rank customers by prediction, cut into ten equal groups, compare predicted with actual."""
    order = np.argsort(-predicted.to_numpy(dtype=float), kind="stable")
    group = np.empty(len(order), dtype=int)
    group[order] = np.arange(len(order)) * groups // len(order) + 1
    frame = pd.DataFrame({"Decile": group, "predicted": predicted.to_numpy(), "actual": actual.to_numpy()})
    table = frame.groupby("Decile").agg(**{"Customers": ("actual", "size"),
                                           "Avg predicted (Rs)": ("predicted", "mean"),
                                           "Avg actual (Rs)": ("actual", "mean"),
                                           "Bought again %": ("actual", lambda spend: 100 * (spend > 0).mean()),
                                           "Actual revenue (Rs)": ("actual", "sum")}).reset_index()
    table["Share of actual revenue %"] = 100 * table["Actual revenue (Rs)"] / table["Actual revenue (Rs)"].sum()
    table["Cumulative share %"] = table["Share of actual revenue %"].cumsum()
    return table


def gains_curve(actual: pd.Series, predicted: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """x = % of customers taken from the top of the ranking, y = % of actual revenue they brought."""
    order = np.argsort(-predicted.to_numpy(dtype=float), kind="stable")
    captured = np.concatenate([[0.0], np.cumsum(actual.to_numpy(dtype=float)[order])])
    return np.linspace(0, 100, len(captured)), 100 * captured / captured[-1]


# ------------------------------------------------------------------ validate on the holdout
def validate(orders: pd.DataFrame, config: dict) -> dict:
    """Hide the last `horizon_months`, predict them three ways, and compare with what happened."""
    horizon = pd.DateOffset(months=config["horizon_months"])
    first_order = orders["order_date"].min()
    score_date = orders["order_date"].max() + pd.Timedelta(1, "D")      # the first day we know nothing about
    cutoff = score_date - horizon                                       # calibration | holdout
    cutoffs = training_cutoffs(first_order, cutoff - horizon, config)
    if not cutoffs:
        months = 2 * config["horizon_months"] + config["min_history_months"]
        raise ValueError(f"Not enough history: the file covers {first_order.date()} to {(score_date - pd.Timedelta(1, "D")).date()}, "
                         f"but a holdout test needs at least {months} months.")

    rule = config["heuristic"]
    features = build_features(orders, cutoff, rule["min_tenure_days"])
    actual = actual_spend(orders, features.index, cutoff, score_date)
    typical_gap = float(features["avg_gap_days"].median())

    train_x, train_y = training_set(orders, cutoffs, cutoff, config)
    model = TwoPartModel(config["random_seed"]).fit(train_x, train_y)

    predictions = {NAIVE: predict_naive(features, config["horizon_months"]),
                   RULE: predict_rule(features, typical_gap, config["horizon_months"], rule["half_life_gaps"])[0],
                   ML: model.predict(features)[0]}
    comparison = pd.DataFrame([score_prediction(name, actual, guess) for name, guess in predictions.items()])
    ceiling = score_prediction("Perfect foresight (upper limit)", actual, actual)
    comparison = pd.concat([comparison, pd.DataFrame([ceiling])], ignore_index=True)
    best = comparison.iloc[:3].sort_values(["Revenue in predicted top 20% (%)", "Spearman"],
                                           ascending=False).iloc[0]["Approach"]
    gaps = pd.DataFrame([(f"{ML} minus {other}", *capture_gap(actual, predictions[ML], predictions[other],
                                                              seed=config["random_seed"]))
                         for other in (RULE, NAIVE)],
                        columns=["Revenue in predicted top 20%", "Gap (points)", "90% range low", "90% range high"])
    deciles = pd.concat([decile_table(actual, guess).assign(Approach=name) for name, guess in predictions.items()],
                        ignore_index=True)
    deciles = deciles[["Approach"] + [column for column in deciles.columns if column != "Approach"]]
    return {"cutoff": cutoff, "score_date": score_date, "first_order": first_order, "train_cutoffs": cutoffs,
            "train_rows": len(train_x), "features": features, "actual": actual, "predictions": predictions,
            "comparison": comparison, "gaps": gaps, "deciles": deciles, "best": best}


# ------------------------------------------------------------------ score every customer
def assign_tiers(value: pd.Series, tiers: dict[str, float]) -> pd.Series:
    """Cut the ranking by predicted value into tiers, e.g. the top 5% are Platinum."""
    rank = value.rank(method="first", ascending=False)
    limits = np.cumsum(list(tiers.values())) / 100 * len(value)
    position = np.searchsorted(limits, rank.to_numpy(), side="left").clip(max=len(tiers) - 1)
    return pd.Series(np.array(list(tiers))[position], index=value.index)


def score_customers(orders: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Refit on everything that is known and predict the next `horizon_months` for every customer."""
    horizon = pd.DateOffset(months=config["horizon_months"])
    score_date = orders["order_date"].max() + pd.Timedelta(1, "D")
    rule = config["heuristic"]
    cutoffs = training_cutoffs(orders["order_date"].min(), score_date - horizon, config)
    train_x, train_y = training_set(orders, cutoffs, score_date, config)
    model = TwoPartModel(config["random_seed"]).fit(train_x, train_y)

    features = build_features(orders, score_date, rule["min_tenure_days"])
    value, chance = model.predict(features)
    rule_value, _ = predict_rule(features, float(features["avg_gap_days"].median()), config["horizon_months"],
                                 rule["half_life_gaps"])
    scores = pd.DataFrame({
        "customer_id": features.index,
        "predicted_value": value.round(0).to_numpy(),
        "prob_active": chance.round(3).to_numpy(),
        "value_tier": assign_tiers(value, config["tiers"]).to_numpy(),
        "max_spend_rule_of_thumb": (value * config["gross_margin"] * config["payback_months"]
                                    / config["horizon_months"]).round(0).to_numpy(),
        "orders_to_date": features["frequency"].to_numpy(),
        "spend_to_date": features["monetary_total"].round(0).to_numpy(),
        "spend_last_12m": features["spend_last_12m"].round(0).to_numpy(),
        "last_order_date": features["last_order_date"].dt.date.to_numpy(),
        "days_since_last_order": features["recency_days"].to_numpy(),
        "simple_rule_value": rule_value.round(0).to_numpy(),
    })
    scores = scores.sort_values("predicted_value", ascending=False, kind="stable").reset_index(drop=True)
    scores.insert(0, "rank", scores.index + 1)
    return scores


def tier_summary(scores: pd.DataFrame, tiers: dict[str, float]) -> pd.DataFrame:
    table = scores.groupby("value_tier").agg(**{
        "Customers": ("customer_id", "size"),
        "Avg predicted value (Rs)": ("predicted_value", "mean"),
        "Total predicted value (Rs)": ("predicted_value", "sum"),
        "Avg chance of being active %": ("prob_active", lambda chance: 100 * chance.mean()),
        "Avg max spend per customer (Rs)": ("max_spend_rule_of_thumb", "mean"),
        "Total spend ceiling (Rs)": ("max_spend_rule_of_thumb", "sum")}).reindex(list(tiers)).reset_index()
    table = table.rename(columns={"value_tier": "Tier"})
    table.insert(2, "Share of customers %", 100 * table["Customers"] / table["Customers"].sum())
    table.insert(5, "Share of predicted value %",
                 100 * table["Total predicted value (Rs)"] / table["Total predicted value (Rs)"].sum())
    return table


# ------------------------------------------------------------------ charts
def tidy(axis) -> None:
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_axisbelow(True)


def chart_deciles(deciles: pd.DataFrame, approach: str, target: Path) -> None:
    table = deciles[deciles["Approach"] == approach]
    average = (table["Actual revenue (Rs)"].sum() / table["Customers"].sum())
    multiple = table["Avg actual (Rs)"].iloc[0] / average
    position, width = np.arange(len(table)), 0.4
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.bar(position - width / 2, table["Avg predicted (Rs)"] / 1e3, width, color=BLUE, label="Predicted")
    axis.bar(position + width / 2, table["Avg actual (Rs)"] / 1e3, width, color=NAVY, label="Actual (holdout year)")
    axis.set_xticks(position, [str(decile) for decile in table["Decile"]])
    axis.set_xlabel("Customers ranked by predicted value (1 = top 10%, 10 = bottom 10%)")
    axis.set_ylabel("Average spend per customer (Rs thousand)")
    axis.set_title(f"The predicted top 10% went on to spend {multiple:.1f}x the average customer",
                   loc="left", weight="bold", pad=24)
    axis.text(0, 1.02, f"{approach}: predicted vs actual spend in the holdout months it never saw",
              transform=axis.transAxes, fontsize=9, color=GREY, va="bottom")
    axis.legend(frameon=False, loc="upper right")
    axis.grid(axis="y", color="#E3E7ED")
    tidy(axis)
    figure.tight_layout()
    figure.savefig(target, dpi=130)
    plt.close(figure)


def chart_gains(actual: pd.Series, predictions: dict[str, pd.Series], best: str, target: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    x, y = gains_curve(actual, actual)
    axis.plot(x, y, color=GREY, linestyle="--", linewidth=1.2, label="Perfect foresight (upper limit)")
    styles = {ML: (NAVY, "-", 2.4), RULE: (AMBER, "-", 1.8), NAIVE: (BLUE, "-", 1.8)}
    for name, guess in predictions.items():
        colour, line, width = styles[name]
        x, y = gains_curve(actual, guess)
        axis.plot(x, y, color=colour, linestyle=line, linewidth=width, label=name)
    axis.plot([0, 100], [0, 100], color=GREY, linestyle=":", linewidth=1.2, label="No ranking (random)")
    at_20 = {name: 100 * share_captured(actual.to_numpy(), guess.to_numpy(), 0.20)
             for name, guess in predictions.items()}
    reached, spread = at_20[best], max(at_20.values()) - min(at_20.values())
    axis.plot([20], [reached], marker="o", color=styles[best][0], markersize=6)
    axis.annotate(f"{best}: top 20% of customers\nbrought {reached:.0f}% of next year's revenue",
                  xy=(20, reached), xytext=(30, reached - 16), fontsize=9, va="top",
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": 2},
                  arrowprops={"arrowstyle": "->", "color": GREY, "relpos": (0.0, 1.0)})
    axis.set_xlabel("Customers, highest predicted value first (%)")
    axis.set_ylabel("Share of actual holdout revenue (%)")
    axis.set_title(f"A fifth of the customers, chosen in advance, brought {reached:.0f}% of the revenue",
                   loc="left", weight="bold", pad=24)
    axis.text(0, 1.02, f"The three rankings are within {spread:.1f} points of each other at the 20% mark; "
              "random selection would bring 20%", transform=axis.transAxes, fontsize=9, color=GREY, va="bottom")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 102)
    axis.legend(frameon=False, loc="lower right")
    axis.grid(color="#E3E7ED")
    tidy(axis)
    figure.tight_layout()
    figure.savefig(target, dpi=130)
    plt.close(figure)


# ------------------------------------------------------------------ workbook
def write_workbook(target: Path, sheets: dict[str, pd.DataFrame], notes: dict[str, str]) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            top = 2 if name in notes else 0
            frame.to_excel(writer, sheet_name=name, index=False, startrow=top)
            sheet = writer.sheets[name]
            if name in notes:
                sheet["A1"] = notes[name]
                sheet["A1"].font = Font(italic=True, color="595959")
            for cell in sheet[top + 1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(wrap_text=True, vertical="center")
            sheet.freeze_panes = sheet.cell(row=top + 2, column=1)
            for index, column in enumerate(frame.columns, start=1):
                letter = sheet.cell(row=top + 1, column=index).column_letter
                longest = max([len(str(column))] + [len(str(value)) for value in frame[column].head(200)])
                sheet.column_dimensions[letter].width = min(max(12, longest + 2), 48)
                if pd.api.types.is_float_dtype(frame[column]):
                    whole = "(Rs)" in str(column) or column in ("predicted_value", "max_spend_rule_of_thumb",
                                                                "spend_to_date", "spend_last_12m", "simple_rule_value")
                    for row in sheet.iter_rows(min_row=top + 2, min_col=index, max_col=index):
                        row[0].number_format = "#,##0" if whole else "0.000" if column in ("prob_active", "Spearman") else "0.0"


# ------------------------------------------------------------------ run
def run(config: dict, input_path: Path, output: Path, images: Path) -> dict:
    raw = read_transactions(input_path)
    orders, rejected, quality = clean_transactions(raw, config)
    if orders.empty:
        raise ValueError("No usable orders were found in the input file.")

    result = validate(orders, config)
    scores = score_customers(orders, config)
    tiers = tier_summary(scores, config["tiers"])

    output.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    chart_deciles(result["deciles"], ML, images / "deciles.png")
    chart_gains(result["actual"], result["predictions"], result["best"], images / "gains.png")

    months = config["horizon_months"]
    spend_note = (f"max_spend_rule_of_thumb = predicted_value x gross margin {config['gross_margin']:.0%} x payback "
                  f"{config['payback_months']} months / {months}. A rule of thumb to set a ceiling, not a forecast of what a campaign will earn.")
    notes = {
        "Customer Scores": f"Prediction for the {months} months from {result['score_date'].date()}, model: {ML}. " + spend_note,
        "Decile Table": f"Holdout test: customers ranked by the prediction made on {result['cutoff'].date()}, "
                        f"compared with what they really spent in the following {months} months.",
        "Model Comparison": f"Holdout test on {len(result['actual']):,} customers. Most revenue in the predicted top 20%: {result['best']}. "
                            "MAE = average error per customer; Spearman = rank agreement (1 = identical order).",
        "Tier Summary": spend_note,
    }
    sheets = {"Customer Scores": scores, "Decile Table": result["deciles"], "Model Comparison": result["comparison"],
              "Tier Summary": tiers, "Ranking Gaps": result["gaps"], "Data Quality": quality, "Rejected Rows": rejected}
    workbook = output / "clv_scores.xlsx"
    write_workbook(workbook, sheets, notes)
    return {**result, "orders": orders, "rejected": rejected, "quality": quality, "scores": scores,
            "tiers": tiers, "workbook": workbook}


def print_report(result: dict, config: dict) -> None:
    pd.set_option("display.width", 250)
    money = "{:,.0f}".format
    one = "{:.1f}".format
    months = config["horizon_months"]
    actual = result["actual"]

    print("Data quality")
    print(result["quality"].to_string(index=False))
    print(f"\nOrders from {result['first_order'].date()} to {(result['score_date'] - pd.Timedelta(1, "D")).date()}, "
          f"{result['orders']['customer_id'].nunique():,} customers")

    print(f"\nHoldout test: features use orders before {result['cutoff'].date()}, "
          f"the target is spend in the {months} months from that date")
    print(f"  customers tested: {len(actual):,}   bought again in the holdout: {(actual > 0).mean():.1%}   "
          f"holdout revenue: Rs {actual.sum():,.0f}")
    print(f"  the model learnt from {result['train_rows']:,} rows at past cut-offs "
          f"{', '.join(str(cutoff.date()) for cutoff in sorted(result['train_cutoffs']))} "
          f"(their outcomes all end by {result['cutoff'].date()})")

    print("\nModel comparison on the holdout")
    print(result["comparison"].to_string(index=False, formatters={
        "MAE (Rs)": money, "Total predicted vs actual %": "{:+.1f}".format, "Spearman": "{:.3f}".format,
        "Revenue in predicted top 10% (%)": one, "Revenue in predicted top 20% (%)": one}))
    print(f"Most revenue in the predicted top 20%: {result['best']}. Is the gap real? (90% range from 500 re-draws of the customers)")
    for _, row in result["gaps"].iterrows():
        verdict = "a real difference" if row["90% range low"] > 0 or row["90% range high"] < 0 else "too close to call"
        print(f"  {row.iloc[0]}: {row['Gap (points)']:+.1f} points (range {row['90% range low']:+.1f} to "
              f"{row['90% range high']:+.1f}): {verdict}")

    deciles = result["deciles"]
    print(f"\nDecile table ({ML})")
    print(deciles[deciles["Approach"] == ML].drop(columns="Approach").to_string(index=False, formatters={
        "Avg predicted (Rs)": money, "Avg actual (Rs)": money, "Actual revenue (Rs)": money, "Bought again %": one,
        "Share of actual revenue %": one, "Cumulative share %": one}))

    print(f"\nScores for the next {months} months from {result['score_date'].date()} "
          f"({len(result['scores']):,} customers; margin {config['gross_margin']:.0%}, payback {config['payback_months']} months)")
    print(result["tiers"].to_string(index=False, formatters={
        column: (one if "%" in column else money) for column in result["tiers"].columns[1:]}))
    print("\nTop 5 customers")
    print(result["scores"].head(5).drop(columns=["simple_rule_value", "spend_last_12m"]).to_string(index=False, formatters={
        "predicted_value": money, "max_spend_rule_of_thumb": money, "spend_to_date": money}))
    print(f"\nSaved: {result['workbook']}")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict each customer's spend over the next 12 months and prove it on a holdout.")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml", help="YAML settings file")
    parser.add_argument("--input", type=Path, help="transaction file (.csv or .xlsx)")
    parser.add_argument("--customer-col", help="name of the customer id column")
    parser.add_argument("--date-col", help="name of the order date column")
    parser.add_argument("--value-col", help="name of the order value column")
    parser.add_argument("--horizon-months", type=int, help="months to predict ahead (and length of the holdout)")
    parser.add_argument("--margin", type=float, help="gross margin as a share, e.g. 0.35")
    parser.add_argument("--payback-months", type=float, help="months within which the spend must be earned back")
    parser.add_argument("--output", type=Path, default=HERE / "output", help="folder for the workbook")
    parser.add_argument("--images", type=Path, default=HERE / "images", help="folder for the charts")
    args = parser.parse_args(arguments)

    try:
        config = load_config(args.config)
        for key, value in (("customer_id", args.customer_col), ("order_date", args.date_col),
                           ("order_value", args.value_col)):
            if value:
                config["columns"][key] = value
        for key, value in (("horizon_months", args.horizon_months), ("gross_margin", args.margin),
                           ("payback_months", args.payback_months)):
            if value is not None:
                config[key] = value
        input_path = args.input or (args.config.parent / config["input"])
        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")
        result = run(config, input_path, args.output, args.images)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print_report(result, config)
    print(f"Charts: {args.images / 'deciles.png'}, {args.images / 'gains.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
