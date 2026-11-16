"""Create the sample transaction file used by clv_model.py (fixed seed, so it is reproducible).

The file imitates three years of orders (1 Oct 2023 to 30 Sep 2026) at a made-up
online home and kitchen retailer, "Kaveri Home Store". About 5,000 customers are
simulated with the "buy till you die" process that real repeat-purchase data follows:

    * every customer has their OWN purchase rate (a few order monthly, most order
      once or twice a year) and their OWN typical order value in rupees
    * every customer can silently churn: nobody cancels, they simply stop ordering.
      About a quarter never come back after the first order; the rest have an
      individual "lifetime" that may be months or many years
    * orders arrive at random around the customer's rate, with a festive-season
      bump in October and November
    * a customer's order values scatter around their typical value and drift
      slowly up or down over the years

Because orders are random and churn is invisible, future spend can be predicted
only partly. That is the point: a result that looked perfect would be a mistake.

Problems planted on purpose, so the cleaning step has something to report:

    * 40 exact duplicate rows                 (a double export)
    * 25 rows with no customer id             (guest checkouts)
    * 15 rows with an unreadable order date
    * 30 rows with a zero or negative value   (refunds booked as orders)

Usage:
    python make_sample_data.py

Author : Swaminathan Narayanan
Repo   : https://github.com/Swami529796/customer-lifetime-value
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SEED = 42
CUSTOMERS = 5000
START, END = pd.Timestamp("2023-10-01"), pd.Timestamp("2026-09-30")
FESTIVE_MONTHS = (10, 11)          # October and November sell about 50% more
FESTIVE_LIFT = 1.5
MONTH_OF_DAY = pd.date_range(START, END, freq="D").month.to_numpy()


def simulate_customer(rng: np.random.Generator, first_day: int, total_days: int) -> list[tuple[int, float]]:
    """Orders of one customer as (day number, order value in Rs)."""
    orders_per_year = rng.gamma(shape=1.1, scale=3.4)                 # most are low, a few are very high
    typical_value = rng.lognormal(mean=np.log(1700), sigma=0.55) * (1 + 0.04 * min(orders_per_year, 10))
    yearly_drift = rng.normal(0.0, 0.12)                              # spend creeping up or down
    if rng.random() < 0.25:
        lifetime_days = 0.0                                           # one-time buyer
    else:
        lifetime_days = rng.exponential(rng.gamma(shape=2.0, scale=450))
    last_possible = min(total_days - 1, first_day + lifetime_days)

    days = [first_day]
    day = float(first_day)
    while True:
        day += rng.exponential(365.0 / (orders_per_year * FESTIVE_LIFT))
        if day > last_possible:
            break
        if MONTH_OF_DAY[int(day)] in FESTIVE_MONTHS or rng.random() < 1 / FESTIVE_LIFT:   # thin out the non-festive months
            days.append(int(day))
    orders = []
    for order_day in days:
        level = typical_value * (1 + yearly_drift) ** ((order_day - first_day) / 365.0)
        orders.append((order_day, max(99.0, level * rng.gamma(shape=4.0, scale=0.25))))
    return orders


def build_orders(rng: np.random.Generator) -> pd.DataFrame:
    total_days = (END - START).days + 1
    rows = []
    for number in range(1, CUSTOMERS + 1):
        first_day = int(rng.integers(0, total_days))
        for order_day, value in simulate_customer(rng, first_day, total_days):
            rows.append((f"C{number:05d}", order_day, round(value)))
    orders = pd.DataFrame(rows, columns=["customer_id", "day", "order_value"])
    orders = orders.sort_values(["day", "customer_id"], kind="stable").reset_index(drop=True)
    orders["order_date"] = (START + pd.to_timedelta(orders["day"], unit="D")).dt.strftime("%Y-%m-%d")
    orders.insert(0, "order_id", [f"KH{number:06d}" for number in range(1, len(orders) + 1)])
    return orders[["order_id", "customer_id", "order_date", "order_value"]]


def plant_problems(orders: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    orders = orders.astype({"order_value": object})
    picks = rng.choice(len(orders), size=70, replace=False)
    orders.loc[picks[:25], "customer_id"] = ""
    orders.loc[picks[25:40], "order_date"] = rng.choice(["2025-13-45", "not available", "00/00/0000"], size=15)
    orders.loc[picks[40:60], "order_value"] = [-int(value) for value in orders.loc[picks[40:60], "order_value"]]
    orders.loc[picks[60:70], "order_value"] = 0
    duplicates = orders.iloc[rng.choice(len(orders), size=40, replace=False)]
    return pd.concat([orders, duplicates]).sort_index(kind="stable").reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    orders = plant_problems(build_orders(rng), rng)
    target = HERE / "data" / "transactions.csv"
    target.parent.mkdir(exist_ok=True)
    orders.to_csv(target, index=False, lineterminator="\n")
    print(f"Wrote {len(orders):,} rows for {CUSTOMERS:,} customers to {target} "
          f"({target.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
