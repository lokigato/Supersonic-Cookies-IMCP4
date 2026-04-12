"""
main.py
=======
IMC Prosperity Trading System 

Pipeline:
    1. Fetch market data from Yahoo Finance (falls back to synthetic).
    2. Run the Trend Following strategy with the MEDIUM risk profile.
    3. Print the performance report.
    4. Run all three risk-profile scenarios (conservative / medium / high).
    5. Monte Carlo simulation on the trend-following equity curve.

Usage
-----
    python main.py

To skip Yahoo Finance and use synthetic data, set:
    FORCE_SYNTHETIC = True
"""

import warnings
from typing import Dict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Local imports -------------------------------------------------------------
from data_feed import fetch_market_data, align_products
from strategies.trend_following import TrendFollowingConfig, TrendFollowingStrategy
from performance import performance_report, multi_strategy_comparison
from sensitivity import (
    RISK_PROFILES,
    run_scenario,
    monte_carlo_equity,
    scenario_comparison_table,
)

# -- Constants -----------------------------------------------------------------
# ============================================================
#  DATA MODE
#  FORCE_SYNTHETIC = False  ->  use Yahoo Finance (real data)
#  FORCE_SYNTHETIC = True   ->  skip download, use synthetic
# ============================================================
FORCE_SYNTHETIC  = False

INITIAL_CAPITAL  = 100_000.0
START_DATE       = "2022-01-01"
END_DATE         = "2026-01-01"
TF_PRODUCTS = ["STARFRUIT", "SQUID_INK", "KELP"]


# -- Data ----------------------------------------------------------------------

def load_data() -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Download (or generate) OHLCV data for all Prosperity products and return
    both the per-product dict and the aligned Close-price matrix.
    """
    data = fetch_market_data(
        start=START_DATE,
        end=END_DATE,
        force_synthetic=FORCE_SYNTHETIC,
    )
    prices_wide = align_products(data)
    print(f"  Aligned price matrix : {prices_wide.shape[0]} bars x "
          f"{prices_wide.shape[1]} products")
    return data, prices_wide


# -- Single backtest (MEDIUM risk profile) -------------------------------------

def run_individual_strategy(
    data: Dict[str, pd.DataFrame],
    prices_wide: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """
    Run the Trend Following strategy with the MEDIUM risk profile and
    print the full performance report.

    Returns a dict with key 'TrendFollowing' mapping to the equity curve.
    """
    profile   = RISK_PROFILES["MEDIUM"]
    tf_products = [p for p in TF_PRODUCTS if p in data]

    print("\n" + "=" * 60)
    print("  TREND FOLLOWING -- MEDIUM risk profile")
    print("=" * 60)

    equity_curves: Dict[str, pd.Series] = {}

    if not tf_products:
        print("  [WARN] None of the TF products are available in the loaded data.")
        return equity_curves

    tf_cfg = TrendFollowingConfig(
        max_position_frac=profile.max_position_frac,
        risk_per_trade=profile.risk_per_trade,
        products=tf_products,
    )
    tf_strat = TrendFollowingStrategy(tf_cfg, INITIAL_CAPITAL)

    n_bars = max(len(data[p]) for p in tf_products)
    for t in range(n_bars):
        snap = {p: data[p].iloc[:t + 1] for p in tf_products if t < len(data[p])}
        tf_strat.generate_orders(snap, t)

    tf_eq = tf_strat.get_equity_curve()
    tf_eq.index = range(len(tf_eq))
    equity_curves["TrendFollowing"] = tf_eq

    performance_report(
        equity=tf_eq,
        trades_df=tf_strat.get_trades(),
        prices=prices_wide,
        strategy="Trend Following [MEDIUM]",
        initial_cap=INITIAL_CAPITAL,
    )

    return equity_curves


# -- Scenario analysis (all three risk profiles) -------------------------------

def run_scenarios(
    data: Dict[str, pd.DataFrame],
    prices_wide: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Run conservative / medium / high risk profiles for Trend Following only
    and print a side-by-side comparison table.
    """
    print("\n" + "=" * 60)
    print("  SCENARIO ANALYSIS -- all risk profiles")
    print("=" * 60)

    scenario_results: Dict[str, Dict] = {}

    for profile_name, profile in RISK_PROFILES.items():
        print(f"\n  Running {profile_name}: {profile.description}")
        result = run_scenario(
            prices_wide=prices_wide,
            history=data,
            profile=profile,
            capital=INITIAL_CAPITAL,
            strategies_to_run=["trend"],   # trend following only
        )
        scenario_results[profile_name] = result

    scenario_comparison_table(scenario_results)
    return scenario_results


# Monte Carlo 

def run_monte_carlo(equity_curves: Dict[str, pd.Series]) -> None:
    """Bootstrap 1 000 equity paths from the trend-following return series."""
    print("\n" + "=" * 60)
    print("  MONTE CARLO SIMULATION (Trend Following)")
    print("=" * 60)

    tf_eq = equity_curves.get("TrendFollowing")
    if tf_eq is None or len(tf_eq) < 10:
        print("  [WARN] Not enough data for Monte Carlo simulation.")
        return

    monte_carlo_equity(tf_eq, n_paths=1000, confidence=0.95)


# Entry point 

def main():
    data, prices_wide = load_data()

    # 1. Backtest with MEDIUM profile
    equity_curves = run_individual_strategy(data, prices_wide)

    # 2. Scenario comparison across all three risk profiles
    scenario_results = run_scenarios(data, prices_wide)

    # 3. Monte Carlo on the MEDIUM equity curve
    run_monte_carlo(equity_curves)

    print("\n  Backtest complete.")
    return equity_curves, scenario_results


if __name__ == "__main__":
    main()
