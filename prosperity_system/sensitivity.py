"""
sensitivity.py
==============
Sensitivity analysis for the Trend Following strategy.

Three risk profiles applied to a $100,000 starting capital:

Profile         max_pos_frac  risk_per_trade  stop_loss_z  target leverage
--------------------------------------------------------------------------
CONSERVATIVE    10%           0.005           2.5          ~0.4x
MEDIUM          20%           0.010           3.0          ~0.8x
HIGH            35%           0.020           4.0          ~1.3x

Functions available to main.py:
    run_scenario(...)             -- backtest one risk profile
    monte_carlo_equity(...)       -- bootstrap path simulation
    scenario_comparison_table()  -- side-by-side profile comparison
    RISK_PROFILES                 -- dict of the three profiles
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from performance import (
    max_drawdown,
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    performance_report,
)
from strategies.trend_following import TrendFollowingConfig, TrendFollowingStrategy


INITIAL_CAPITAL = 100_000.0


# Risk profiles

@dataclass
class RiskProfile:
    name:              str
    max_position_frac: float
    risk_per_trade:    float
    stop_loss_z:       float
    description:       str


RISK_PROFILES: Dict[str, RiskProfile] = {
    "CONSERVATIVE": RiskProfile(
        name="CONSERVATIVE",
        max_position_frac=0.10,
        risk_per_trade=0.005,
        stop_loss_z=2.5,
        description="Low drawdown, small position sizes.",
    ),
    "MEDIUM": RiskProfile(
        name="MEDIUM",
        max_position_frac=0.20,
        risk_per_trade=0.01,
        stop_loss_z=3.0,
        description="Balanced risk/reward, standard ATR sizing.",
    ),
    "HIGH": RiskProfile(
        name="HIGH",
        max_position_frac=0.35,
        risk_per_trade=0.02,
        stop_loss_z=4.0,
        description="Aggressive sizing, larger positions.",
    ),
}


# Config builder

def build_tf_config(profile: RiskProfile) -> TrendFollowingConfig:
    """Build a TrendFollowingConfig from a risk profile."""
    return TrendFollowingConfig(
        max_position_frac=profile.max_position_frac,
        risk_per_trade=profile.risk_per_trade,
        hurst_threshold=0.52,
    )


# Scenario runner

def run_scenario(
    prices_wide: pd.DataFrame,
    history: Dict[str, pd.DataFrame],
    profile: RiskProfile,
    capital: float = INITIAL_CAPITAL,
    strategies_to_run: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the Trend Following backtest for a given risk profile.

    The parameter strategies_to_run is accepted for API compatibility but
    only "trend" is supported. Any other value is silently ignored.

    Returns a dict with keys: "profile", "trend", "combined", "equity_curves".
    """
    tf_cfg = build_tf_config(profile)
    strat  = TrendFollowingStrategy(tf_cfg, capital)
    equity = _run_tf(strat, history, prices_wide)

    report = performance_report(
        equity=equity,
        strategy=f"{profile.name}/trend",
        initial_cap=capital,
        annualise=252,
    )

    return {
        "profile":       profile.name,
        "trend":         report,
        "combined":      report,   # same equity -- used by scenario_comparison_table
        "equity_curves": {"trend": equity},
    }


# Backtest loop 

def _run_tf(
    strat: TrendFollowingStrategy,
    history: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
) -> pd.Series:
    """Iterate bar by bar and call generate_orders on each snapshot."""
    n = max(len(df) for df in history.values()) if history else len(prices)
    for t in range(n):
        snap = {p: df.iloc[:t + 1] for p, df in history.items() if t < len(df)}
        strat.generate_orders(snap, t)
    eq = strat.get_equity_curve()
    eq.index = range(len(eq))
    return eq


# Monte Carlo bootstrap

def monte_carlo_equity(
    equity: pd.Series,
    n_paths: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap resampling of daily returns to generate n_paths simulated
    equity curves. Returns confidence-interval band and terminal statistics.
    """
    rng    = np.random.default_rng(seed)
    ret    = returns_from_equity(equity).values
    n_bars = len(ret)
    init   = float(equity.iloc[0])

    paths = np.zeros((n_paths, n_bars + 1))
    paths[:, 0] = init

    for i in range(n_paths):
        sampled = rng.choice(ret, size=n_bars, replace=True)
        paths[i, 1:] = init * np.cumprod(1 + sampled)

    terminal = paths[:, -1]
    alpha    = 1 - confidence

    result = {
        "median_terminal":    float(np.median(terminal)),
        "mean_terminal":      float(np.mean(terminal)),
        "ci_lower":           float(np.percentile(terminal, alpha / 2 * 100)),
        "ci_upper":           float(np.percentile(terminal, (1 - alpha / 2) * 100)),
        "prob_profit":        float(np.mean(terminal > init)),
        "prob_drawdown_20":   float(np.mean(np.min(paths, axis=1) < init * 0.80)),
        "expected_shortfall": float(np.mean(terminal[terminal <= np.percentile(terminal, 5)])),
        "paths":              paths,
    }

    print(f"\n-- Monte Carlo ({n_paths} paths) --")
    print(f"  Median terminal equity : ${result['median_terminal']:>12,.2f}")
    print(f"  Mean  terminal equity  : ${result['mean_terminal']:>12,.2f}")
    print(f"  {confidence*100:.0f}% CI                : "
          f"[${result['ci_lower']:>10,.2f}, ${result['ci_upper']:>10,.2f}]")
    print(f"  P(profit)              : {result['prob_profit']*100:>7.1f}%")
    print(f"  P(drawdown > 20%)      : {result['prob_drawdown_20']*100:>7.1f}%")

    return result


# Scenario comparison table 

def scenario_comparison_table(scenario_results: Dict[str, Dict]) -> pd.DataFrame:
    """Print and return a side-by-side comparison of all risk profiles."""
    metrics = [
        ("Total Return %",  "total_return_pct"),
        ("Ann. Return %",   "annualised_return_pct"),
        ("Ann. Vol %",      "annualised_vol_pct"),
        ("Sharpe",          "sharpe_ratio"),
        ("Sortino",         "sortino_ratio"),
        ("Max DD %",        "max_drawdown_pct"),
        ("Calmar",          "calmar_ratio"),
        ("Win Rate %",      "win_rate"),
        ("Profit Factor",   "profit_factor"),
    ]

    rows = []
    for label, key in metrics:
        row = {"Metric": label}
        for profile_name, res in scenario_results.items():
            combined = res.get("combined", {})
            val = combined.get(key, "N/A")
            if isinstance(val, float):
                row[profile_name] = f"{val:.3f}" if abs(val) < 1000 else f"{val:,.0f}"
            else:
                row[profile_name] = str(val)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Metric")
    print("\n" + "=" * 70)
    print("  SCENARIO COMPARISON -- Trend Following, $100,000 capital")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70 + "\n")
    return df
