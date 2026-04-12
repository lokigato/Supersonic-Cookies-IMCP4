"""
performance.py
==============

Metrics computed
----------------
* Total return (%)
* Annualised return (%)
* Sharpe ratio (annualised)
* Sortino ratio
* Calmar ratio
* Maximum drawdown (%)
* Maximum drawdown duration (bars)
* Win rate & Win/Loss ratio
* Profit factor
* Average trade PnL
* Value at Risk (95% & 99% historical)
* Expected Shortfall (CVaR)
* Consecutive wins/losses
* Recovery factor

"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Equity curve helpers 
def returns_from_equity(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Rolling drawdown as fraction: (peak - current) / peak."""
    peak = equity.cummax()
    return (equity - peak) / peak


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Maximum number of consecutive bars spent below the previous peak."""
    dd      = drawdown_series(equity)
    in_dd   = dd < 0
    max_dur = 0
    cur_dur = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur  = max(max_dur, cur_dur)
        else:
            cur_dur  = 0
    return max_dur


# Risk-adjusted returns 
def sharpe_ratio(returns: pd.Series, rf: float = 0.0, annualise: int = 252) -> float:
    """Annualised Sharpe ratio. rf is daily risk-free rate."""
    excess = returns - rf / annualise
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(annualise))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, annualise: int = 252) -> float:
    """Sortino ratio using downside deviation."""
    excess      = returns - rf / annualise
    downside    = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf")
    down_dev    = downside.std() * np.sqrt(annualise)
    ann_excess  = excess.mean() * annualise
    return float(ann_excess / down_dev)


def calmar_ratio(equity: pd.Series, annualise: int = 252) -> float:
    """Calmar ratio: annualised return / |max drawdown|."""
    ret = returns_from_equity(equity)
    ann = ret.mean() * annualise
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return float("inf")
    return float(ann / mdd)


# Trade statistics

def trade_statistics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Given a trades DataFrame with at minimum a 'pnl' column,
    compute win-rate, profit-factor, avg trade, etc.
    """
    if trades_df.empty or "pnl" not in trades_df.columns:
        return {
            "n_trades":      0,
            "win_rate":      0.0,
            "win_loss_ratio": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_win":       0.0,
            "avg_loss":      0.0,
            "max_consec_wins":   0,
            "max_consec_losses": 0,
        }

    pnl  = trades_df["pnl"].dropna()
    wins = pnl[pnl > 0]
    loss = pnl[pnl <= 0]

    win_rate      = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
    avg_win       = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss      = float(loss.mean()) if len(loss) > 0 else 0.0
    win_loss_r    = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    profit_factor = float(wins.sum() / abs(loss.sum())) if loss.sum() != 0 else float("inf")

    # Consecutive wins/losses
    signs    = (pnl > 0).astype(int).values
    max_cw   = max_cl = cur_cw = cur_cl = 0
    for s in signs:
        if s == 1:
            cur_cw += 1; cur_cl = 0; max_cw = max(max_cw, cur_cw)
        else:
            cur_cl += 1; cur_cw = 0; max_cl = max(max_cl, cur_cl)

    return {
        "n_trades":          int(len(pnl)),
        "win_rate":          round(win_rate, 4),
        "win_loss_ratio":    round(win_loss_r, 4),
        "profit_factor":     round(profit_factor, 4),
        "avg_trade_pnl":     round(float(pnl.mean()), 4),
        "avg_win":           round(avg_win, 4),
        "avg_loss":          round(avg_loss, 4),
        "max_consec_wins":   max_cw,
        "max_consec_losses": max_cl,
    }


def estimate_trade_pnl(
    trades_df: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Best-effort PnL estimation when trades don't carry explicit PnL.
    Uses entry price at trade time and next-available close for exit.
    """
    if trades_df.empty:
        return trades_df

    df   = trades_df.copy()
    pnls = []

    for _, row in df.iterrows():
        product = row.get("product")
        t_idx   = int(row.get("t", 0))
        qty     = float(row.get("qty", 0))
        price   = float(row.get("price", 0))
        side    = row.get("side", "BUY")

        if product not in prices.columns or t_idx + 1 >= len(prices):
            pnls.append(0.0)
            continue

        exit_price = float(prices[product].iloc[t_idx + 1])
        if side == "BUY":
            pnls.append((exit_price - price) * qty)
        else:
            pnls.append((price - exit_price) * qty)

    df["pnl"] = pnls
    return df


# Tail risk

def var_cvar(
    returns: pd.Series, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Historical VaR and CVaR (Expected Shortfall) at given confidence level.
    Returns (VaR, CVaR) as positive loss magnitudes.
    """
    if len(returns) == 0:
        return 0.0, 0.0
    sorted_ret = returns.sort_values()
    n          = len(sorted_ret)
    cutoff_idx = int(np.floor((1 - confidence) * n))
    var        = float(-sorted_ret.iloc[cutoff_idx])
    cvar       = float(-sorted_ret.iloc[:cutoff_idx + 1].mean())
    return max(var, 0.0), max(cvar, 0.0)


# Full report 

def performance_report(
    equity:      pd.Series,
    trades_df:   Optional[pd.DataFrame] = None,
    prices:      Optional[pd.DataFrame] = None,
    strategy:    str = "Strategy",
    initial_cap: float = 100_000.0,
    rf:          float = 0.05,
    annualise:   int   = 252,
) -> Dict[str, any]:
    """
    Full performance report for a strategy equity curve.

    Returns a dict of metrics, also prints a formatted summary.
    """
    returns = returns_from_equity(equity)

    # Core metrics
    total_ret  = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_bars     = len(returns)
    ann_ret    = (((1 + total_ret / 100) ** (annualise / max(n_bars, 1))) - 1) * 100
    mdd        = max_drawdown(equity) * 100
    mdd_dur    = max_drawdown_duration(equity)
    sharpe     = sharpe_ratio(returns, rf, annualise)
    sortino    = sortino_ratio(returns, rf, annualise)
    calmar     = calmar_ratio(equity, annualise)
    var_95, cvar_95 = var_cvar(returns, 0.95)
    var_99, cvar_99 = var_cvar(returns, 0.99)
    vol_ann    = float(returns.std() * np.sqrt(annualise)) * 100
    recovery   = abs(total_ret / mdd) if mdd != 0 else float("inf")

    # Trade stats
    if trades_df is not None and prices is not None and "pnl" not in trades_df.columns:
        trades_df = estimate_trade_pnl(trades_df, prices)
    t_stats = trade_statistics(trades_df) if trades_df is not None else {}

    report = {
        "strategy":           strategy,
        "initial_capital":    initial_cap,
        "final_equity":       round(float(equity.iloc[-1]), 2),
        "total_return_pct":   round(total_ret, 2),
        "annualised_return_pct": round(ann_ret, 2),
        "annualised_vol_pct": round(vol_ann, 2),
        "sharpe_ratio":       round(sharpe, 3),
        "sortino_ratio":      round(sortino, 3),
        "calmar_ratio":       round(calmar, 3),
        "max_drawdown_pct":   round(mdd, 2),
        "max_dd_duration_bars": mdd_dur,
        "recovery_factor":    round(recovery, 3),
        "var_95_pct":         round(var_95 * 100, 3),
        "cvar_95_pct":        round(cvar_95 * 100, 3),
        "var_99_pct":         round(var_99 * 100, 3),
        "cvar_99_pct":        round(cvar_99 * 100, 3),
        **t_stats,
    }

    # Pretty print
    sep = "-" * 52
    print(f"\n{'=' * 52}")
    print(f"  PERFORMANCE REPORT | {strategy}")
    print(f"{'=' * 52}")
    print(f"  Initial Capital :  ${initial_cap:>12,.2f}")
    print(f"  Final Equity    :  ${report['final_equity']:>12,.2f}")
    print(f"  Total Return    :  {report['total_return_pct']:>+8.2f}%")
    print(f"  Ann. Return     :  {report['annualised_return_pct']:>+8.2f}%")
    print(f"  Ann. Volatility :  {report['annualised_vol_pct']:>8.2f}%")
    print(sep)
    print(f"  Sharpe Ratio    :  {sharpe:>8.3f}")
    print(f"  Sortino Ratio   :  {sortino:>8.3f}")
    print(f"  Calmar Ratio    :  {calmar:>8.3f}")
    print(sep)
    print(f"  Max Drawdown    :  {mdd:>+8.2f}%")
    print(f"  Max DD Duration :  {mdd_dur:>8} bars")
    print(f"  Recovery Factor :  {recovery:>8.3f}")
    print(sep)
    print(f"  VaR  95%        :  {var_95*100:>8.3f}%")
    print(f"  CVaR 95%        :  {cvar_95*100:>8.3f}%")
    print(f"  VaR  99%        :  {var_99*100:>8.3f}%")
    print(f"  CVaR 99%        :  {cvar_99*100:>8.3f}%")
    if t_stats:
        print(sep)
        print(f"  N Trades        :  {t_stats.get('n_trades', 0):>8}")
        print(f"  Win Rate        :  {t_stats.get('win_rate', 0)*100:>8.1f}%")
        print(f"  Win/Loss Ratio  :  {t_stats.get('win_loss_ratio', 0):>8.3f}")
        print(f"  Profit Factor   :  {t_stats.get('profit_factor', 0):>8.3f}")
        print(f"  Avg Trade PnL   :  ${t_stats.get('avg_trade_pnl', 0):>11.2f}")
        print(f"  Max Consec Wins :  {t_stats.get('max_consec_wins', 0):>8}")
        print(f"  Max Consec Loss :  {t_stats.get('max_consec_losses', 0):>8}")
    print(f"{'=' * 52}\n")

    return report


def multi_strategy_comparison(reports: List[Dict]) -> pd.DataFrame:
    """
    Create a summary comparison table across multiple strategy reports.
    """
    key_metrics = [
        "strategy", "total_return_pct", "annualised_return_pct",
        "annualised_vol_pct", "sharpe_ratio", "sortino_ratio",
        "max_drawdown_pct", "win_rate", "profit_factor", "calmar_ratio",
    ]
    rows = []
    for r in reports:
        rows.append({k: r.get(k, "N/A") for k in key_metrics})
    df = pd.DataFrame(rows).set_index("strategy")
    return df
