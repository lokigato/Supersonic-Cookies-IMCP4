"""
strategies/trend_following.py
==============================
Trend-following strategy

Approach:

* Dual EMA crossover (fast / slow) to detect trend direction.
* Average True Range (ATR) for volatility-adjusted position sizing --
  ensures equal-risk allocation across products with different volatilities.
* Optional Hurst exponent filter: only trade in trending regime (H > 0.5).
  Below H ~= 0.5 the product is more mean-reverting than trending.
* Momentum confirmation: require that the rolling n-bar return also
  confirms the EMA crossover direction before entering.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Configuration 

@dataclass
class TrendFollowingConfig:
    fast_ema:           int   = 8
    slow_ema:           int   = 21
    atr_period:         int   = 14
    # Risk per trade as fraction of capital (ATR-based sizing)
    risk_per_trade:     float = 0.01
    # Momentum confirmation window
    momentum_window:    int   = 10
    # Hurst filter: only go long/short when H > this threshold
    hurst_threshold:    float = 0.52
    hurst_window:       int   = 50
    # Max position fraction of capital
    max_position_frac:  float = 0.25
    products:           List[str] = field(default_factory=lambda: [
        "STARFRUIT", "SQUID_INK", "KELP"
    ])


# Indicators

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def hurst_exponent(ts: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent via rescaled range (R/S) analysis.
    H > 0.5 -> trending, H < 0.5 -> mean-reverting, H ~= 0.5 -> random walk.
    """
    if len(ts) < max_lag * 2:
        return 0.5   # insufficient data -- treat as random walk
    lags   = range(min_lag, min(max_lag, len(ts) // 2))
    tau    = []
    rs_arr = []
    for lag in lags:
        n_blocks = len(ts) // lag
        rs_values = []
        for i in range(n_blocks):
            block = ts[i * lag: (i + 1) * lag]
            mean  = block.mean()
            demeaned = block - mean
            cum_dev  = np.cumsum(demeaned)
            R        = cum_dev.max() - cum_dev.min()
            S        = block.std(ddof=1)
            if S > 0:
                rs_values.append(R / S)
        if rs_values:
            tau.append(lag)
            rs_arr.append(np.mean(rs_values))

    if len(tau) < 2:
        return 0.5
    log_tau = np.log(tau)
    log_rs  = np.log(rs_arr)
    slope, _ = np.polyfit(log_tau, log_rs, 1)
    return float(np.clip(slope, 0.01, 0.99))


def rolling_hurst(series: pd.Series, window: int = 50) -> pd.Series:
    """
    Fast Hurst approximation via variance ratio:
        Var(k-step returns) / (k * Var(1-step returns))
    H > 0.5 -> trending, H < 0.5 -> mean-reverting.
    Much faster than full R/S analysis.
    """
    ret = series.pct_change()
    var1 = ret.rolling(window).var()
    ret2 = series.pct_change(2)
    var2 = ret2.rolling(window).var()
    # VR = Var(2-step) / (2 * Var(1-step));  H ~= 0.5*log(VR)/log(2) + 0.5
    vr = var2 / (2 * var1 + 1e-12)
    h  = 0.5 * np.log(vr.clip(1e-6, 10)) / np.log(2) + 0.5
    return h.clip(0.01, 0.99)


# Strategy class

class TrendFollowingStrategy:
    """
    Dual-EMA crossover with ATR position sizing and Hurst regime filter.

    Signal logic (per product, per bar):
    -------------------------------------
        fast_ema > slow_ema  AND  momentum > 0  AND  H > threshold -> BUY
        fast_ema < slow_ema  AND  momentum < 0  AND  H > threshold -> SELL
        H < threshold                                               -> FLAT (mean-reverting regime)
    """

    def __init__(self, config: TrendFollowingConfig, capital: float = 100_000.0):
        self.cfg     = config
        self.capital = capital
        self.positions: Dict[str, float] = {p: 0.0 for p in config.products}
        self.trades:    List[dict]       = []
        self._equity_curve: List[float]  = [capital]
        self._last_prices:  Dict[str, float] = {}

    # Feature computation

    def _compute_features(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Return (fast_ema, slow_ema, atr_series, momentum, hurst)."""
        close   = df["Close"]
        high    = df["High"]  if "High"  in df.columns else close
        low     = df["Low"]   if "Low"   in df.columns else close

        f_ema   = ema(close, self.cfg.fast_ema)
        s_ema   = ema(close, self.cfg.slow_ema)
        atr_s   = atr(high, low, close, self.cfg.atr_period)
        mom     = close.pct_change(self.cfg.momentum_window)
        h_exp   = rolling_hurst(close, self.cfg.hurst_window)

        return f_ema, s_ema, atr_s, mom, h_exp

    # Signal

    def generate_signals(
        self, history: Dict[str, pd.DataFrame], t: int
    ) -> Dict[str, float]:
        """
        Returns a signal dict: +1 long, -1 short, 0 flat.
        Magnitude encodes conviction (1 = baseline, up to 2 = strong trend).
        """
        signals: Dict[str, float] = {}
        min_bars = max(self.cfg.slow_ema, self.cfg.hurst_window) + 5

        for product in self.cfg.products:
            if product not in history:
                signals[product] = 0.0
                continue
            df = history[product]
            if len(df) < min_bars:
                signals[product] = 0.0
                continue

            f_ema, s_ema, atr_s, mom, h_exp = self._compute_features(df)

            last_fast = f_ema.iloc[-1]
            last_slow = s_ema.iloc[-1]
            last_mom  = mom.iloc[-1]
            last_h    = h_exp.iloc[-1]

            if np.isnan(last_fast) or np.isnan(last_slow) or np.isnan(last_h):
                signals[product] = 0.0
                continue

            # Hurst regime filter
            if last_h < self.cfg.hurst_threshold:
                signals[product] = 0.0
                continue

            # Trend direction
            if last_fast > last_slow and last_mom > 0:
                # Conviction: how far above the slow EMA (normalised by ATR)
                gap = (last_fast - last_slow) / (atr_s.iloc[-1] + 1e-9)
                signals[product] = min(1.0 + gap * 0.5, 2.0)
            elif last_fast < last_slow and last_mom < 0:
                gap = (last_slow - last_fast) / (atr_s.iloc[-1] + 1e-9)
                signals[product] = -min(1.0 + gap * 0.5, 2.0)
            else:
                signals[product] = 0.0

        return signals

    # Position sizing

    def _atr_position_size(
        self, product: str, history: Dict[str, pd.DataFrame], price: float
    ) -> float:
        """
        Risk-based position sizing:
            qty = (capital * risk_per_trade) / ATR
        Caps at max_position_frac * capital / price.
        """
        df = history.get(product)
        if df is None or len(df) < self.cfg.atr_period + 2:
            base_qty = self.cfg.risk_per_trade * self.capital / (price + 1e-9)
            return min(base_qty, self.cfg.max_position_frac * self.capital / price)

        close = df["Close"]
        high  = df["High"]  if "High"  in df.columns else close
        low   = df["Low"]   if "Low"   in df.columns else close
        atr_v = atr(high, low, close, self.cfg.atr_period).iloc[-1]
        if np.isnan(atr_v) or atr_v <= 0:
            atr_v = price * 0.02  # fallback: 2% of price

        qty = self.cfg.risk_per_trade * self.capital / atr_v
        max_qty = self.cfg.max_position_frac * self.capital / price
        return min(qty, max_qty)

    # Order generation 

    def generate_orders(
        self,
        history: Dict[str, pd.DataFrame],
        t: int,
        mid_prices: Optional[Dict[str, float]] = None,
    ) -> Dict[str, dict]:
        signals = self.generate_signals(history, t)
        orders:  Dict[str, dict] = {}

        for product, signal in signals.items():
            df = history.get(product)
            if df is None or df.empty:
                continue

            price = (
                mid_prices.get(product)
                if mid_prices
                else float(df["Close"].iloc[-1])
            )
            if not price or np.isnan(price):
                continue

            sized_qty    = self._atr_position_size(product, history, price)
            target_qty   = np.sign(signal) * sized_qty * abs(signal)
            current_qty  = self.positions.get(product, 0.0)
            delta_qty    = target_qty - current_qty

            if abs(delta_qty) < 0.01:
                orders[product] = {"side": "HOLD", "qty": 0.0, "price": price}
                continue

            side = "BUY" if delta_qty > 0 else "SELL"
            orders[product] = {
                "side":  side,
                "qty":   abs(delta_qty),
                "price": price,
            }

            self.capital               -= delta_qty * price
            self.positions[product]     = target_qty
            self._last_prices[product]  = price
            self.trades.append({
                "t": t, "product": product, "side": side,
                "qty": abs(delta_qty), "price": price, "hurst": 0.0,
            })

        mtm = self.capital + sum(
            self.positions.get(p, 0.0) * (
                mid_prices.get(p) if mid_prices else float(history[p]["Close"].iloc[-1])
            )
            for p in self.positions if p in (mid_prices or history)
        )
        self._equity_curve.append(mtm)
        return orders

    # Results 

    def get_equity_curve(self) -> pd.Series:
        return pd.Series(self._equity_curve, name="TrendFollowing_Equity")

    def get_trades(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

    def reset(self):
        self.positions      = {p: 0.0 for p in self.cfg.products}
        self.trades         = []
        self._equity_curve  = [self.capital]
        self._last_prices   = {}
