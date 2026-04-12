"""
data_feed.py
============
DATA MODE
---------
Tries Yahoo Finance FIRST for every product. Falls back to synthetic data
only for products that fail to download. A status table always tells you
which products are real vs synthetic.

Prosperity product -> real market proxy
---------------------------------------
RAINFOREST_RESIN  ->  GLD   Gold ETF         (stable, mean-reverting)
STARFRUIT         ->  CORN  Teucrium Corn     (trending soft commodity)
KELP              ->  XOM   ExxonMobil        (energy, moderate trend)
SQUID_INK         ->  NVDA  Nvidia            (high-vol momentum)
COCONUT           ->  SPY   S&P 500 ETF       (options underlier)
COCONUT_COUPON    ->  BSM call derived from SPY (always synthetic/derived)
GIFT_BASKET       ->  XLP   Consumer Staples  (basket ETF proxy)
CROISSANTS        ->  WEAT  Teucrium Wheat    (basket component 1)
JAM               ->  SOYB  Teucrium Soybean  (basket component 2)
DJEMBE            ->  DBA   Agriculture ETF   (basket component 3)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Product -> Yahoo Finance ticker mapping
PRODUCT_TICKERS: Dict[str, str] = {
    "RAINFOREST_RESIN": "GLD",
    "STARFRUIT":        "CORN",
    "KELP":             "XOM",
    "SQUID_INK":        "NVDA",
    "COCONUT":          "SPY",
    "COCONUT_COUPON":   "SPY",   # always derived via BSM -- not downloaded directly
    "GIFT_BASKET":      "XLP",
    "CROISSANTS":       "WEAT",
    "JAM":              "SOYB",
    "DJEMBE":           "DBA",
}

STAT_ARB_PAIRS: List[Tuple[str, str]] = [
    ("GIFT_BASKET", "CROISSANTS"),
    ("GIFT_BASKET", "JAM"),
    ("GIFT_BASKET", "DJEMBE"),
    ("RAINFOREST_RESIN", "KELP"),
    ("COCONUT", "SQUID_INK"),
]

# Data structures
@dataclass
class Bar:
    timestamp: pd.Timestamp
    product:   str
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    mid:       float = field(init=False)

    def __post_init__(self):
        self.mid = (self.high + self.low) / 2.0


@dataclass
class MarketSnapshot:
    timestamp:    int
    bars:         Dict[str, Bar]
    history:      Dict[str, pd.DataFrame]
    order_depths: Dict[str, dict]

# Yahoo Finance download
def _download_yf(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download one ticker. Raises ValueError on any failure."""
    import yfinance as yf
    df = yf.download(
        ticker, start=start, end=end,
        interval=interval, auto_adjust=True, progress=False
    )
    if df.empty:
        raise ValueError(f"Empty response for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} rows -- too short to use")
    return df


# Synthetic fallback generators (per product type)
def _synthetic_for(product: str, n: int, dates: pd.DatetimeIndex,
                   rng: np.random.Generator) -> pd.DataFrame:
    """Generate plausible synthetic OHLCV for a single product."""
    if product == "RAINFOREST_RESIN":
        close = 10000 + np.cumsum(rng.normal(0, 2, n)) * 0.05
    elif product in ("STARFRUIT", "CORN"):
        close = 5000 * np.cumprod(1 + rng.normal(0.0005, 0.013, n))
    elif product in ("KELP", "XOM"):
        close = 2000 * np.cumprod(1 + rng.normal(0.0003, 0.010, n))
    elif product in ("SQUID_INK", "NVDA"):
        close = 3000 * np.cumprod(1 + rng.normal(0.001, 0.025, n))
    elif product in ("COCONUT", "SPY"):
        close = 8000 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    elif product in ("GIFT_BASKET", "XLP"):
        crop  = 4000 * np.cumprod(1 + rng.normal(0.0003, 0.010, n))
        jam   = 1500 * np.cumprod(1 + rng.normal(0.0002, 0.009, n))
        close = 3 * crop + 2 * jam + rng.normal(0, 80, n)
    elif product in ("CROISSANTS", "WEAT"):
        close = 4000 * np.cumprod(1 + rng.normal(0.0003, 0.010, n))
    elif product in ("JAM", "SOYB"):
        close = 1500 * np.cumprod(1 + rng.normal(0.0002, 0.009, n))
    elif product in ("DJEMBE", "DBA"):
        close = 1300 * np.cumprod(1 + rng.normal(0.0001, 0.008, n))
    else:
        close = 1000 * np.cumprod(1 + rng.normal(0.0002, 0.012, n))

    high   = close * rng.uniform(1.000, 1.015, n)
    low    = close * rng.uniform(0.985, 1.000, n)
    open_  = close * rng.uniform(0.990, 1.010, n)
    volume = np.abs(rng.normal(1e6, 2e5, n))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates[:n]
    )


# Main public API
def fetch_market_data(
    products:        Optional[List[str]] = None,
    start:           str  = "2022-01-01",
    end:             str  = "2024-12-31",
    interval:        str  = "1d",
    force_synthetic: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Download real market data from Yahoo Finance. Falls back to synthetic
    data for any product that fails to download.

    Parameters
    ----------
    force_synthetic : set True to skip Yahoo Finance entirely (offline mode)
    """
    if products is None:
        products = list(PRODUCT_TICKERS.keys())

    date_grid = pd.bdate_range(start=start, end=end)
    n_bars    = len(date_grid)
    rng       = np.random.default_rng(42)

    downloaded: Dict[str, Optional[pd.DataFrame]] = {}
    status: Dict[str, str] = {}
    data:   Dict[str, pd.DataFrame] = {}

    print()
    print("=" * 60)
    print("  DATA FEED")
    print(f"  Range  : {start}  ->  {end}")
    print(f"  Mode   : {'SYNTHETIC (forced)' if force_synthetic else 'Yahoo Finance + synthetic fallback'}")
    print("=" * 60)

    for product in products:
        if product == "COCONUT_COUPON":
            continue   # derived separately below

        ticker = PRODUCT_TICKERS[product]

        # -- attempt Yahoo Finance download (once per ticker) --
        if not force_synthetic and ticker not in downloaded:
            try:
                print(f"  {ticker:<6}  [{product}] ... ", end="", flush=True)
                df = _download_yf(ticker, start, end, interval)
                downloaded[ticker] = df
                print(f"OK  ({len(df)} bars)")
                time.sleep(0.25)   # polite rate limiting
            except Exception as exc:
                print(f"FAILED  ({exc})")
                downloaded[ticker] = None

        # -- assign data or fall back to synthetic --
        real_df = downloaded.get(ticker) if not force_synthetic else None
        if real_df is not None:
            df = real_df.copy()
            df["Product"] = product
            data[product] = df
            status[product] = "REAL"
        else:
            df = _synthetic_for(product, n_bars, date_grid, rng)
            df["Product"] = product
            data[product] = df
            status[product] = "SYNTHETIC"

    # -- COCONUT_COUPON: BSM call on COCONUT (real or synthetic) --
    if "COCONUT_COUPON" in products and "COCONUT" in data:
        from scipy.stats import norm as snorm
        spot  = data["COCONUT"]["Close"]
        sigma = spot.pct_change().rolling(20).std().bfill().fillna(0.15) * np.sqrt(252)
        K     = spot.rolling(20).mean().bfill()
        T, r  = 30 / 252, 0.05
        d1    = (np.log(spot / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-9)
        d2    = d1 - sigma * np.sqrt(T)
        call  = np.clip(
            spot * snorm.cdf(d1) - K * np.exp(-r * T) * snorm.cdf(d2),
            1.0, None
        )
        coupon_df = data["COCONUT"].copy()
        coupon_df["Close"]   = call.values
        coupon_df["Open"]    = call.values * 0.995
        coupon_df["High"]    = call.values * 1.010
        coupon_df["Low"]     = call.values * 0.990
        coupon_df["Product"] = "COCONUT_COUPON"
        data["COCONUT_COUPON"] = coupon_df
        src = "BSM on real SPY" if status.get("COCONUT") == "REAL" else "BSM on synthetic"
        status["COCONUT_COUPON"] = f"DERIVED ({src})"

    # -- status summary table --
    print()
    print(f"  {'Product':<20}  {'Ticker':<6}  {'Bars':>5}  {'Source'}")
    print("  " + "-" * 54)
    for p in products:
        if p not in data:
            continue
        ticker = PRODUCT_TICKERS.get(p, "N/A")
        bars   = len(data[p])
        src    = status.get(p, "UNKNOWN")
        flag   = "** " if "SYNTHETIC" in src else "   "
        print(f"  {flag}{p:<20}  {ticker:<6}  {bars:>5}  {src}")
    print()

    n_real = sum(1 for s in status.values() if s == "REAL")
    n_syn  = sum(1 for s in status.values() if "SYNTHETIC" in s)
    if n_syn > 0:
        print(f"  WARNING: {n_syn} product(s) marked ** are using SYNTHETIC data.")
        print("  To fix: check internet connection, then run:")
        print("    pip install -U yfinance")
        print()
    else:
        print(f"  All {n_real} products loaded from Yahoo Finance.")
        print()

    return data


def align_products(
    data: Dict[str, pd.DataFrame],
    products: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Wide DataFrame of Close prices aligned on a common date index."""
    if products is None:
        products = list(data.keys())
    frames = {p: data[p]["Close"].rename(p) for p in products if p in data}
    if not frames:
        raise ValueError("No product data available to align.")
    return pd.concat(frames, axis=1).dropna()


def build_market_snapshot(
    data: Dict[str, pd.DataFrame],
    timestamp_idx: int,
    spread_bps: float = 10.0,
) -> MarketSnapshot:
    """Construct a MarketSnapshot at a given bar index."""
    bars: Dict[str, Bar] = {}
    order_depths: Dict[str, dict] = {}
    for product, df in data.items():
        if timestamp_idx >= len(df):
            continue
        row = df.iloc[timestamp_idx]
        bar = Bar(
            timestamp=df.index[timestamp_idx],
            product=product,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
        )
        bars[product] = bar
        half_spread = bar.mid * spread_bps / 10_000
        order_depths[product] = {
            "best_bid":   bar.mid - half_spread,
            "best_ask":   bar.mid + half_spread,
            "bid_volume": int(bar.volume * 0.1),
            "ask_volume": int(bar.volume * 0.1),
        }
    return MarketSnapshot(
        timestamp=timestamp_idx,
        bars=bars,
        history={p: df.iloc[:timestamp_idx + 1] for p, df in data.items()},
        order_depths=order_depths,
    )
