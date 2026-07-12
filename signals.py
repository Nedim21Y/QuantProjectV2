"""
QuantProject v2 - signals.py
Loads the latest trained model and generates current BUY/SELL rankings.

Run: python signals.py

Output: console table of top 10 LONG / top 10 SHORT signals with scores.
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import config
from main import compute_features, fetch_fundamentals


def load_model() -> tuple:
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {config.MODEL_PATH}. Run main.py first."
        )
    with open(config.MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    # Support both old single-model format and new 3-model stack format
    if "model" in bundle:
        return bundle["model"], bundle["feature_cols"]
    return bundle, bundle["feature_cols"]


def download_recent_data(n_weeks: int = 60) -> dict:
    """
    Download the last n_weeks of weekly OHLCV for the universe + SPY.
    Uses a fresh yfinance call (no cache) to get most current data.
    """
    import datetime
    end   = datetime.datetime.today().strftime("%Y-%m-%d")
    start = (datetime.datetime.today() -
             datetime.timedelta(weeks=n_weeks + 5)).strftime("%Y-%m-%d")

    print(f"  Downloading recent data ({n_weeks}w, to {end})...")
    all_tickers = config.UNIVERSE + ["SPY"]
    raw = yf.download(
        all_tickers,
        start=start,
        end=end,
        interval="1wk",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns.names = ["field", "ticker"]

    field_map = {"Close": "close", "High": "high", "Low": "low", "Volume": "volume"}
    data = {}
    for src, dst in field_map.items():
        if src in raw.columns.get_level_values(0):
            data[dst] = raw[src]

    spy_col = [c for c in data["close"].columns if c == "SPY"]
    uni_col = [c for c in data["close"].columns if c != "SPY"]

    result = {
        "close":     data["close"][uni_col],
        "high":      data["high"][uni_col],
        "low":       data["low"][uni_col],
        "volume":    data["volume"][uni_col],
        "spy_close": data["close"][spy_col].squeeze().rename("SPY"),
    }
    return result


def generate_signals(model, feature_cols: list, top_n: int = 10) -> pd.DataFrame:
    """
    Compute features on the most recent week and score every stock.
    Returns DataFrame sorted by score (highest = strongest buy signal).
    """
    data    = download_recent_data(n_weeks=config.LOOKBACK_WEEKS + 5)
    fund_df = fetch_fundamentals(config.UNIVERSE)
    panel   = compute_features(data, fund_df)

    # Get the most recent date with complete feature data
    recent_dates = panel.index.get_level_values("Date").unique().sort_values()
    if len(recent_dates) == 0:
        raise ValueError("No complete feature rows found in recent data.")

    latest_date = recent_dates[-1]
    latest      = panel.xs(latest_date, level="Date")
    latest      = latest.dropna(subset=feature_cols)

    if latest.empty:
        raise ValueError(f"No valid rows on {latest_date.date()}. "
                         "Try increasing n_weeks or check data quality.")

    X = latest[feature_cols].values
    # Stack bundle has lgbm/xgb/ridge keys; legacy bundle is a single model
    if isinstance(model, dict) and "lgbm" in model:
        s_lgbm  = model["lgbm"].predict(X)
        s_xgb   = model["xgb"].predict(X)
        s_ridge = model["ridge"].predict(model["scaler"].transform(X))
        scores  = (s_lgbm + s_xgb + s_ridge) / 3.0
    else:
        scores = model.predict(X)
    latest = latest.copy()
    latest["score"] = scores
    latest["rank"]  = latest["score"].rank(pct=True, ascending=True)

    def strength(pct):
        if pct >= 0.95:   return "STRONG BUY  ▲▲▲"
        elif pct >= 0.85: return "BUY         ▲▲ "
        elif pct >= 0.75: return "WEAK BUY    ▲  "
        elif pct <= 0.05: return "STRONG SELL ▼▼▼"
        elif pct <= 0.15: return "SELL        ▼▼ "
        elif pct <= 0.25: return "WEAK SELL   ▼  "
        else:             return "NEUTRAL     -  "

    latest["signal"]  = latest["rank"].apply(strength)
    latest_sorted     = latest.sort_values("score", ascending=False)

    print(f"\n  Signal date: {latest_date.date()} (next-week forecast)")
    print(f"  Universe scored: {len(latest_sorted)} stocks\n")

    print_signals(latest_sorted, top_n, latest_date)
    return latest_sorted


def print_signals(df: pd.DataFrame, top_n: int, signal_date):
    sep = "═" * 62
    print(f"{sep}")
    print(f"  QUANTPROJECT v2 - SIGNALS as of {signal_date.date()}")
    print(f"  Regime: SPY vs 40w MA determines long eligibility")
    print(f"{sep}")

    print(f"\n  ▲ TOP {top_n} LONG SIGNALS (highest predicted return)")
    print(f"  {'#':<4} {'Ticker':<8} {'Signal':<18} {'Model Score':>12}")
    print(f"  {'─'*4} {'─'*8} {'─'*18} {'─'*12}")
    for i, (ticker, row) in enumerate(df.head(top_n).iterrows(), 1):
        print(f"  {i:<4} {ticker:<8} {row['signal']:<18} {row['score']:>12.5f}")

    print(f"\n  ▼ TOP {top_n} SHORT SIGNALS (lowest predicted return)")
    print(f"  {'#':<4} {'Ticker':<8} {'Signal':<18} {'Model Score':>12}")
    print(f"  {'─'*4} {'─'*8} {'─'*18} {'─'*12}")
    for i, (ticker, row) in enumerate(df.tail(top_n).iloc[::-1].iterrows(), 1):
        print(f"  {i:<4} {ticker:<8} {row['signal']:<18} {row['score']:>12.5f}")

    print(f"\n{sep}")
    print("  DISCLAIMER: For educational purposes only. Not financial advice.")
    print("  Past model performance does not guarantee future returns.")
    print(f"{sep}\n")


if __name__ == "__main__":
    print("Loading model...")
    model, feature_cols = load_model()
    print(f"  Model loaded from {config.MODEL_PATH}")
    generate_signals(model, feature_cols, top_n=10)
