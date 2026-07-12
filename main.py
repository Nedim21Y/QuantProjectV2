"""
QuantProject v2 - main.py
End-to-end pipeline: Download → Features → Walk-Forward Train → Backtest → Save

Run: python main.py

Bias notes:
  - Survivorship bias: yfinance only provides current S&P 500 constituents;
    delisted stocks are absent, inflating backtest returns.
  - No lookahead bias: features at week t use only data up to close of week t;
    targets use week t+1 close; model is never trained on future data.
"""

import os
import sys
import warnings
import logging
import pickle
import time

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
import config

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 - DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════

_DOWNLOAD_BATCH_SIZE = 100   # tickers per yfinance request
_MIN_WEEKS           = 100   # drop tickers with fewer valid weekly bars


def _raw_to_fields(raw: pd.DataFrame, batch: list[str]) -> dict:
    """Normalise a yfinance download result to {field: wide_DataFrame}."""
    if raw is None or raw.empty:
        return {}
    raw.index = pd.to_datetime(raw.index)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns.names = ["field", "ticker"]
    else:
        # Single-ticker batch: yfinance returns flat columns
        raw = pd.concat({batch[0]: raw.T}).T
        raw.columns = pd.MultiIndex.from_product([raw.columns, [batch[0]]])
        raw.columns.names = ["field", "ticker"]

    field_map = {"Close": "close", "High": "high", "Low": "low",
                 "Volume": "volume", "Open": "open"}
    return {field_map[f]: raw[f]
            for f in raw.columns.get_level_values("field").unique()
            if f in field_map}


def download_data() -> dict[str, pd.DataFrame]:
    """
    Downloads weekly OHLCV for all universe tickers + SPY.
    Downloads in batches of _DOWNLOAD_BATCH_SIZE; a failed batch is logged
    and skipped so one bad ticker cannot abort the full download.
    Caches to data/prices.parquet; subsequent runs load from cache.

    Returns dict with keys: 'close', 'high', 'low', 'volume', 'spy_close'.
    Each value is a wide DataFrame: index=Date, columns=tickers.
    """
    config.PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if config.PRICES_PATH.exists():
        log.info("Loading prices from cache: %s", config.PRICES_PATH)
        store = pd.read_parquet(config.PRICES_PATH)
        result = {}
        for field in store.columns.get_level_values(0).unique():
            result[field] = store[field]
        if isinstance(result.get("spy_close"), pd.DataFrame):
            result["spy_close"] = result["spy_close"].squeeze().rename("SPY")
        log.info("Cache loaded: %d weeks × %d tickers",
                 len(result["close"]), result["close"].shape[1])
        return result

    # Always include SPY (needed for regime filter + idio_vol)
    all_tickers = ["SPY"] + [t for t in config.UNIVERSE if t != "SPY"]
    batches = [all_tickers[i : i + _DOWNLOAD_BATCH_SIZE]
               for i in range(0, len(all_tickers), _DOWNLOAD_BATCH_SIZE)]

    log.info("Downloading %d tickers in %d batches (batch size %d)...",
             len(all_tickers), len(batches), _DOWNLOAD_BATCH_SIZE)

    field_frames: dict[str, list[pd.DataFrame]] = {
        "close": [], "high": [], "low": [], "volume": []
    }
    failed_batches   = 0
    successful_ticks = 0

    for i, batch in enumerate(batches, 1):
        log.info("  Batch %d/%d - %d tickers (%s … %s)",
                 i, len(batches), len(batch), batch[0], batch[-1])
        try:
            raw = yf.download(
                batch,
                start=config.START_DATE,
                end=config.END_DATE,
                interval="1wk",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            fields = _raw_to_fields(raw, batch)
            if not fields or "close" not in fields:
                log.warning("  Batch %d: empty response, skipping", i)
                failed_batches += 1
                continue

            # Per-ticker quality filter: drop columns with < _MIN_WEEKS valid bars
            close_batch = fields["close"]
            valid = close_batch.columns[
                close_batch.notna().sum() >= _MIN_WEEKS
            ].tolist()
            bad = set(close_batch.columns) - set(valid)
            if bad:
                log.warning("  Batch %d: dropped %d thin tickers: %s",
                            i, len(bad), sorted(bad))

            if not valid:
                log.warning("  Batch %d: no tickers passed quality filter", i)
                failed_batches += 1
                continue

            for f in field_frames:
                if f in fields:
                    field_frames[f].append(fields[f][valid])

            successful_ticks += len(valid)
            log.info("  Batch %d: %d/%d tickers passed ✓",
                     i, len(valid), len(batch))

        except Exception as exc:
            log.warning("  Batch %d failed - %s: %s",
                        i, type(exc).__name__, str(exc)[:120])
            failed_batches += 1
            continue

    if not field_frames["close"]:
        raise RuntimeError(
            "Every download batch failed. Check network / yfinance version."
        )

    log.info("Download complete: %d tickers across %d batches "
             "(%d batches failed or skipped)",
             successful_ticks, len(batches), failed_batches)

    # Concatenate batches along columns
    combined: dict[str, pd.DataFrame] = {}
    for f, frames in field_frames.items():
        combined[f] = pd.concat(frames, axis=1)

    # Normalize all fields to a unified weekly frequency ending Friday.
    # Without this, different yfinance batches return week-start dates on
    # different weekdays (Mon vs Fri) which doubles the row count on concat.
    for f in combined:
        combined[f] = combined[f].resample("W-FRI").last()
    date_idx = combined["close"].index

    # Separate SPY from universe
    all_cols      = combined["close"].columns.tolist()
    universe_cols = [c for c in all_cols if c != "SPY"]
    spy_present   = "SPY" in all_cols

    if not spy_present:
        raise RuntimeError("SPY download failed - required for regime filter.")
    if len(universe_cols) < 50:
        raise RuntimeError(
            f"Only {len(universe_cols)} universe tickers downloaded "
            "(need ≥ 50). Check network."
        )

    log.info("Final universe: %d tickers (target ≥ 400)", len(universe_cols))

    result = {
        "close":     combined["close"][universe_cols],
        "high":      combined["high"][universe_cols],
        "low":       combined["low"][universe_cols],
        "volume":    combined["volume"][universe_cols],
        "spy_close": combined["close"]["SPY"].rename("SPY"),
    }

    # Persist to parquet
    frames_to_save = {k: v for k, v in result.items()
                      if isinstance(v, pd.DataFrame)}
    frames_to_save["spy_close"] = result["spy_close"].to_frame("SPY")
    store = pd.concat(frames_to_save, axis=1)
    store.to_parquet(config.PRICES_PATH)
    log.info("Cached to %s (%d weeks, %d tickers)",
             config.PRICES_PATH, len(date_idx), len(universe_cols))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1b - FUNDAMENTAL DATA (yfinance .info)
# ═══════════════════════════════════════════════════════════════════════════

# yfinance field name → internal column name
_FUND_FIELDS = {
    "trailingPE":    "pe_ratio",
    "priceToBook":   "pb_ratio",
    "profitMargins": "profit_margin",
    "revenueGrowth": "revenue_growth",
    "debtToEquity":  "debt_to_equity",
}
_FUND_COLS = list(_FUND_FIELDS.values())   # ["pe_ratio", "pb_ratio", ...]


def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch PE, PB, profit margin, revenue growth YoY, and debt-to-equity
    for every ticker using yfinance .info.

    Results are cached to data/fundamentals.json and refreshed only when
    the cache is older than config.FUNDAMENTALS_TTL_DAYS days, so repeated
    pipeline runs don't hit the network.

    Returns a DataFrame indexed by ticker with one column per fundamental.
    Missing values within each column are filled with the cross-sectional
    median so no ticker is entirely excluded for a missing single metric.
    """
    config.FUNDAMENTALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Try loading a fresh-enough cache ─────────────────────────────────
    if config.FUNDAMENTALS_PATH.exists():
        with open(config.FUNDAMENTALS_PATH) as fh:
            cache = json.load(fh)
        cache_age = (pd.Timestamp.now() - pd.Timestamp(cache.get("_date", "2000-01-01"))).days
        if cache_age < config.FUNDAMENTALS_TTL_DAYS:
            log.info("Fundamentals: loaded from cache (%s, %d days old)",
                     cache["_date"], cache_age)
            records = {k: v for k, v in cache.items() if k != "_date"}
            return _build_fund_df(records, tickers)

    # ── Fetch one ticker at a time, in parallel ───────────────────────────
    log.info("Fundamentals: fetching .info for %d tickers "
             "(parallelised, ~1-2 min) …", len(tickers))

    def _fetch_one(ticker: str) -> tuple[str, dict]:
        """Return (ticker, {col: value}) - empty dict on any error."""
        for attempt in range(2):
            try:
                info = yf.Ticker(ticker).info
                # yfinance returns a near-empty dict for unknown tickers
                if not info or len(info) < 5:
                    return ticker, {}
                return ticker, {
                    alias: info.get(yf_key)
                    for yf_key, alias in _FUND_FIELDS.items()
                }
            except Exception:
                if attempt == 0:
                    time.sleep(0.3)   # brief back-off before retry
        return ticker, {}

    records: dict[str, dict] = {}
    failed = 0
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker, row = fut.result()
            records[ticker] = row
            if not row:
                failed += 1
            if i % 100 == 0 or i == len(tickers):
                log.info("  Fundamentals: %d/%d fetched (%d with data, %d empty)",
                         i, len(tickers), i - failed, failed)

    log.info("Fundamentals fetch complete: %d/%d tickers returned data",
             len(tickers) - failed, len(tickers))

    # ── Persist to cache ──────────────────────────────────────────────────
    cache = {"_date": pd.Timestamp.now().strftime("%Y-%m-%d"), **records}
    with open(config.FUNDAMENTALS_PATH, "w") as fh:
        json.dump(cache, fh)
    log.info("Fundamentals cached → %s", config.FUNDAMENTALS_PATH)

    return _build_fund_df(records, tickers)


def _build_fund_df(records: dict, tickers: list[str]) -> pd.DataFrame:
    """
    Convert the raw {ticker: {col: value}} records into a clean DataFrame.
    Cross-sectional median fill: each missing value is replaced by the
    median of that metric across all tickers that did report it.
    """
    df = pd.DataFrame.from_dict(records, orient="index", columns=_FUND_COLS)
    df = df.reindex(tickers)   # ensure every universe ticker is present

    # Convert to numeric (some yfinance values come back as strings)
    for col in _FUND_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cross-sectional median fill
    medians = df.median()
    df = df.fillna(medians)

    null_counts = df.isna().sum()
    if null_counts.any():
        # A column whose median was also NaN (e.g., debtToEquity for financials)
        # → fill remaining NaN with 0 so no row is dropped
        df = df.fillna(0.0)
        log.warning("Fundamentals: %s columns still had NaN after median fill "
                    "(filled with 0): %s",
                    int(null_counts[null_counts > 0].sum()),
                    null_counts[null_counts > 0].to_dict())

    log.info("Fundamentals: %d tickers × %d metrics (0 NaN remaining)",
             len(df), len(_FUND_COLS))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 - FEATURE ENGINEERING (30 FACTORS)
# ═══════════════════════════════════════════════════════════════════════════

def _rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI with smoothed moving average. 0=oversold, 100=overbought."""
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_features(data: dict, fund_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Compute 30 cross-sectional features for every (week, ticker) pair:
      25 technical features (rolling windows, all backward-looking)
      +5 fundamental features (PE, PB, profit margin, revenue growth, D/E)

    Technical features have zero lookahead bias.
    Fundamental features use current yfinance .info values broadcast
    across all dates - a static cross-sectional signal (see config.py note).

    All 30 features are cross-sectionally rank-normalised to [0, 1] within
    each week so the model learns relative ordering, not absolute levels.

    Returns long-format DataFrame: MultiIndex (Date, Ticker), 30 feature cols
    + 'fwd_ret' (raw forward return) + 'fwd_rank' (cross-sectional rank, 0-1).
    """
    close   = data["close"]
    high    = data["high"]
    low     = data["low"]
    volume  = data["volume"]

    # Always work with spy_close as a 1-D Series regardless of cache vs fresh download
    spy_raw = data["spy_close"]
    if isinstance(spy_raw, pd.DataFrame):
        spy_raw = spy_raw.squeeze()
    spy     = spy_raw.reindex(close.index)

    log_ret = np.log(close / close.shift(1))
    spy_ret = np.log(spy   / spy.shift(1))   # Series

    feats: dict[str, pd.DataFrame] = {}

    # ── 1-4. Price momentum (multiple horizons) ───────────────────────
    for weeks, name in [(4, "mom_4w"), (13, "mom_13w"), (26, "mom_26w"), (52, "mom_52w")]:
        feats[name] = close / close.shift(weeks) - 1

    # ── 5. RSI-14 (weekly bars) ───────────────────────────────────────
    feats["rsi_14"] = _rsi(close, 14)

    # ── 6. Volume z-score over 4 weeks ───────────────────────────────
    vol_ma4  = volume.rolling(4).mean()
    vol_std4 = volume.rolling(4).std().replace(0, np.nan)
    feats["vol_zscore_4w"] = (volume - vol_ma4) / vol_std4

    # ── 7. Price vs 52-week high ─────────────────────────────────────
    feats["price_52w_high"] = close / close.rolling(52).max()

    # ── 8. Idiosyncratic volatility (residual from 52w SPY regression) ─
    # Var decomposition: idio_var = total_var - beta² × spy_var
    # This is the "low volatility anomaly" factor - lower idio vol tends to outperform
    rolling_cov_52 = log_ret.rolling(52).cov(spy_ret)
    spy_var_52     = spy_ret.rolling(52).var()
    beta_52        = rolling_cov_52.div(spy_var_52, axis=0)
    total_var_52   = log_ret.rolling(52).var()
    idio_var       = (total_var_52 - (beta_52 ** 2).mul(spy_var_52, axis=0)).clip(lower=0)
    feats["idio_vol"] = np.sqrt(idio_var) * np.sqrt(52)

    # ── 9-10. Relative strength vs SPY ───────────────────────────────
    spy_4w  = spy / spy.shift(4)  - 1
    spy_13w = spy / spy.shift(13) - 1
    feats["rel_str_spy_4w"]  = (close / close.shift(4)  - 1).sub(spy_4w,  axis=0)
    feats["rel_str_spy_13w"] = (close / close.shift(13) - 1).sub(spy_13w, axis=0)

    # ── 11. Short-term reversal (1-week contrarian) ───────────────────
    feats["reversal_1w"] = -(close / close.shift(1) - 1)

    # ── 12-13. Realised volatility (annualised) ───────────────────────
    feats["vol_4w"]  = log_ret.rolling(4).std()  * np.sqrt(52)
    feats["vol_13w"] = log_ret.rolling(13).std() * np.sqrt(52)

    # ── 14. MACD histogram (4w / 9w EMAs - adapted for weekly data) ───
    ema4         = close.ewm(span=4,  adjust=False).mean()
    ema9         = close.ewm(span=9,  adjust=False).mean()
    macd_line    = ema4 - ema9
    macd_signal  = macd_line.ewm(span=4, adjust=False).mean()
    feats["macd_hist"] = macd_line - macd_signal

    # ── 15. Bollinger Band position (20w, 2 std) ─────────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feats["bb_pos"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # ── 16. ATR 4w as % of price ──────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.DataFrame(
        np.maximum(
            np.maximum((high - low).values, (high - prev_close).abs().values),
            (low - prev_close).abs().values,
        ),
        index=close.index, columns=close.columns,
    )
    feats["atr_4w_pct"] = (tr.rolling(4).mean() / close.replace(0, np.nan)) * 100

    # ── 17-18. Moving-average crossovers ─────────────────────────────
    ma4  = close.rolling(4).mean()
    ma13 = close.rolling(13).mean()
    ma26 = close.rolling(26).mean()
    feats["ma_cross_4_13"]  = ma4  / ma13.replace(0, np.nan) - 1
    feats["ma_cross_13_26"] = ma13 / ma26.replace(0, np.nan) - 1

    # ── 19. Distance from 52-week low ─────────────────────────────────
    feats["dist_52w_low"] = close / close.rolling(52).min()

    # ── 20. Rolling beta (52w) ────────────────────────────────────────
    feats["rolling_beta"] = beta_52.shift(1)   # shift so beta uses t-1 data

    # ── 21. Volume trend (4w avg / 13w avg) ──────────────────────────
    feats["vol_trend"] = volume.rolling(4).mean() / volume.rolling(13).mean().replace(0, np.nan)

    # ── 22. Price relative to 26w MA ─────────────────────────────────
    feats["price_to_ma26"] = close / ma26.replace(0, np.nan)

    # ── 23. Return skewness (13w) ─────────────────────────────────────
    feats["skew_13w"] = log_ret.rolling(13).skew()

    # ── 24. Rolling correlation with SPY (13w) ────────────────────────
    # Use the built-in vectorised rolling.corr(Series) - avoids the slow
    # per-window lambda and works correctly whether spy_ret is loaded from
    # cache or a fresh download.
    feats["corr_spy_13w"] = log_ret.rolling(13).corr(spy_ret)

    # ── 25. Upside / downside volatility ratio (13w) ─────────────────
    # Positive returns contribute to upside vol; negative to downside vol
    up_vol   = log_ret.clip(lower=0).rolling(13).std()
    down_vol = log_ret.clip(upper=0).rolling(13).std().replace(0, np.nan)
    feats["up_down_vol"] = up_vol / down_vol

    # ── 26-30. Fundamental features (static cross-sectional signal) ──────
    # Each feature is a wide DataFrame where every row (date) holds the same
    # per-ticker value, since yfinance .info provides only the current figure.
    # Missing tickers (fund_df is None or ticker absent) are median-filled so
    # no rows are lost in the subsequent dropna step.
    if fund_df is not None and not fund_df.empty:
        for col in _FUND_COLS:
            ticker_vals = fund_df[col].reindex(close.columns)
            # Remaining NaN after reindex (tickers not in fund_df) → median fill
            ticker_vals = ticker_vals.fillna(ticker_vals.median())
            feats[col] = pd.DataFrame(
                np.tile(ticker_vals.values, (len(close.index), 1)),
                index=close.index,
                columns=close.columns,
            )

    # ── Target variable (NOT a feature - no lookahead) ────────────────
    # Predict FORWARD_WEEKS-ahead return: smoother signal, less weekly noise
    # A 4-week horizon has better signal-to-noise than 1-week
    fwd_ret = close.shift(-config.FORWARD_WEEKS) / close - 1
    # Cross-sectional rank within each week (0=worst, 1=best cross-sectionally)
    fwd_rank = fwd_ret.rank(axis=1, pct=True)

    # ── Assemble panel (wide → long) ──────────────────────────────────
    panel = pd.concat(feats, axis=1)              # MultiIndex cols: (feat, ticker)
    panel.columns.names = ["feature", "ticker"]
    panel = panel.stack(level="ticker")            # long: (Date, Ticker) × feature
    panel.index.names = ["Date", "Ticker"]

    # Add targets
    fwd_ret_long  = fwd_ret.stack();  fwd_ret_long.index.names  = ["Date", "Ticker"]
    fwd_rank_long = fwd_rank.stack(); fwd_rank_long.index.names = ["Date", "Ticker"]
    panel["fwd_ret"]  = fwd_ret_long
    panel["fwd_rank"] = fwd_rank_long

    # Drop rows missing target (last week has no forward return)
    panel = panel.dropna(subset=["fwd_rank"])

    # Cross-sectional rank normalisation of features (per date, rank [0,1])
    # This removes absolute-level effects so model learns relative ordering
    def cs_rank(df):
        return df.groupby(level="Date")[config.FEATURE_COLS].rank(pct=True)

    panel[config.FEATURE_COLS] = cs_rank(panel)

    # Drop rows where ALL features are NaN (e.g., insufficient warmup)
    panel = panel.dropna(subset=config.FEATURE_COLS, how="all")

    log.info("Feature panel: %d (Date,Ticker) rows, %d features",
             len(panel), len(config.FEATURE_COLS))
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 - WALK-FORWARD TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def _fold_ic(scores: np.ndarray, panel_slice: pd.DataFrame) -> tuple[float, float]:
    """
    Compute mean Spearman IC and IC>0 rate for one fold's predictions.
    Returns (mean_ic, ic_pos_rate).  Both NaN if no date has ≥5 stocks.
    """
    ics = []
    tmp = panel_slice.copy()
    tmp["_score"] = scores
    for _, grp in tmp.groupby(level="Date"):
        valid = grp.dropna(subset=["_score", "fwd_rank"])
        if len(valid) < 5:
            continue
        ic, _ = spearmanr(valid["_score"], valid["fwd_rank"])
        ics.append(ic)
    if not ics:
        return np.nan, np.nan
    return float(np.mean(ics)), float(np.mean([x > 0 for x in ics]))


def walk_forward_train(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Walk-forward cross-validation with quarterly retraining.

    Three models are trained per fold:
      M1 - LightGBM  (gradient-boosted trees, handles non-linearity well)
      M2 - XGBoost   (same hyper-params as LightGBM, different boosting impl)
      M3 - Ridge     (linear baseline; provides regularised OLS on ranked features)

    Ensemble score = equal-weight average of the three raw predictions,
    then cross-sectionally ranked within each date so the final score is
    always on a uniform [0,1] scale regardless of model scale differences.

    IC is tracked separately for each model and the ensemble; a comparison
    table is printed at the end of the walk-forward loop.

    Returns:
      predictions : DataFrame, MultiIndex (Date, Ticker), col 'score'
                    (ensemble score, cross-sectionally ranked)
      ic_records  : list of dicts (one per fold) saved to ic_log.csv
    """
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.IC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_dates        = panel.index.get_level_values("Date")
    panel_dates_norm = pd.DatetimeIndex(raw_dates).tz_localize(None).normalize()

    dates_unique = panel_dates_norm.unique().sort_values()
    start_dt = dates_unique[dates_unique >= pd.Timestamp(config.START_DATE)][0]
    end_dt   = dates_unique[dates_unique <= pd.Timestamp(config.END_DATE)][-1]

    first_cutoff = start_dt + pd.Timedelta(weeks=config.MIN_TRAIN_WEEKS)
    cutoffs = pd.date_range(first_cutoff, end_dt, freq=config.RETRAIN_FREQ)

    if len(cutoffs) == 0:
        log.error("No cutoffs generated - START_DATE/END_DATE range too narrow.")
        return pd.DataFrame(), []

    log.info("Walk-forward: %d quarterly folds from %s to %s",
             len(cutoffs), cutoffs[0].date(), cutoffs[-1].date())
    log.info("Panel date range (normalised): %s → %s", start_dt.date(), end_dt.date())
    log.info("Models: LightGBM | XGBoost | Ridge  →  equal-weight ensemble")

    all_preds  = []
    ic_records = []

    # Track fold-level ICs for each model and the ensemble (for final table)
    fold_ics: dict[str, list[float]] = {
        "lgbm": [], "xgb": [], "ridge": [], "ensemble": []
    }

    # Keep the last trained instances so we can persist them after the loop
    last_lgbm  = None
    last_xgb   = None
    last_ridge = None
    last_scaler = None

    for fold_idx, cutoff in enumerate(cutoffs, 1):
        cutoff_norm     = pd.Timestamp(cutoff).tz_localize(None).normalize()
        next_cutoff_raw = (cutoffs[fold_idx] if fold_idx < len(cutoffs)
                           else end_dt + pd.Timedelta(weeks=13))
        next_cutoff     = pd.Timestamp(next_cutoff_raw).tz_localize(None).normalize()

        train_mask = panel_dates_norm < cutoff_norm
        train = panel.iloc[train_mask].dropna(
            subset=config.FEATURE_COLS + ["fwd_rank"]
        )

        if len(train) < 500:
            log.warning("Fold %2d: only %d training rows (need 500), skipping",
                        fold_idx, len(train))
            continue

        X_tr = train[config.FEATURE_COLS].values
        y_tr = train["fwd_rank"].values

        # ── Model 1: LightGBM ────────────────────────────────────────
        m_lgbm = lgb.LGBMRegressor(**config.LGBM_PARAMS)
        m_lgbm.fit(X_tr, y_tr, feature_name=config.FEATURE_COLS)

        # ── Model 2: XGBoost (same tree structure hyper-params) ──────
        m_xgb = xgb.XGBRegressor(
            n_estimators      = config.LGBM_PARAMS["n_estimators"],
            learning_rate     = config.LGBM_PARAMS["learning_rate"],
            max_depth         = 4,          # XGBoost uses depth, not num_leaves
            subsample         = config.LGBM_PARAMS["subsample"],
            colsample_bytree  = config.LGBM_PARAMS["colsample_bytree"],
            reg_alpha         = config.LGBM_PARAMS["reg_alpha"],
            reg_lambda        = config.LGBM_PARAMS["reg_lambda"],
            random_state      = config.LGBM_PARAMS["random_state"],
            n_jobs            = -1,
            verbosity         = 0,
            tree_method       = "hist",     # fast histogram method
        )
        m_xgb.fit(X_tr, y_tr)

        # ── Model 3: Ridge regression (z-score features first) ───────
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        m_ridge = Ridge(alpha=1.0, fit_intercept=True)
        m_ridge.fit(X_tr_sc, y_tr)

        # ── OOS prediction window ─────────────────────────────────────
        pred_mask  = ((panel_dates_norm >= cutoff_norm) &
                      (panel_dates_norm <  next_cutoff))
        pred_panel = panel.iloc[pred_mask].dropna(subset=config.FEATURE_COLS)

        if pred_panel.empty:
            continue

        X_pr = pred_panel[config.FEATURE_COLS].values

        s_lgbm  = m_lgbm.predict(X_pr)
        s_xgb   = m_xgb.predict(X_pr)
        s_ridge = m_ridge.predict(scaler.transform(X_pr))

        # Equal-weight average, then cross-sectional rank within each date
        s_avg = (s_lgbm + s_xgb + s_ridge) / 3.0
        pred_panel = pred_panel.copy()
        pred_panel["s_lgbm"]  = s_lgbm
        pred_panel["s_xgb"]   = s_xgb
        pred_panel["s_ridge"] = s_ridge
        pred_panel["s_avg"]   = s_avg

        # Cross-sectional rank of the ensemble score (per date, pct=True → [0,1])
        cs_ranked = (pred_panel.groupby(level="Date")["s_avg"]
                               .rank(pct=True))
        pred_panel["score"] = cs_ranked

        all_preds.append(pred_panel[["score", "fwd_rank", "fwd_ret"]])

        # ── Per-model IC for this fold ────────────────────────────────
        ic_lgbm,  pos_lgbm  = _fold_ic(s_lgbm,  pred_panel)
        ic_xgb,   pos_xgb   = _fold_ic(s_xgb,   pred_panel)
        ic_ridge, pos_ridge  = _fold_ic(s_ridge, pred_panel)
        ic_ens,   pos_ens    = _fold_ic(pred_panel["score"].values, pred_panel)

        for key, val in [("lgbm", ic_lgbm), ("xgb", ic_xgb),
                         ("ridge", ic_ridge), ("ensemble", ic_ens)]:
            if not np.isnan(val):
                fold_ics[key].append(val)

        ic_records.append({
            "fold":       fold_idx,
            "train_end":  cutoff_norm.date(),
            "pred_start": cutoff_norm.date(),
            "pred_end":   min(next_cutoff, end_dt).date(),
            "n_dates":    int(pred_panel.index.get_level_values("Date").nunique()),
            "ic_lgbm":    round(ic_lgbm,  5) if not np.isnan(ic_lgbm)  else None,
            "ic_xgb":     round(ic_xgb,   5) if not np.isnan(ic_xgb)   else None,
            "ic_ridge":   round(ic_ridge,  5) if not np.isnan(ic_ridge) else None,
            "mean_ic":    round(ic_ens,    5) if not np.isnan(ic_ens)   else None,
            "ic_pos_rate": round(pos_ens,  3) if not np.isnan(pos_ens)  else None,
        })

        log.info(
            "Fold %2d/%d | n=%5d | "
            "LGB=%+.4f  XGB=%+.4f  RDG=%+.4f  ENS=%+.4f | IC>0=%4.1f%%",
            fold_idx, len(cutoffs), len(train),
            ic_lgbm, ic_xgb, ic_ridge, ic_ens,
            pos_ens * 100 if not np.isnan(pos_ens) else 0,
        )

        last_lgbm, last_xgb, last_ridge, last_scaler = (
            m_lgbm, m_xgb, m_ridge, scaler
        )

    # ── Guard: no folds completed ─────────────────────────────────────
    ic_df = pd.DataFrame(ic_records)
    if ic_df.empty:
        log.error("No folds completed - all folds had fewer than 500 training rows.")
        return pd.DataFrame(), []

    ic_df.to_csv(config.IC_LOG_PATH, index=False)

    # ── Model comparison table ────────────────────────────────────────
    sep = "═" * 60
    log.info(sep)
    log.info("Walk-forward complete  (%d folds ran)", len(ic_df))
    log.info("")
    log.info("  Per-model IC comparison (mean OOS Spearman IC):")
    log.info("  %-12s  %8s  %8s  %9s", "Model", "Mean IC", "Std IC", "IC>0 rate")
    log.info("  %s", "─" * 44)
    for key, label in [("lgbm", "LightGBM"), ("xgb", "XGBoost"),
                        ("ridge", "Ridge"), ("ensemble", "Ensemble ★")]:
        vals = fold_ics[key]
        if vals:
            log.info("  %-12s  %+.5f  %8.5f  %8.1f%%",
                     label, np.mean(vals), np.std(vals),
                     np.mean([v > 0 for v in vals]) * 100)
        else:
            log.info("  %-12s  %8s", label, "n/a")
    log.info("  %s", "─" * 44)
    ens_mean = np.mean(fold_ics["ensemble"]) if fold_ics["ensemble"] else np.nan
    log.info("  Ensemble IC > 0.03 : %s",
             "YES ✓" if ens_mean > 0.03 else "NO  ✗")
    log.info("  IC log → %s", config.IC_LOG_PATH)
    log.info(sep)

    # ── Persist the final-fold model stack ───────────────────────────
    if last_lgbm is not None:
        with open(config.MODEL_PATH, "wb") as fh:
            pickle.dump({
                "lgbm":        last_lgbm,
                "xgb":         last_xgb,
                "ridge":       last_ridge,
                "scaler":      last_scaler,
                "feature_cols": config.FEATURE_COLS,
            }, fh)
        log.info("Model stack saved → %s  (lgbm + xgb + ridge)", config.MODEL_PATH)

    predictions = pd.concat(all_preds) if all_preds else pd.DataFrame()
    return predictions, ic_records


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 - BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def _sector_neutralize(w: pd.Series, sector_map: dict) -> pd.Series:
    """
    Make each GICS sector dollar-neutral by subtracting the sector's net weight
    equally from every stock in that sector.  Re-scales gross exposure to match
    the pre-adjustment total so position sizing is unchanged in aggregate.
    """
    if not sector_map or len(w) == 0:
        return w

    orig_abs = w.abs().sum()
    w_adj    = w.copy()
    sectors  = pd.Series({t: sector_map.get(t, "Unknown") for t in w.index})

    for sector in sectors.unique():
        if sector == "Unknown":
            continue
        mask  = sectors == sector
        ticks = mask[mask].index
        sw    = w_adj.reindex(ticks)
        if len(sw) < 2:
            continue
        net = float(sw.sum())
        if abs(net) < 1e-10:
            continue
        w_adj[ticks] -= net / len(ticks)

    new_abs = w_adj.abs().sum()
    if new_abs > 1e-9 and orig_abs > 1e-9:
        w_adj *= orig_abs / new_abs
    return w_adj


def _apply_weight_cap(w: pd.Series, max_w: float) -> pd.Series:
    """Hard clip every individual weight to ±max_w."""
    if len(w) == 0 or max_w <= 0:
        return w
    return w.clip(-max_w, max_w)


def _build_weights(scores: pd.Series, n: int, in_regime: bool) -> pd.Series:
    """
    Soft continuous weights: score rank → weight, dollar-neutral.
    Avoids the binary top-N/bottom-N churn that drives 96% weekly turnover.

    Method: convert scores to percentile ranks, demean, then apply a power
    transform (alpha) to concentrate weight on extreme ranks while keeping
    moderate ranks in the portfolio. This dramatically reduces turnover
    because weight changes are gradual rather than stocks entering/exiting
    a hard top-N cutoff.

    Go FLAT (zero weights) when SPY is below its 40-week MA - regime filter
    applies to BOTH legs to avoid a net-short bias during bear-market rallies.
    """
    clean = scores.dropna()
    if len(clean) < 10:
        return pd.Series(dtype=float)

    # Regime filter: go completely flat in bear markets
    if not in_regime:
        return pd.Series(0.0, index=clean.index)

    # Convert to percentile ranks [0, 1]
    pct_rank = clean.rank(pct=True)

    # Demean around 0.5 → range [-0.5, +0.5]
    centered = pct_rank - 0.5

    # Power transform: amplify extremes, shrink middle
    # sign(x) * |x|^alpha keeps the sign while concentrating weight
    alpha = config.SOFT_WEIGHT_ALPHA
    w_raw = np.sign(centered) * (np.abs(centered) ** alpha)

    # Normalize to dollar-neutral: long sum = 1, short sum = -1 roughly
    # Divide by the absolute sum so gross exposure ≈ 1
    abs_sum = w_raw.abs().sum()
    if abs_sum < 1e-9:
        return pd.Series(0.0, index=clean.index)

    w = w_raw / abs_sum
    return w


def run_backtest(
    predictions: pd.DataFrame,
    data:        dict,
) -> pd.DataFrame:
    """
    Weekly backtest: long top-decile / short bottom-decile each week.
    P&L timing: earn last week's P&L, THEN rebalance to new weights.

    Regime filter  : no long positions when SPY < 40-week MA.
    Vol targeting  : scale weights so trailing 52w portfolio vol ≈ TARGET_VOL.
    Transaction costs: TC_BPS per side.

    Returns backtest DataFrame with weekly P&L, portfolio value, metadata.
    """
    close    = data["close"]
    spy      = data["spy_close"]

    # Weekly close-to-close returns for each stock
    stock_wret = close.pct_change()

    # SPY regime filter: SPY > 40-week MA
    spy_ma40   = spy.rolling(config.SPY_MA_WEEKS).mean()
    in_regime  = spy > spy_ma40   # True = bull market

    # All rebalancing dates (weeks with OOS predictions)
    pred_dates = predictions.index.get_level_values("Date").unique().sort_values()

    TC        = config.TC_BPS / 10_000
    rows      = []
    prev_w    = pd.Series(dtype=float)
    port_val  = float(config.CASH)
    port_rets_history = []  # for rolling vol targeting
    smoothed_scores   = {}  # EMA-smoothed scores per ticker (reduces noise-churn)
    weeks_since_rebal = 0
    stop_loss_set     = set()   # tickers to exit on next weight construction
    weekly_stats      = []      # for quarterly reporting

    for date in pred_dates:
        # ── Fetch this week's score from predictions ──────────────────
        if date not in predictions.index.get_level_values("Date"):
            continue
        raw_scores = predictions.xs(date, level="Date")["score"]

        # ── EMA score smoothing (α=0.4): stabilises weekly score fluctuations ──
        EMA_ALPHA = 0.4
        for ticker, sc in raw_scores.items():
            if ticker in smoothed_scores:
                smoothed_scores[ticker] = EMA_ALPHA * sc + (1 - EMA_ALPHA) * smoothed_scores[ticker]
            else:
                smoothed_scores[ticker] = sc
        scores = pd.Series(smoothed_scores).reindex(raw_scores.index)

        # ── Step 1: Earn P&L with PREVIOUS weights; detect stop-losses ──────
        week_pnl      = 0.0
        new_stop_loss = set()
        if len(prev_w) > 0 and date in stock_wret.index:
            rets_today = stock_wret.loc[date]
            valid      = prev_w.index.intersection(rets_today.dropna().index)
            week_pnl   = float((prev_w[valid] * rets_today[valid]).sum())

            # Position loss = -(sign_of_weight) × stock_return
            # Long +8% drop → loss 8%; Short +8% rise → loss 8%
            for ticker in valid:
                pos_loss = -float(np.sign(prev_w[ticker])) * float(rets_today[ticker])
                if pos_loss > config.STOP_LOSS_PCT:
                    new_stop_loss.add(ticker)

        stop_loss_set = new_stop_loss
        port_rets_history.append(week_pnl)
        port_val = port_val * (1.0 + week_pnl)

        # ── Step 2: Rebalance every REBALANCE_WEEKS weeks only ──────────────
        weeks_since_rebal += 1
        bull_market   = bool(in_regime.get(date, True))
        rebalance_now = (weeks_since_rebal >= config.REBALANCE_WEEKS) or (len(prev_w) == 0)

        if rebalance_now:
            weeks_since_rebal = 0
            new_w = _build_weights(scores, config.DECILE_N, in_regime=bull_market)
            # Sector neutralisation: make each GICS sector dollar-neutral
            if len(new_w) > 0:
                new_w = _sector_neutralize(new_w, config.SECTOR_MAP)
        else:
            new_w = prev_w.copy()   # hold; copy so stop-loss edits don't corrupt prev_w

        # ── Stop-loss exits: zero out triggered positions (any week) ─────────
        if stop_loss_set and len(new_w) > 0:
            for ticker in stop_loss_set:
                if ticker in new_w.index:
                    new_w[ticker] = 0.0

        # ── Volatility targeting ─────────────────────────────────────────────
        vol_scale = 1.0
        if len(port_rets_history) >= config.LOOKBACK_WEEKS:
            window     = pd.Series(port_rets_history[-config.LOOKBACK_WEEKS:])
            realised_v = window.std() * np.sqrt(52)
            if realised_v > 1e-6:
                vol_scale = min(config.TARGET_VOL / realised_v, config.MAX_LEVERAGE)
                new_w     = new_w * vol_scale

        # ── Hard per-stock weight cap ─────────────────────────────────────────
        if len(new_w) > 0:
            new_w = _apply_weight_cap(new_w, config.MAX_STOCK_WEIGHT)

        # ── Transaction costs: proportional to turnover ──────────────────────
        if len(prev_w) > 0 and len(new_w) > 0:
            old_aligned = prev_w.reindex(new_w.index, fill_value=0.0)
            turnover    = (new_w - old_aligned).abs().sum() / 2.0
        else:
            turnover = new_w.abs().sum() / 2.0 if len(new_w) > 0 else 0.0

        tc_cost  = turnover * TC
        port_val = port_val * (1.0 - tc_cost)

        # ── Sector exposure metric (max |net sector weight| across sectors) ───
        sec_exp = 0.0
        if len(new_w) > 0:
            sec_net: dict = {}
            for t, wt in new_w.items():
                s = config.SECTOR_MAP.get(t, "Unknown")
                sec_net[s] = sec_net.get(s, 0.0) + wt
            sec_net.pop("Unknown", None)
            sec_exp = max(abs(v) for v in sec_net.values()) if sec_net else 0.0

        n_pos = int((new_w.abs() > 1e-6).sum()) if len(new_w) > 0 else 0

        rows.append({
            "Date":            date,
            "Strategy_Return": week_pnl - tc_cost,
            "Portfolio_Value": port_val,
            "In_Regime":       int(bull_market),
            "Turnover":        round(turnover, 4),
            "Vol_Scale":       round(vol_scale, 4),
            "TC_Cost":         round(tc_cost, 6),
        })
        weekly_stats.append({
            "Date":         date,
            "Turnover":     turnover,
            "N_positions":  n_pos,
            "Stop_losses":  len(stop_loss_set),
            "Max_sec_exp":  sec_exp,
        })

        prev_w = new_w

    # ── Quarterly stats report ────────────────────────────────────────────────
    _print_quarterly_stats(weekly_stats)

    # Add SPY buy-and-hold for comparison
    backtest_df = pd.DataFrame(rows).set_index("Date")
    spy_aligned = spy.reindex(backtest_df.index, method="ffill")
    spy_cum     = spy_aligned / spy_aligned.iloc[0] * config.CASH
    backtest_df["SPY_Value"] = spy_cum.values
    backtest_df["SPY_Return"] = spy_aligned.pct_change().reindex(backtest_df.index).fillna(0).values

    return backtest_df


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 - MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def _print_quarterly_stats(weekly_stats: list):
    """Summarise backtest stats by calendar quarter and print a table."""
    if not weekly_stats:
        return
    df = pd.DataFrame(weekly_stats)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    quarterly = df.groupby(pd.Grouper(freq="QS")).agg(
        Avg_Turnover =("Turnover",    "mean"),
        Avg_Positions=("N_positions", "mean"),
        Stop_losses  =("Stop_losses", "sum"),
        Max_sec_exp  =("Max_sec_exp", "mean"),
    )

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  QUARTERLY BACKTEST STATS")
    print(f"  {'Quarter':<12} {'Avg Turnover':>13} {'Avg Positions':>14} "
          f"{'Stop-losses':>12} {'Avg Max SecExp':>15}")
    print(f"  {sep}")
    for qdate, row in quarterly.iterrows():
        print(f"  {str(qdate.date()):<12} "
              f"{row['Avg_Turnover']:>13.1%} "
              f"{row['Avg_Positions']:>14.0f} "
              f"{row['Stop_losses']:>12.0f} "
              f"{row['Max_sec_exp']:>15.3f}")
    print(f"{sep}\n")


def _print_backtest_summary(bt: pd.DataFrame):
    """Print key performance metrics to console."""
    rets   = bt["Strategy_Return"]
    spy_r  = bt["SPY_Return"]
    total  = bt["Portfolio_Value"].iloc[-1] / config.CASH - 1
    n_wks  = len(rets)
    ann_r  = (1 + total) ** (52 / n_wks) - 1
    ann_v  = rets.std() * np.sqrt(52)
    sharpe = (ann_r - 0.05) / ann_v if ann_v > 0 else 0
    mdd    = ((bt["Portfolio_Value"] - bt["Portfolio_Value"].cummax())
               / bt["Portfolio_Value"].cummax()).min()
    spy_tot= bt["SPY_Value"].iloc[-1] / config.CASH - 1
    wr     = (rets > 0).mean()

    # Alpha / Beta
    cov_  = rets.cov(spy_r)
    var_  = spy_r.var()
    beta_ = cov_ / var_ if var_ > 0 else 0
    alpha_= (ann_r - 0.05) - beta_ * ((1 + spy_r.mean() * 52) - 1 - 0.05)

    sep = "═" * 55
    print(f"\n{sep}")
    print(f"  QUANTPROJECT v2 - BACKTEST RESULTS")
    print(f"  Period: {config.START_DATE} → {config.END_DATE}")
    print(f"{sep}")
    print(f"  Total Return (Strategy) : {total:>+.1%}")
    print(f"  Total Return (SPY B&H)  : {spy_tot:>+.1%}")
    print(f"  Annualised Return       : {ann_r:>+.1%}")
    print(f"  Annualised Volatility   : {ann_v:>.1%}")
    print(f"  Sharpe Ratio            : {sharpe:>.3f}")
    print(f"  Max Drawdown            : {mdd:>.1%}")
    print(f"  Weekly Win Rate         : {wr:>.1%}")
    print(f"  Alpha (annualised)      : {alpha_:>+.1%}")
    print(f"  Beta vs SPY             : {beta_:>.3f}")
    print(f"{sep}\n")


def main():
    t0 = time.time()
    config.DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info("  QUANTPROJECT v2 - STARTING PIPELINE")
    log.info("═" * 60)

    # 1. Download
    log.info("STAGE 1/4 - Data Download")
    data = download_data()

    # 2. Features
    log.info("STAGE 2/4 - Feature Engineering (30 factors: 25 technical + 5 fundamental)")
    fund_df = fetch_fundamentals(config.UNIVERSE)
    panel   = compute_features(data, fund_df)

    # 3. Walk-forward training
    log.info("STAGE 3/4 - Walk-Forward LightGBM Training")
    predictions, ic_records = walk_forward_train(panel)

    if predictions.empty:
        log.error("No predictions generated. Check data range.")
        sys.exit(1)

    # Save predictions for paper_trade.py
    config.PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(config.PREDICTIONS_PATH)
    log.info("Predictions saved → %s", config.PREDICTIONS_PATH)

    # 4. Backtest
    log.info("STAGE 4/4 - Backtest Simulation")
    backtest = run_backtest(predictions, data)
    backtest.to_csv(config.BACKTEST_PATH)
    log.info("Backtest saved → %s", config.BACKTEST_PATH)

    _print_backtest_summary(backtest)
    log.info("Pipeline complete in %.0f seconds.", time.time() - t0)
    log.info("Next: run  python analytics.py   (dashboard)")
    log.info("      run  python signals.py      (current signals)")
    log.info("      run  python paper_trade.py  (paper simulation)")


if __name__ == "__main__":
    main()
