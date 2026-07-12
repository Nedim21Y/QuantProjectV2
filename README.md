# QuantProject v2

A systematic, walk-forward validated equity trading system that ranks S&P 500 stocks weekly using a three-model ensemble (LightGBM + XGBoost + Ridge) trained on 30 cross-sectional features, then constructs a dollar-neutral long/short portfolio with regime filtering and volatility targeting. Every backtest prediction is made strictly out-of-sample - data the model has never seen during training.

---

## Strategy Overview

**Universe:** 459 S&P 500 constituents · weekly OHLCV 2017-2025 via yfinance

### Signal generation
- **25 technical features:** price momentum (4w / 13w / 26w / 52w), RSI, idiosyncratic volatility, Bollinger Band position, MACD histogram, ATR, rolling beta, up/down volatility ratio, relative strength vs SPY, MA crossovers, return skewness, and more
- **5 fundamental features:** PE ratio, price-to-book, profit margin, revenue growth YoY, debt-to-equity (yfinance `.info`)
- All features **cross-sectionally rank-normalised** within each week to remove level effects
- **Three models** (LightGBM · XGBoost · Ridge) trained per fold; final score = equal-weight average, re-ranked cross-sectionally

### Portfolio construction

| Control | Setting | Purpose |
|---------|---------|---------|
| **Soft weights** | `sign(rank-0.5) × |rank-0.5|³` | Continuous weights across all stocks; reduces turnover from 96% → 29% vs hard top-N |
| **Dollar-neutral** | long sum = short sum | Market-neutral by construction |
| **Sector neutrality** | each GICS sector net = 0 | Prevents sector tilts from masquerading as stock-picking alpha |
| **Regime filter** | SPY < 40-week MA → flat | Both legs go to zero in bear markets; avoids net-short during rallies |
| **Volatility targeting** | 12% annualised (cap 1.5×) | Consistent risk exposure across market regimes |
| **Stop-loss** | -8% single-week position loss | Exit next week regardless of model signal |
| **Position cap** | ±5% per stock | Hard weight ceiling after vol scaling |
| **Rebalancing** | every 4 weeks | Monthly cadence cuts TC ~4× vs weekly |
| **Transaction costs** | 5 bps per side | Conservative estimate for liquid S&P 500 names |

### Walk-forward validation
- 27 quarterly folds covering April 2018 → December 2024 (349 weeks)
- Each fold: train on all history up to fold date, predict the following quarter out-of-sample
- No data leakage: features use only past prices; target returns not seen during training

---

## Results

Backtest period: **6 April 2018 → 6 December 2024** · 349 weeks · 27 OOS folds

### Performance vs SPY

| Metric                | Strategy  | SPY Buy & Hold |
|-----------------------|-----------|----------------|
| Total Return          | +69.3%    | +160.2%        |
| Annualised Return     | +8.2%     | +15.3%         |
| Annualised Volatility | 10.7%     | 18.1%          |
| Sharpe Ratio          | 0.30      | 0.55           |
| Max Drawdown          | **-21.7%**| -32.2%         |
| Alpha (ann.)          | **+2.9%** | -              |
| Beta vs SPY           | 0.03      | 1.00           |
| Weekly Win Rate       | 44.4%     | 59.3%          |
| Total TC Drag         | 4.98%     | -              |

### Model IC (mean OOS Spearman, 27 folds)

| Model        | Mean IC  | IC > 0  |
|--------------|----------|---------|
| LightGBM     | 0.0491   | 88.9%   |
| XGBoost      | 0.0498   | 92.6%   |
| Ridge        | 0.0387   | 77.8%   |
| **Ensemble** | **0.0506**| **85.2%** |

> **Survivorship bias note:** yfinance returns only current S&P 500 members; delisted stocks are absent, which inflates backtest returns relative to live trading.

Full results with charts: see [RESULTS.md](RESULTS.md)

Complete mathematical specification of every formula in the system: see [MATHEMATICS.md](MATHEMATICS.md)

---

## How to Run

**Prerequisites:** Python 3.11+, ~4 GB RAM, internet connection

```
1.  Clone and enter the project directory

        git clone <repo-url>
        cd QuantProjectV2

2.  Create and activate a virtual environment

        python -m venv venv
        source venv/bin/activate          # macOS / Linux
        venv\Scripts\activate             # Windows

3.  Install dependencies

        pip install -r requirements.txt

4.  Run the full pipeline  (~20-40 min on first run - downloads 460 tickers)

        python main.py

    Outputs: data/prices.parquet, models/lgbm_latest.pkl,
             results/backtest.csv, results/ic_log.csv

5.  Generate the analytics dashboard

        python analytics.py
        # → results/plots/dashboard.png

6.  View current BUY / SELL signals

        python signals.py

7.  Run the paper trading simulation ($10k from 2025-01-01)

        python paper_trade.py

8.  Launch the interactive Streamlit dashboard

        streamlit run app.py
        # Opens at http://localhost:8501
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      QUANTPROJECT v2 PIPELINE                      │
└────────────────────────────────────────────────────────────────────┘

  yfinance API
  459 S&P 500 tickers
  Weekly OHLCV 2017-2025
         │
         ▼
  ┌─────────────────────┐
  │    DATA DOWNLOAD     │   download_data()  ·  main.py
  │  5 batches × ~90    │   → data/prices.parquet  (Parquet cache)
  │  W-FRI resample     │   → data/fundamentals.json  (7-day cache)
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  FEATURE ENGINEERING │   compute_features()  ·  main.py
  │                      │
  │  25 technical        │   All features cross-sectionally
  │  +  5 fundamental    │   rank-normalised within each week
  │  = 30 total          │
  └──────────┬──────────┘
             │
             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │              WALK-FORWARD CROSS-VALIDATION                   │
  │              27 quarterly folds  ·  2018-2024                │
  │                                                              │
  │   ◀── train on all history ──▶│◀── predict (13 weeks) ──▶   │
  │   [fold 1]  [fold 2]  · · ·  [fold 27]                      │
  │                                                              │
  │   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
  │   │   LightGBM    │  │    XGBoost    │  │     Ridge     │   │
  │   │  300 trees    │  │  300 trees    │  │   L2 = 1.0    │   │
  │   │  num_leaves=15│  │  max_depth=4  │  │  z-scored X   │   │
  │   └──────┬────────┘  └──────┬────────┘  └──────┬────────┘   │
  │          └──────────────────┴──────────────────┘            │
  │                             │  equal-weight average         │
  │                             ▼                               │
  │              cross-sectional rank  →  score [0,1]           │
  └─────────────────────────────┬────────────────────────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              │                                    │
              ▼                                    ▼
  ┌───────────────────────┐          ┌───────────────────────────┐
  │        BACKTEST        │          │       LIVE SIGNALS         │
  │   run_backtest()       │          │                           │
  │                        │          │  signals.py               │
  │  · Soft continuous     │          │    → top 20 BUY / SELL    │
  │    weights (power 3)   │          │                           │
  │  · Sector neutrality   │          │  paper_trade.py           │
  │  · Regime filter       │          │    → $10k simulation      │
  │  · Vol targeting 12%   │          │                           │
  │  · Stop-loss 8%        │          │  app.py  (Streamlit)      │
  │  · Weight cap 5%       │          │    → interactive dashboard │
  │  · Monthly rebalance   │          └───────────────────────────┘
  │  · 5 bps TC per side   │
  └───────────┬────────────┘
              │
              ▼
  ┌───────────────────────┐
  │       ANALYTICS        │   analytics.py
  │   9-panel dashboard    │   → results/plots/dashboard.png
  │   IC · Sharpe · Alpha  │   → results/backtest.csv
  │   drawdown · heatmap   │   → results/ic_log.csv
  └───────────────────────┘
```

---

## Project Structure

```
QuantProjectV2/
├── main.py          # Full pipeline: download → features → train → backtest
├── config.py        # Single source of truth: universe (459), sector map, params
├── signals.py       # Live BUY/SELL signal generation from latest weekly bar
├── paper_trade.py   # $10k paper portfolio simulation from 2025-01-01
├── analytics.py     # 9-panel dark-theme performance dashboard
├── app.py           # Streamlit dashboard (Live Signals / Backtest / IC / Models)
├── requirements.txt
├── README.md
├── RESULTS.md
├── data/            # (gitignored)  prices.parquet, fundamentals.json
├── models/          # (gitignored)  lgbm_latest.pkl  (3-model stack)
└── results/         # (gitignored)  backtest.csv, ic_log.csv, plots/
```

---

## Requirements

**Python:** 3.11 or 3.12 (developed and tested on 3.12)

| Package | Version | Role |
|---------|---------|------|
| `yfinance` | 0.2.43 | Market data download |
| `lightgbm` | 4.3.0 | Gradient boosting model (M1) |
| `xgboost` | 3.3.0 | Gradient boosting model (M2) |
| `scikit-learn` | 1.4.2 | Ridge regression (M3), StandardScaler |
| `pandas` | 2.2.2 | Data manipulation |
| `numpy` | 1.26.4 | Numerical computing |
| `scipy` | 1.13.0 | Spearman IC calculation |
| `matplotlib` | 3.8.4 | Analytics plots |
| `seaborn` | 0.13.2 | Heatmap styling |
| `streamlit` | ≥1.35.0 | Interactive web dashboard |
| `pyarrow` | 15.0.2 | Parquet cache I/O |
| `joblib` | 1.4.0 | Model serialisation |

Install all: `pip install -r requirements.txt`

---

## Key Design Decisions

| Decision | Alternative considered | Why this approach |
|----------|----------------------|-------------------|
| 4-week return target | 1-week | Smoother signal; OOS IC improved 0.015 → 0.051 |
| Soft continuous weights | Binary top-N | Hard cutoff caused 96% weekly turnover → 34% TC drag |
| Monthly rebalancing | Weekly | Cuts total TC drag from 34% to 5% |
| Sector neutralisation | No sector control | Removes hidden sector-factor exposure from alpha |
| Regime filter (both legs) | Long leg only | Flat avoids net-short bias during bear-market rallies |
| Walk-forward CV (27 folds) | Random / k-fold | Respects time ordering; eliminates lookahead bias |
| 3-model ensemble | Single model | Variance reduction; ensemble IC > any individual model |
