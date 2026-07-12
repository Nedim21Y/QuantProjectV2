"""
QuantProject v2 - Central Configuration
Single source of truth for all parameters.
"""

# ── Universe: full S&P 500 constituent list (503 tickers as of June 2025)
# Source: https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
# Survivorship bias note: yfinance only returns data for currently listed
# tickers; delisted stocks are absent, which inflates backtest returns
# versus live trading. Acknowledged limitation.
# Tickers with insufficient history (< 100 weekly bars since START_DATE)
# are dropped automatically during download - see download_data() in main.py.
UNIVERSE = [
    # Communication Services
    "CHTR", "CMCSA", "DIS", "EA", "FOX", "FOXA", "GOOGL", "GOOG",
    "LYV", "META", "NFLX", "NWS", "NWSA", "OMC", "PARA", "T",
    "TMUS", "TTWO", "VZ", "WBD",
    # Consumer Discretionary
    "AMZN", "AN", "APTV", "AZO", "BBY", "BKNG", "BWA", "CCL",
    "CMG", "DPZ", "DRI", "EBAY", "EXPE", "F", "GM", "HLT",
    "HD", "LEN", "LVS", "LOW", "MAR", "MCD", "MGM", "MHK",
    "NKE", "NVR", "NCLH", "ORLY", "PHM", "RCL", "RL", "ROST",
    "SBUX", "TGT", "TJX", "TPR", "TSLA", "ULTA", "WYNN", "YUM",
    "CVNA", "DASH", "DECK", "LULU",
    # Consumer Staples
    "ADM", "CAG", "CHD", "CL", "CLX", "COST", "GIS", "HRL",
    "HSY", "K", "KHC", "KMB", "KO", "KR", "MDLZ", "MKC",
    "MO", "MNST", "PEP", "PG", "PM", "SJM", "STZ", "SYY",
    "TAP", "TSN", "WMT", "BG", "KVUE",
    # Energy
    "APA", "BKR", "COP", "CVX", "DVN", "EOG", "EQT", "FANG",
    "FCX", "HAL", "HES", "KMI", "MPC", "MRO", "OKE", "OXY",
    "PSX", "SLB", "TRGP", "VLO", "XOM",
    # Financials
    "AFL", "AIG", "AIZ", "AJG", "ALL", "AMP", "AON", "AXP",
    "BAC", "BEN", "BK", "BLK", "BRK-B", "BRO", "C", "CB",
    "CBOE", "CFG", "CINF", "CME", "COF", "CNA", "DFS", "EG",
    "ERIE", "FIS", "FISV", "FITB", "GL", "GPN", "GS", "HBAN",
    "ICE", "IVZ", "JPM", "KEY", "KKR", "L", "LNC", "MA",
    "MCO", "MET", "MS", "MSCI", "MTB", "NDAQ", "NTRS", "PFG",
    "PGR", "PNC", "PRU", "RJF", "RF", "SCHW", "SPGI", "STT",
    "SYF", "TFC", "TROW", "TRV", "USB", "V", "WFC", "WRB",
    "APO", "ARES", "BNY", "BX", "CPAY", "HOOD", "IBKR", "KDP",
    # Health Care
    "A", "ABBV", "ABT", "ALGN", "AMGN", "BAX", "BDX", "BIIB",
    "BMY", "BSX", "CAH", "CNC", "COO", "CRL", "CVS", "DXCM",
    "DHR", "DVA", "ELV", "EW", "GEHC", "GILD", "HCA", "HUM",
    "HSIC", "IDXX", "INCY", "IQV", "ISRG", "JNJ", "LH", "LLY",
    "MCK", "MDT", "MRNA", "MRK", "MTD", "PFE", "PODD", "REGN",
    "RMD", "RVTY", "STE", "SYK", "TMO", "UNH", "VRTX", "WAT",
    "ZBH", "ZTS", "SOLV",
    # Industrials
    "ALLE", "AME", "AOS", "AXON", "BA", "BLDR", "CAT", "CHRW",
    "CTAS", "DAL", "DE", "DOV", "EME", "EMR", "EXPD", "FAST",
    "FDX", "FTV", "GD", "GE", "GEV", "GWW", "HON", "HUBB",
    "HII", "HWM", "IR", "ITW", "J", "JBHT", "JCI", "LDOS",
    "LHX", "LII", "LMT", "LUV", "MAS", "MMM", "NOC", "NSC",
    "ODFL", "OTIS", "PCAR", "PH", "PNR", "PWR", "ROK", "ROL",
    "ROP", "RSG", "RTX", "SNA", "SWK", "TDG", "TDY", "TT",
    "TXT", "UAL", "UBER", "UNP", "UPS", "URI", "WAB", "WM",
    "WMB", "XYL", "AXON", "CARR", "GNRC",
    # Information Technology
    "AAPL", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AMAT", "AMD",
    "ANET", "APH", "APP", "AVGO", "BR", "CDNS", "CDW", "CRM",
    "CRWD", "CSCO", "CTSH", "DDOG", "DELL", "FFIV", "FICO", "FLEX",
    "FTNT", "GLW", "HPE", "HPQ", "IBM", "ICE", "INTC", "INTU",
    "IT", "JKHY", "KEYS", "KLAC", "LRCX", "MCHP", "MPWR", "MRVL",
    "MSFT", "MSI", "MU", "NXPI", "NOW", "NTAP", "ON", "ORCL",
    "PANW", "PAYX", "PLTR", "PTC", "PYPL", "QCOM", "SMCI", "SNPS",
    "STX", "SWKS", "TEL", "TER", "TXN", "TYL", "VEEV", "VRSN",
    "VRT", "WDC", "WDAY", "ZBRA", "TTD", "COIN",
    # Materials
    "ALB", "AMCR", "APD", "AVY", "BALL", "CF", "DD", "DOW",
    "ECL", "EMN", "FMC", "IFF", "IP", "LIN", "LYB", "MLM",
    "MOS", "NEM", "NUE", "PKG", "PPG", "STLD", "SW", "VMC",
    "WRK",
    # Real Estate
    "AMT", "ARE", "AVB", "BXP", "CBRE", "CCI", "CPT", "DLR",
    "DOC", "EQIX", "EQR", "ESS", "EXR", "FRT", "HST", "INVH",
    "IRM", "KIM", "MAA", "O", "PLD", "PSA", "REG", "SBAC",
    "SPG", "UDR", "VICI", "VTR", "WELL", "WY",
    # Utilities
    "AEE", "AEP", "AES", "ATO", "AWK", "CMS", "CNP", "D",
    "DTE", "DUK", "ED", "EIX", "ES", "ETR", "EVRG", "EXC",
    "FE", "LNT", "NEE", "NI", "NRG", "NWS", "PCG", "PEG",
    "PNW", "PPL", "SRE", "SO", "VST", "WEC", "XEL",
]
# De-duplicate while preserving order (a few tickers appear in two sectors above)
_seen: set = set()
UNIVERSE = [t for t in UNIVERSE if t not in _seen and not _seen.add(t)]

# ── Dates ─────────────────────────────────────────────────────────────────
START_DATE    = "2017-01-01"   # start of data download (needs warmup for features)
END_DATE      = "2025-01-01"   # end of backtest - use historical data only

# ── Strategy ──────────────────────────────────────────────────────────────
LOOKBACK_WEEKS = 52            # feature lookback (52 weeks = 1 year warmup needed)
TARGET_VOL     = 0.12          # 12% annualised portfolio volatility target
CASH           = 10_000        # starting capital ($)

# ── Walk-forward ──────────────────────────────────────────────────────────
MIN_TRAIN_WEEKS   = 52         # minimum training history before first fold
RETRAIN_FREQ      = "QS"       # quarterly retraining (Quarter Start)
FORWARD_WEEKS     = 4          # predict 4-week return - smoother signal, less weekly noise
REBALANCE_WEEKS   = 4          # only rebalance every 4 weeks (cuts TC ~4×)

# ── Regime filter ─────────────────────────────────────────────────────────
SPY_MA_WEEKS      = 40         # 200 trading days ≈ 40 weekly bars

# ── Position sizing ────────────────────────────────────────────────────────
DECILE_N          = 8          # top/bottom N stocks (soft-weight scheme uses all stocks)
SOFT_WEIGHT_ALPHA = 3.0        # steepness of soft-weight score → weight mapping
MAX_LEVERAGE      = 1.5        # cap vol-scaled leverage at 1.5×

# ── Costs ─────────────────────────────────────────────────────────────────
TC_BPS            = 5          # 5 bps per side (realistic for liquid S&P 500 large caps)
PAPER_COMMISSION  = 5.0        # $5 flat per trade for paper trading

# ── Paths ─────────────────────────────────────────────────────────────────
from pathlib import Path as _Path

_ROOT = _Path(__file__).parent          # absolute path to the project directory

PRICES_PATH       = _ROOT / "data"    / "prices.parquet"
FUNDAMENTALS_PATH = _ROOT / "data"    / "fundamentals.json"
IC_LOG_PATH       = _ROOT / "results" / "ic_log.csv"
BACKTEST_PATH     = _ROOT / "results" / "backtest.csv"
PREDICTIONS_PATH  = _ROOT / "results" / "predictions.parquet"
MODEL_PATH        = _ROOT / "models"  / "lgbm_latest.pkl"
DASHBOARD_PATH    = _ROOT / "results" / "plots" / "dashboard.png"
PAPER_PLOT_PATH   = _ROOT / "results" / "plots" / "paper_portfolio.png"

FUNDAMENTALS_TTL_DAYS = 7   # refresh .info cache if older than this

# ── Feature columns (30 total: 25 technical + 5 fundamental) ──────────────
FEATURE_COLS = [
    # ── Technical features (25) ──────────────────────────────────────────
    "mom_4w",            # 4-week price momentum
    "mom_13w",           # 13-week momentum (3 months)
    "mom_26w",           # 26-week momentum (6 months)
    "mom_52w",           # 52-week momentum (1 year)
    "rsi_14",            # RSI with 14 weekly periods
    "vol_zscore_4w",     # volume z-score over 4 weeks
    "price_52w_high",    # price / 52-week high
    "idio_vol",          # idiosyncratic volatility (52w regression residual vs SPY)
    "rel_str_spy_4w",    # relative strength vs SPY 4 weeks
    "rel_str_spy_13w",   # relative strength vs SPY 13 weeks
    "reversal_1w",       # 1-week mean reversion (contrarian)
    "vol_4w",            # realised volatility 4 weeks (annualised)
    "vol_13w",           # realised volatility 13 weeks (annualised)
    "macd_hist",         # MACD histogram (4w/9w/4w EMA for weekly bars)
    "bb_pos",            # Bollinger Band position (20w, 2 std)
    "atr_4w_pct",        # ATR 4 weeks as % of price
    "ma_cross_4_13",     # 4w MA / 13w MA - 1
    "ma_cross_13_26",    # 13w MA / 26w MA - 1
    "dist_52w_low",      # price / 52-week low
    "rolling_beta",      # rolling 52w beta vs SPY
    "vol_trend",         # short-term volume vs long-term (4w avg / 13w avg)
    "price_to_ma26",     # price / 26-week moving average
    "skew_13w",          # return skewness over 13 weeks
    "corr_spy_13w",      # rolling correlation with SPY 13 weeks
    "up_down_vol",       # upside vol / downside vol 13 weeks
    # ── Fundamental features (5) - sourced from yfinance .info ───────────
    # Note: yfinance .info returns current values only, not historical
    # point-in-time data. These are treated as static cross-sectional signals
    # (same value broadcast across all dates). This introduces mild lookahead
    # bias in the historical backtest; the signals are most valid for live
    # trading via signals.py where current fundamentals are appropriate.
    "pe_ratio",          # trailing 12-month P/E (trailingPE)
    "pb_ratio",          # price-to-book ratio (priceToBook)
    "profit_margin",     # net profit margin TTM (profitMargins)
    "revenue_growth",    # YoY revenue growth (revenueGrowth)
    "debt_to_equity",    # total debt / shareholders' equity (debtToEquity)
]

# ── LightGBM hyperparameters ───────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":          "regression",
    "n_estimators":       300,
    "learning_rate":      0.02,
    "num_leaves":         15,         # reduced from 31 - less overfit
    "min_child_samples":  50,         # increased - forces broader splits
    "subsample":          0.7,
    "colsample_bytree":   0.7,
    "reg_alpha":          0.1,        # more L1
    "reg_lambda":         1.0,        # more L2
    "random_state":       42,
    "n_jobs":             -1,
    "verbosity":          -1,
}

# ── Risk controls ─────────────────────────────────────────────────────────────
STOP_LOSS_PCT   = 0.08   # exit a position the following week if it loses >8% in one week
MAX_STOCK_WEIGHT = 0.05  # hard cap: no single stock can exceed 5% of gross portfolio value

# ── GICS Sector map: ticker → sector (11 GICS sectors) ───────────────────────
# Source: GICS classifications for S&P 500 constituents.
# Any ticker absent from this map is treated as "Unknown" and excluded from
# sector-neutralisation (it keeps its raw soft weight).
SECTOR_MAP: dict = {
    # Communication Services
    "CHTR": "Communication Services", "CMCSA": "Communication Services",
    "DIS":  "Communication Services", "EA":    "Communication Services",
    "FOX":  "Communication Services", "FOXA":  "Communication Services",
    "GOOGL":"Communication Services", "GOOG":  "Communication Services",
    "LYV":  "Communication Services", "META":  "Communication Services",
    "NFLX": "Communication Services", "NWS":   "Communication Services",
    "NWSA": "Communication Services", "OMC":   "Communication Services",
    "PARA": "Communication Services", "T":     "Communication Services",
    "TMUS": "Communication Services", "TTWO":  "Communication Services",
    "VZ":   "Communication Services", "WBD":   "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "AN":   "Consumer Discretionary",
    "APTV": "Consumer Discretionary", "AZO":  "Consumer Discretionary",
    "BBY":  "Consumer Discretionary", "BKNG": "Consumer Discretionary",
    "BWA":  "Consumer Discretionary", "CCL":  "Consumer Discretionary",
    "CMG":  "Consumer Discretionary", "DPZ":  "Consumer Discretionary",
    "DRI":  "Consumer Discretionary", "EBAY": "Consumer Discretionary",
    "EXPE": "Consumer Discretionary", "F":    "Consumer Discretionary",
    "GM":   "Consumer Discretionary", "HLT":  "Consumer Discretionary",
    "HD":   "Consumer Discretionary", "LEN":  "Consumer Discretionary",
    "LVS":  "Consumer Discretionary", "LOW":  "Consumer Discretionary",
    "MAR":  "Consumer Discretionary", "MCD":  "Consumer Discretionary",
    "MGM":  "Consumer Discretionary", "MHK":  "Consumer Discretionary",
    "NKE":  "Consumer Discretionary", "NVR":  "Consumer Discretionary",
    "NCLH": "Consumer Discretionary", "ORLY": "Consumer Discretionary",
    "PHM":  "Consumer Discretionary", "RCL":  "Consumer Discretionary",
    "RL":   "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TGT":  "Consumer Discretionary",
    "TJX":  "Consumer Discretionary", "TPR":  "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "ULTA": "Consumer Discretionary",
    "WYNN": "Consumer Discretionary", "YUM":  "Consumer Discretionary",
    "CVNA": "Consumer Discretionary", "DASH": "Consumer Discretionary",
    "DECK": "Consumer Discretionary", "LULU": "Consumer Discretionary",
    # Consumer Staples
    "ADM":  "Consumer Staples", "CAG":  "Consumer Staples",
    "CHD":  "Consumer Staples", "CL":   "Consumer Staples",
    "CLX":  "Consumer Staples", "COST": "Consumer Staples",
    "GIS":  "Consumer Staples", "HRL":  "Consumer Staples",
    "HSY":  "Consumer Staples", "K":    "Consumer Staples",
    "KHC":  "Consumer Staples", "KMB":  "Consumer Staples",
    "KO":   "Consumer Staples", "KR":   "Consumer Staples",
    "MDLZ": "Consumer Staples", "MKC":  "Consumer Staples",
    "MO":   "Consumer Staples", "MNST": "Consumer Staples",
    "PEP":  "Consumer Staples", "PG":   "Consumer Staples",
    "PM":   "Consumer Staples", "SJM":  "Consumer Staples",
    "STZ":  "Consumer Staples", "SYY":  "Consumer Staples",
    "TAP":  "Consumer Staples", "TSN":  "Consumer Staples",
    "WMT":  "Consumer Staples", "BG":   "Consumer Staples",
    "KVUE": "Consumer Staples",
    # Energy
    "APA":  "Energy", "BKR":  "Energy",
    "COP":  "Energy", "CVX":  "Energy",
    "DVN":  "Energy", "EOG":  "Energy",
    "EQT":  "Energy", "FANG": "Energy",
    "FCX":  "Energy", "HAL":  "Energy",
    "HES":  "Energy", "KMI":  "Energy",
    "MPC":  "Energy", "MRO":  "Energy",
    "OKE":  "Energy", "OXY":  "Energy",
    "PSX":  "Energy", "SLB":  "Energy",
    "TRGP": "Energy", "VLO":  "Energy",
    "XOM":  "Energy",
    # Financials
    "AFL":   "Financials", "AIG":   "Financials",
    "AIZ":   "Financials", "AJG":   "Financials",
    "ALL":   "Financials", "AMP":   "Financials",
    "AON":   "Financials", "AXP":   "Financials",
    "BAC":   "Financials", "BEN":   "Financials",
    "BK":    "Financials", "BLK":   "Financials",
    "BRK-B": "Financials", "BRO":   "Financials",
    "C":     "Financials", "CB":    "Financials",
    "CBOE":  "Financials", "CFG":   "Financials",
    "CINF":  "Financials", "CME":   "Financials",
    "COF":   "Financials", "CNA":   "Financials",
    "DFS":   "Financials", "EG":    "Financials",
    "ERIE":  "Financials", "FIS":   "Financials",
    "FISV":  "Financials", "FITB":  "Financials",
    "GL":    "Financials", "GPN":   "Financials",
    "GS":    "Financials", "HBAN":  "Financials",
    "ICE":   "Financials", "IVZ":   "Financials",
    "JPM":   "Financials", "KEY":   "Financials",
    "KKR":   "Financials", "L":     "Financials",
    "LNC":   "Financials", "MA":    "Financials",
    "MCO":   "Financials", "MET":   "Financials",
    "MS":    "Financials", "MSCI":  "Financials",
    "MTB":   "Financials", "NDAQ":  "Financials",
    "NTRS":  "Financials", "PFG":   "Financials",
    "PGR":   "Financials", "PNC":   "Financials",
    "PRU":   "Financials", "RJF":   "Financials",
    "RF":    "Financials", "SCHW":  "Financials",
    "SPGI":  "Financials", "STT":   "Financials",
    "SYF":   "Financials", "TFC":   "Financials",
    "TROW":  "Financials", "TRV":   "Financials",
    "USB":   "Financials", "V":     "Financials",
    "WFC":   "Financials", "WRB":   "Financials",
    "APO":   "Financials", "ARES":  "Financials",
    "BNY":   "Financials", "BX":    "Financials",
    "CPAY":  "Financials", "HOOD":  "Financials",
    "IBKR":  "Financials", "KDP":   "Financials",
    # Health Care
    "A":    "Health Care", "ABBV":  "Health Care",
    "ABT":  "Health Care", "ALGN":  "Health Care",
    "AMGN": "Health Care", "BAX":   "Health Care",
    "BDX":  "Health Care", "BIIB":  "Health Care",
    "BMY":  "Health Care", "BSX":   "Health Care",
    "CAH":  "Health Care", "CNC":   "Health Care",
    "COO":  "Health Care", "CRL":   "Health Care",
    "CVS":  "Health Care", "DXCM":  "Health Care",
    "DHR":  "Health Care", "DVA":   "Health Care",
    "ELV":  "Health Care", "EW":    "Health Care",
    "GEHC": "Health Care", "GILD":  "Health Care",
    "HCA":  "Health Care", "HUM":   "Health Care",
    "HSIC": "Health Care", "IDXX":  "Health Care",
    "INCY": "Health Care", "IQV":   "Health Care",
    "ISRG": "Health Care", "JNJ":   "Health Care",
    "LH":   "Health Care", "LLY":   "Health Care",
    "MCK":  "Health Care", "MDT":   "Health Care",
    "MRNA": "Health Care", "MRK":   "Health Care",
    "MTD":  "Health Care", "PFE":   "Health Care",
    "PODD": "Health Care", "REGN":  "Health Care",
    "RMD":  "Health Care", "RVTY":  "Health Care",
    "STE":  "Health Care", "SYK":   "Health Care",
    "TMO":  "Health Care", "UNH":   "Health Care",
    "VRTX": "Health Care", "WAT":   "Health Care",
    "ZBH":  "Health Care", "ZTS":   "Health Care",
    "SOLV": "Health Care",
    # Industrials
    "ALLE": "Industrials", "AME":  "Industrials",
    "AOS":  "Industrials", "AXON": "Industrials",
    "BA":   "Industrials", "BLDR": "Industrials",
    "CAT":  "Industrials", "CHRW": "Industrials",
    "CTAS": "Industrials", "DAL":  "Industrials",
    "DE":   "Industrials", "DOV":  "Industrials",
    "EME":  "Industrials", "EMR":  "Industrials",
    "EXPD": "Industrials", "FAST": "Industrials",
    "FDX":  "Industrials", "FTV":  "Industrials",
    "GD":   "Industrials", "GE":   "Industrials",
    "GEV":  "Industrials", "GWW":  "Industrials",
    "HON":  "Industrials", "HUBB": "Industrials",
    "HII":  "Industrials", "HWM":  "Industrials",
    "IR":   "Industrials", "ITW":  "Industrials",
    "J":    "Industrials", "JBHT": "Industrials",
    "JCI":  "Industrials", "LDOS": "Industrials",
    "LHX":  "Industrials", "LII":  "Industrials",
    "LMT":  "Industrials", "LUV":  "Industrials",
    "MAS":  "Industrials", "MMM":  "Industrials",
    "NOC":  "Industrials", "NSC":  "Industrials",
    "ODFL": "Industrials", "OTIS": "Industrials",
    "PCAR": "Industrials", "PH":   "Industrials",
    "PNR":  "Industrials", "PWR":  "Industrials",
    "ROK":  "Industrials", "ROL":  "Industrials",
    "ROP":  "Industrials", "RSG":  "Industrials",
    "RTX":  "Industrials", "SNA":  "Industrials",
    "SWK":  "Industrials", "TDG":  "Industrials",
    "TDY":  "Industrials", "TT":   "Industrials",
    "TXT":  "Industrials", "UAL":  "Industrials",
    "UBER": "Industrials", "UNP":  "Industrials",
    "UPS":  "Industrials", "URI":  "Industrials",
    "WAB":  "Industrials", "WM":   "Industrials",
    "WMB":  "Industrials", "XYL":  "Industrials",
    "CARR": "Industrials", "GNRC": "Industrials",
    # Information Technology (ICE already mapped to Financials above)
    "AAPL": "Information Technology", "ACN":  "Information Technology",
    "ADBE": "Information Technology", "ADI":  "Information Technology",
    "ADP":  "Information Technology", "ADSK": "Information Technology",
    "AMAT": "Information Technology", "AMD":  "Information Technology",
    "ANET": "Information Technology", "APH":  "Information Technology",
    "APP":  "Information Technology", "AVGO": "Information Technology",
    "BR":   "Information Technology", "CDNS": "Information Technology",
    "CDW":  "Information Technology", "CRM":  "Information Technology",
    "CRWD": "Information Technology", "CSCO": "Information Technology",
    "CTSH": "Information Technology", "DDOG": "Information Technology",
    "DELL": "Information Technology", "FFIV": "Information Technology",
    "FICO": "Information Technology", "FLEX": "Information Technology",
    "FTNT": "Information Technology", "GLW":  "Information Technology",
    "HPE":  "Information Technology", "HPQ":  "Information Technology",
    "IBM":  "Information Technology", "INTC": "Information Technology",
    "INTU": "Information Technology", "IT":   "Information Technology",
    "JKHY": "Information Technology", "KEYS": "Information Technology",
    "KLAC": "Information Technology", "LRCX": "Information Technology",
    "MCHP": "Information Technology", "MPWR": "Information Technology",
    "MRVL": "Information Technology", "MSFT": "Information Technology",
    "MSI":  "Information Technology", "MU":   "Information Technology",
    "NXPI": "Information Technology", "NOW":  "Information Technology",
    "NTAP": "Information Technology", "ON":   "Information Technology",
    "ORCL": "Information Technology", "PANW": "Information Technology",
    "PAYX": "Information Technology", "PLTR": "Information Technology",
    "PTC":  "Information Technology", "PYPL": "Information Technology",
    "QCOM": "Information Technology", "SMCI": "Information Technology",
    "SNPS": "Information Technology", "STX":  "Information Technology",
    "SWKS": "Information Technology", "TEL":  "Information Technology",
    "TER":  "Information Technology", "TXN":  "Information Technology",
    "TYL":  "Information Technology", "VEEV": "Information Technology",
    "VRSN": "Information Technology", "VRT":  "Information Technology",
    "WDC":  "Information Technology", "WDAY": "Information Technology",
    "ZBRA": "Information Technology", "TTD":  "Information Technology",
    "COIN": "Information Technology",
    # Materials
    "ALB":  "Materials", "AMCR": "Materials",
    "APD":  "Materials", "AVY":  "Materials",
    "BALL": "Materials", "CF":   "Materials",
    "DD":   "Materials", "DOW":  "Materials",
    "ECL":  "Materials", "EMN":  "Materials",
    "FMC":  "Materials", "IFF":  "Materials",
    "IP":   "Materials", "LIN":  "Materials",
    "LYB":  "Materials", "MLM":  "Materials",
    "MOS":  "Materials", "NEM":  "Materials",
    "NUE":  "Materials", "PKG":  "Materials",
    "PPG":  "Materials", "STLD": "Materials",
    "SW":   "Materials", "VMC":  "Materials",
    "WRK":  "Materials",
    # Real Estate
    "AMT":  "Real Estate", "ARE":  "Real Estate",
    "AVB":  "Real Estate", "BXP":  "Real Estate",
    "CBRE": "Real Estate", "CCI":  "Real Estate",
    "CPT":  "Real Estate", "DLR":  "Real Estate",
    "DOC":  "Real Estate", "EQIX": "Real Estate",
    "EQR":  "Real Estate", "ESS":  "Real Estate",
    "EXR":  "Real Estate", "FRT":  "Real Estate",
    "HST":  "Real Estate", "INVH": "Real Estate",
    "IRM":  "Real Estate", "KIM":  "Real Estate",
    "MAA":  "Real Estate", "O":    "Real Estate",
    "PLD":  "Real Estate", "PSA":  "Real Estate",
    "REG":  "Real Estate", "SBAC": "Real Estate",
    "SPG":  "Real Estate", "UDR":  "Real Estate",
    "VICI": "Real Estate", "VTR":  "Real Estate",
    "WELL": "Real Estate", "WY":   "Real Estate",
    # Utilities (NWS already mapped to Communication Services above)
    "AEE":  "Utilities", "AEP":  "Utilities",
    "AES":  "Utilities", "ATO":  "Utilities",
    "AWK":  "Utilities", "CMS":  "Utilities",
    "CNP":  "Utilities", "D":    "Utilities",
    "DTE":  "Utilities", "DUK":  "Utilities",
    "ED":   "Utilities", "EIX":  "Utilities",
    "ES":   "Utilities", "ETR":  "Utilities",
    "EVRG": "Utilities", "EXC":  "Utilities",
    "FE":   "Utilities", "LNT":  "Utilities",
    "NEE":  "Utilities", "NI":   "Utilities",
    "NRG":  "Utilities", "PCG":  "Utilities",
    "PEG":  "Utilities", "PNW":  "Utilities",
    "PPL":  "Utilities", "SRE":  "Utilities",
    "SO":   "Utilities", "VST":  "Utilities",
    "WEC":  "Utilities", "XEL":  "Utilities",
}
