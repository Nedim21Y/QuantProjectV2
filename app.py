"""
QuantProject v2 - Streamlit Dashboard
Run:  streamlit run app.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

warnings.filterwarnings("ignore")

# Ensure project is importable regardless of where `streamlit run` is called from
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import config
from main import compute_features, fetch_fundamentals
from signals import load_model, download_recent_data


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantProject v2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_backtest() -> pd.DataFrame:
    if not config.BACKTEST_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(config.BACKTEST_PATH, parse_dates=["Date"]).set_index("Date")


@st.cache_data(show_spinner=False)
def load_ic_log() -> pd.DataFrame:
    if not config.IC_LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(
        config.IC_LOG_PATH,
        parse_dates=["pred_start", "pred_end", "train_end"],
    )


@st.cache_resource(show_spinner=False)
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def get_live_signals(_version: int = 0):
    """
    Download recent weekly prices, compute all 30 features (technical + fundamental),
    score every stock with the 3-model ensemble.
    Cached 1 hour; increment _version to force a refresh.
    Returns (scored_df, signal_date) or (empty_df, None).
    """
    model, feat_cols = get_model()
    if model is None or feat_cols is None:
        return pd.DataFrame(), None

    data    = download_recent_data(n_weeks=config.LOOKBACK_WEEKS + 5)
    fund_df = fetch_fundamentals(config.UNIVERSE)
    panel   = compute_features(data, fund_df)

    if panel.empty:
        return pd.DataFrame(), None

    dates       = panel.index.get_level_values("Date").unique().sort_values()
    latest_date = dates[-1]
    latest      = panel.xs(latest_date, level="Date").copy()

    # Fill any residual NaN in fundamental features with column median
    for col in feat_cols:
        if col in latest.columns:
            med = latest[col].median()
            latest[col] = latest[col].fillna(0.0 if pd.isna(med) else med)

    latest = latest.dropna(subset=feat_cols)
    if latest.empty:
        return pd.DataFrame(), latest_date

    X = latest[feat_cols].values
    if isinstance(model, dict) and "lgbm" in model:
        s_lgbm  = model["lgbm"].predict(X)
        s_xgb   = model["xgb"].predict(X)
        s_ridge = model["ridge"].predict(model["scaler"].transform(X))
        scores  = (s_lgbm + s_xgb + s_ridge) / 3.0
    else:
        scores = model.predict(X)

    out = latest[["mom_4w"]].copy()
    out["score"]    = scores
    out["pct_rank"] = out["score"].rank(pct=True)
    out["sector"]   = [config.SECTOR_MAP.get(t, "-") for t in out.index]
    return out.sort_values("score", ascending=False), latest_date


# ── Sidebar metrics ───────────────────────────────────────────────────────────

bt = load_backtest()
ic = load_ic_log()

if not bt.empty:
    _total  = bt["Portfolio_Value"].iloc[-1] / config.CASH - 1
    _n      = len(bt)
    _annr   = (1 + _total) ** (52 / _n) - 1
    _annv   = bt["Strategy_Return"].std() * np.sqrt(52)
    _sharpe = (_annr - 0.05) / _annv if _annv > 0 else 0.0
    _last   = bt.index.max().strftime("%Y-%m-%d")
else:
    _total = _annr = _sharpe = 0.0
    _last = "-"

_mean_ic = float(ic["mean_ic"].mean()) if not ic.empty else 0.0

with st.sidebar:
    st.title("📈 QuantProject v2")
    st.caption("S&P 500 long/short · LightGBM + XGBoost + Ridge")
    st.divider()

    ca, cb = st.columns(2)
    ca.metric("Universe",    str(len(config.UNIVERSE)))
    cb.metric("CV Folds",    str(len(ic)))
    ca.metric("Mean OOS IC", f"{_mean_ic:+.4f}")
    cb.metric("Sharpe",      f"{_sharpe:.3f}")
    ca.metric("Ann. Return", f"{_annr:+.1%}")
    cb.metric("Last Run",    _last)

    st.divider()
    st.caption(f"Period: {config.START_DATE} → {config.END_DATE}")
    st.caption(f"Rebal: every {config.REBALANCE_WEEKS}w · TC: {config.TC_BPS} bps/side")
    st.caption(f"Target vol: {config.TARGET_VOL:.0%} · Max lev: {config.MAX_LEVERAGE}×")
    st.caption(f"Stop-loss: {config.STOP_LOSS_PCT:.0%} · Max pos: {config.MAX_STOCK_WEIGHT:.0%}")

    page = st.radio(
        "Navigate",
        [
            "📡 Live Signals",
            "📊 Backtest Results",
            "📈 IC History",
            "🤖 Model Comparison",
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 - Live Signals
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📡 Live Signals":
    st.header("📡 Live Signals")
    st.caption(
        "Scores all S&P 500 stocks using the latest weekly price bar. "
        "Click any column header to sort. Hit **Refresh** to pull fresh prices."
    )

    ctl1, ctl2, _ = st.columns([1, 1, 4])
    top_n   = int(ctl1.number_input("Stocks per side", 5, 50, 20, 5, key="top_n"))
    refresh = ctl2.button("🔄  Refresh", use_container_width=True)

    if refresh:
        st.session_state["sig_ver"] = st.session_state.get("sig_ver", 0) + 1

    with st.spinner("Downloading prices and scoring stocks…"):
        sig_df, sig_date = get_live_signals(st.session_state.get("sig_ver", 0))

    if sig_df.empty:
        st.error("Model not found or no data available. Run `python main.py` first.")
        st.stop()

    st.success(f"Signal date: **{sig_date.date()}** - **{len(sig_df)}** stocks scored")

    def _label(pct: float) -> str:
        if pct >= 0.95:    return "▲▲▲ Strong Buy"
        elif pct >= 0.85:  return "▲▲  Buy"
        elif pct >= 0.75:  return "▲   Weak Buy"
        elif pct <= 0.05:  return "▼▼▼ Strong Sell"
        elif pct <= 0.15:  return "▼▼  Sell"
        elif pct <= 0.25:  return "▼   Weak Sell"
        return             "-   Neutral"

    sig_df = sig_df.copy()
    sig_df["Signal"]    = sig_df["pct_rank"].apply(_label)
    sig_df["Rank %"]    = (sig_df["pct_rank"] * 100).round(1)
    sig_df["Score"]     = sig_df["score"].round(5)
    sig_df["4w Mom %"]  = (sig_df["mom_4w"] * 100).round(2)
    sig_df["Sector"]    = sig_df["sector"]

    show_cols = ["Signal", "Rank %", "Score", "4w Mom %", "Sector"]
    row_h = min(35 * top_n + 42, 740)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"🟢 Top {top_n} BUY")
        st.dataframe(
            sig_df.head(top_n)[show_cols],
            use_container_width=True,
            height=row_h,
        )
    with c2:
        st.subheader(f"🔴 Top {top_n} SELL")
        st.dataframe(
            sig_df.tail(top_n).iloc[::-1][show_cols],
            use_container_width=True,
            height=row_h,
        )

    st.caption(
        "⚠️ Educational purposes only. Not financial advice. "
        "Past model performance does not guarantee future returns."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 - Backtest Results
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Backtest Results":
    st.header("📊 Backtest Results")
    st.caption(f"Walk-forward backtest · {config.START_DATE} → {config.END_DATE}")

    if bt.empty:
        st.error("No backtest data found. Run `python main.py` first.")
        st.stop()

    _spy_tot = bt["SPY_Value"].iloc[-1] / config.CASH - 1
    _mdd     = ((bt["Portfolio_Value"] - bt["Portfolio_Value"].cummax())
                 / bt["Portfolio_Value"].cummax()).min()
    _wr      = (bt["Strategy_Return"] > 0).mean()
    _tc_drag = bt["TC_Cost"].sum()
    _spy_r   = bt["SPY_Return"]
    _sr      = bt["Strategy_Return"]
    _cov     = _sr.cov(_spy_r)
    _var     = _spy_r.var()
    _beta    = _cov / _var if _var > 0 else 0.0
    _alpha   = (_annr - 0.05) - _beta * ((1 + _spy_r.mean() * 52) - 1 - 0.05)

    r1 = st.columns(4)
    r1[0].metric("Total Return",      f"{_total:+.1%}", f"vs SPY {_spy_tot:+.1%}")
    r1[1].metric("Annualised Return",  f"{_annr:+.1%}")
    r1[2].metric("Sharpe Ratio",       f"{_sharpe:.3f}")
    r1[3].metric("Max Drawdown",       f"{_mdd:.1%}")

    r2 = st.columns(4)
    r2[0].metric("Weekly Win Rate",    f"{_wr:.1%}")
    r2[1].metric("Alpha (ann.)",       f"{_alpha:+.1%}")
    r2[2].metric("Beta vs SPY",        f"{_beta:.3f}")
    r2[3].metric("Total TC Drag",      f"{_tc_drag:.2%}")

    st.divider()

    if config.DASHBOARD_PATH.exists():
        st.image(str(config.DASHBOARD_PATH), use_container_width=True)
    else:
        st.warning(
            "Dashboard image not found. Run `python analytics.py` to generate it."
        )

    st.divider()
    with st.expander("📋 Weekly data - last 52 weeks"):
        disp = bt.tail(52)[[
            "Strategy_Return", "SPY_Return", "Portfolio_Value",
            "SPY_Value", "In_Regime", "Turnover", "Vol_Scale", "TC_Cost",
        ]].copy()
        for c in ["Strategy_Return", "SPY_Return", "Turnover"]:
            disp[c] = (disp[c] * 100).round(3)
        disp["Portfolio_Value"] = disp["Portfolio_Value"].round(2)
        disp["SPY_Value"]       = disp["SPY_Value"].round(2)
        st.dataframe(disp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 - IC History
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 IC History":
    st.header("📈 IC History")
    st.caption(
        "Out-of-sample Spearman Information Coefficient per walk-forward fold "
        "(one fold ≈ one calendar quarter). Dashed line at IC = 0.03."
    )

    if ic.empty:
        st.error("No IC log. Run `python main.py` first.")
        st.stop()

    ic = ic.copy()
    ic["quarter"] = ic["pred_start"].dt.to_period("Q").astype(str)
    n = len(ic)

    # ── Grouped bar chart - all 4 models ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, n * 0.65), 5))
    x = np.arange(n)
    w = 0.18
    model_specs = [
        ("ic_lgbm",  "LightGBM",   "#2196F3"),
        ("ic_xgb",   "XGBoost",    "#FF9800"),
        ("ic_ridge", "Ridge",      "#9C27B0"),
        ("mean_ic",  "Ensemble ★", "#4CAF50"),
    ]
    for (col, label, color), offset in zip(model_specs, [-1.5, -0.5, 0.5, 1.5]):
        vals = ic[col].fillna(0).values
        bars = ax.bar(x + offset * w, vals, w, label=label,
                      color=color, alpha=0.82, edgecolor="none")
        for bar, v in zip(bars, vals):
            if v < 0:
                bar.set_alpha(0.40)
                bar.set_edgecolor("#c62828")
                bar.set_linewidth(0.8)

    ax.axhline(0,    color="#444",    linewidth=0.9)
    ax.axhline(0.03, color="#4CAF50", linewidth=1.6, linestyle="--",
               label="IC = 0.03 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(ic["quarter"], rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("Spearman IC", fontsize=10)
    ax.set_title("Walk-forward OOS IC by Fold and Model",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.grid(axis="y", alpha=0.22, linestyle=":")
    ax.set_xlim(-0.6, n - 0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Cumulative ensemble IC ────────────────────────────────────────────────
    st.divider()
    st.subheader("Cumulative Ensemble IC")
    cum = ic["mean_ic"].fillna(0).cumsum().values
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(range(n), cum, color="#4CAF50", linewidth=2.2, zorder=3)
    ax2.fill_between(range(n), cum, alpha=0.12, color="#4CAF50")
    ax2.axhline(0, color="#444", linewidth=0.9)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(ic["quarter"], rotation=45, ha="right", fontsize=8.5)
    ax2.set_ylabel("Cumulative IC", fontsize=10)
    ax2.set_title("Cumulative OOS Ensemble IC", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(alpha=0.22, linestyle=":")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # ── Summary stats ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary Statistics")
    rows = []
    for col, label in [
        ("ic_lgbm",  "LightGBM"),
        ("ic_xgb",   "XGBoost"),
        ("ic_ridge", "Ridge"),
        ("mean_ic",  "Ensemble ★"),
    ]:
        v = ic[col].dropna()
        rows.append({
            "Model":         label,
            "Mean IC":       round(float(v.mean()), 5),
            "Std IC":        round(float(v.std()),  5),
            "Min IC":        round(float(v.min()),  5),
            "Max IC":        round(float(v.max()),  5),
            "IC > 0 (%)":    round(float((v > 0).mean())    * 100, 1),
            "IC > 0.03 (%)": round(float((v > 0.03).mean()) * 100, 1),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 - Model Comparison
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Comparison":
    st.header("🤖 Model Comparison")
    st.caption(
        "Per-fold OOS Spearman IC for each component of the ensemble stack. "
        "🟢 Green = IC ≥ 0.05 · 🟡 Yellow = negative · 🔴 Red = IC < -0.02"
    )

    if ic.empty:
        st.error("No IC log. Run `python main.py` first.")
        st.stop()

    ic = ic.copy()
    ic["quarter"] = ic["pred_start"].dt.to_period("Q").astype(str)

    # ── Per-fold styled table ─────────────────────────────────────────────────
    tbl = ic[[
        "fold", "quarter", "n_dates",
        "ic_lgbm", "ic_xgb", "ic_ridge", "mean_ic", "ic_pos_rate",
    ]].copy()
    tbl.columns = [
        "Fold", "Quarter", "N Weeks",
        "LightGBM IC", "XGBoost IC", "Ridge IC", "Ensemble IC", "IC>0 Rate",
    ]
    tbl["IC>0 Rate"] = (tbl["IC>0 Rate"] * 100).round(1).astype(str) + "%"
    tbl = tbl.set_index("Fold")

    IC_COLS = ["LightGBM IC", "XGBoost IC", "Ridge IC", "Ensemble IC"]

    def _cell_color(val):
        if not isinstance(val, (int, float)) or np.isnan(float(val)):
            return ""
        v = float(val)
        if v >= 0.05:    return "background-color: #c8e6c9; color: #1b5e20"
        elif v >= 0.02:  return "background-color: #dcedc8; color: #33691e"
        elif v >= 0:     return "background-color: #f1f8e9"
        elif v >= -0.02: return "background-color: #fff9c4; color: #e65100"
        else:            return "background-color: #ffcdd2; color: #b71c1c"

    styled = (
        tbl.style
        .map(_cell_color, subset=IC_COLS)
        .format({c: "{:+.5f}" for c in IC_COLS})
    )
    st.dataframe(
        styled,
        use_container_width=True,
        height=min(38 * len(tbl) + 42, 620),
    )

    st.divider()

    # ── Box plots + best-model bar chart ──────────────────────────────────────
    ic_vals = {
        "LightGBM": ic["ic_lgbm"].dropna().values,
        "XGBoost":  ic["ic_xgb"].dropna().values,
        "Ridge":    ic["ic_ridge"].dropna().values,
        "Ensemble": ic["mean_ic"].dropna().values,
    }
    _colors = {
        "LightGBM": "#2196F3", "XGBoost": "#FF9800",
        "Ridge":    "#9C27B0", "Ensemble": "#4CAF50",
    }

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))

    bp = ax3a.boxplot(
        list(ic_vals.values()),
        labels=list(ic_vals.keys()),
        patch_artist=True,
        medianprops=dict(color="#222", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
    )
    for patch, key in zip(bp["boxes"], ic_vals.keys()):
        patch.set_facecolor(_colors[key])
        patch.set_alpha(0.55)
    ax3a.axhline(0,    color="#444",    linewidth=0.9)
    ax3a.axhline(0.03, color="#4CAF50", linewidth=1.5, linestyle="--",
                 label="IC = 0.03")
    ax3a.set_title("IC Distribution per Model",
                   fontsize=11, fontweight="bold", pad=10)
    ax3a.set_ylabel("OOS Spearman IC", fontsize=10)
    ax3a.legend(fontsize=9)
    ax3a.grid(axis="y", alpha=0.22, linestyle=":")

    ic_matrix = pd.DataFrame({
        "LightGBM": ic["ic_lgbm"].values,
        "XGBoost":  ic["ic_xgb"].values,
        "Ridge":    ic["ic_ridge"].values,
        "Ensemble": ic["mean_ic"].values,
    })
    best = ic_matrix.idxmax(axis=1).value_counts()
    bars = ax3b.bar(
        best.index, best.values,
        color=[_colors.get(m, "#999") for m in best.index],
        alpha=0.82, edgecolor="none",
    )
    for bar, val in zip(bars, best.values):
        ax3b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(int(val)),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax3b.set_title("Folds Where Each Model Had Highest IC",
                   fontsize=11, fontweight="bold", pad=10)
    ax3b.set_ylabel("Number of folds", fontsize=10)
    ax3b.grid(axis="y", alpha=0.22, linestyle=":")

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    # ── IC correlation heatmap ────────────────────────────────────────────────
    st.divider()
    st.subheader("IC Correlation Between Models")
    corr = pd.DataFrame({
        "LightGBM": ic["ic_lgbm"],
        "XGBoost":  ic["ic_xgb"],
        "Ridge":    ic["ic_ridge"],
        "Ensemble": ic["mean_ic"],
    }).corr().round(3)

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    im  = ax4.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    ax4.set_xticks(range(4))
    ax4.set_yticks(range(4))
    ax4.set_xticklabels(corr.columns, fontsize=10)
    ax4.set_yticklabels(corr.columns, fontsize=10)
    for i in range(4):
        for j in range(4):
            ax4.text(
                j, i, f"{corr.values[i, j]:.2f}",
                ha="center", va="center", fontsize=11,
                color="white" if abs(corr.values[i, j]) > 0.7 else "black",
            )
    plt.colorbar(im, ax=ax4, shrink=0.8)
    ax4.set_title("IC Correlation Matrix", fontsize=11, fontweight="bold", pad=10)
    fig4.tight_layout()

    _, mid, _ = st.columns([1, 2, 1])
    mid.pyplot(fig4)
    plt.close(fig4)
