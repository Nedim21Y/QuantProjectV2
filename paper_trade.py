"""
QuantProject v2 - paper_trade.py
Simulates $10,000 paper portfolio over the final year of the backtest period.
Uses the same walk-forward OOS predictions and cached prices as the main pipeline
so dates always align correctly.

Run: python paper_trade.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import config

PAPER_START   = pd.Timestamp(config.END_DATE) - pd.DateOffset(years=1)
PAPER_END     = pd.Timestamp(config.END_DATE)
STARTING_CASH = float(config.CASH)
TOP_N         = 10
COMMISSION    = config.PAPER_COMMISSION   # $5 flat per trade

BG = "#0b0b12"; CARD = "#12121e"; FG = "#e8e8f2"; DIM = "#55556a"
ACC = "#00d4ff"; GRN = "#00e676"; RED = "#ff3d5a"; YEL = "#ffd600"


def run_simulation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate weekly paper trading using cached prices + OOS predictions.
    Uses the same data and date index as the main backtest so there is no
    date alignment gap.
    """
    # ── Load required data ─────────────────────────────────────────────
    for path in [config.BACKTEST_PATH, config.PRICES_PATH, config.PREDICTIONS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Run main.py first - missing: {path}")

    bt   = pd.read_csv(config.BACKTEST_PATH, index_col=0, parse_dates=True)
    preds = pd.read_parquet(config.PREDICTIONS_PATH)

    # Load cached prices (same weekly dates as backtest)
    store  = pd.read_parquet(config.PRICES_PATH)
    close  = store["close"] if "close" in store else store
    spy    = store["spy_close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.squeeze()

    # Restrict to paper period
    bt_paper = bt[(bt.index >= PAPER_START) & (bt.index <= PAPER_END)].copy()
    if bt_paper.empty:
        raise ValueError(
            f"No backtest data for paper period {PAPER_START.date()} - {PAPER_END.date()}.\n"
            "Check config.END_DATE and re-run main.py."
        )

    preds_paper = preds[
        (preds.index.get_level_values("Date") >= PAPER_START) &
        (preds.index.get_level_values("Date") <= PAPER_END)
    ]
    if preds_paper.empty:
        raise ValueError(
            "No OOS predictions for the paper period. "
            "The walk-forward training may not have produced predictions in this date range. "
            "Check that PAPER_START falls within the walk-forward OOS window."
        )

    # ── Simulate week-by-week ─────────────────────────────────────────
    cash      = STARTING_CASH
    holdings  = {}   # ticker → shares
    all_rows  = []
    trade_log = []
    prev_port_value = STARTING_CASH  # track end-of-last-week value for return calc

    pred_dates      = preds_paper.index.get_level_values("Date").unique().sort_values()
    weeks_since_reb = 0   # rebalance every REBALANCE_WEEKS to match main backtest

    for week_date in pred_dates:
        # ── Mark portfolio at THIS week's close (end-of-week value) ───
        port_value_start = prev_port_value  # for return calc: last week's close

        # ── Rebalance only every REBALANCE_WEEKS weeks (matches backtest) ─
        weeks_since_reb += 1
        if weeks_since_reb >= config.REBALANCE_WEEKS or len(holdings) == 0:
            weeks_since_reb = 0

            # Determine new target positions
            if week_date in preds_paper.index.get_level_values("Date"):
                scores    = preds_paper.xs(week_date, level="Date")["score"]
                top_ticks = set(scores.nlargest(TOP_N).index.tolist())
            else:
                top_ticks = set(holdings.keys())  # hold current on missing week

            current_ticks = set(holdings.keys())
            to_sell = current_ticks - top_ticks   # exited top-N
            to_buy  = top_ticks - current_ticks   # entered top-N

            # ── Sell only stocks leaving the portfolio ────────────────
            for ticker in to_sell:
                shares = holdings.pop(ticker, 0)
                if shares == 0:
                    continue
                if week_date in close.index and ticker in close.columns:
                    p = close.loc[week_date, ticker]
                    if not np.isnan(p) and p > 0:
                        gross = shares * p
                        cash += max(gross - COMMISSION, 0)
                        trade_log.append({
                            "Date": week_date, "Action": "SELL",
                            "Ticker": ticker, "Shares": round(shares, 4),
                            "Price": round(p, 2), "Commission": COMMISSION,
                        })

            # ── Buy new entrants, sized to match equal-weight target ──
            total_target = port_value_start   # target total portfolio value
            alloc_per = total_target / TOP_N  # equal weight per holding
            for ticker in to_buy:
                if cash < COMMISSION + 1:
                    break
                if week_date not in close.index or ticker not in close.columns:
                    continue
                p = close.loc[week_date, ticker]
                if np.isnan(p) or p <= 0:
                    continue
                invest = min(alloc_per - COMMISSION, cash - COMMISSION)
                if invest <= 0:
                    continue
                shares = invest / p
                holdings[ticker] = shares
                cash -= invest + COMMISSION
                trade_log.append({
                    "Date": week_date, "Action": "BUY",
                    "Ticker": ticker, "Shares": round(shares, 4),
                    "Price": round(p, 2), "Commission": COMMISSION,
                })

        # ── Mark-to-market at end of week ─────────────────────────────
        port_value_end = cash
        for ticker, shares in holdings.items():
            if week_date in close.index and ticker in close.columns:
                p = close.loc[week_date, ticker]
                if not np.isnan(p):
                    port_value_end += shares * p

        wr = (port_value_end / port_value_start - 1) if port_value_start > 0 else 0.0
        spy_val = spy.loc[week_date] if week_date in spy.index else np.nan

        all_rows.append({
            "Date":           week_date,
            "Portfolio_Value": port_value_end,
            "Weekly_Return":  wr,
            "PnL":            port_value_end - port_value_start,
            "Holdings":       list(holdings.keys()),
            "SPY_Price":      spy_val,
        })
        prev_port_value = port_value_end  # update for next week's return calc

    weekly_df   = pd.DataFrame(all_rows).set_index("Date")
    trade_log_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
        columns=["Date", "Action", "Ticker", "Shares", "Price", "Commission"]
    )

    # ── Add SPY benchmark ──────────────────────────────────────────────
    spy_paper = spy.reindex(weekly_df.index, method="ffill").dropna()
    if len(spy_paper) > 0:
        weekly_df["SPY_Value"] = spy_paper / spy_paper.iloc[0] * STARTING_CASH

    return weekly_df, trade_log_df


def compute_paper_metrics(df: pd.DataFrame) -> dict:
    rets    = df["Weekly_Return"].replace([np.inf, -np.inf], np.nan).dropna()
    n       = len(rets)
    end_val = df["Portfolio_Value"].iloc[-1]
    total   = end_val / STARTING_CASH - 1
    ann_r   = (1 + total) ** (52 / max(n, 1)) - 1
    ann_v   = rets.std() * np.sqrt(52) if n > 1 else 0.0
    sharpe  = (ann_r - 0.05) / ann_v if ann_v > 0 else 0.0
    cum     = (1 + rets).cumprod()
    dd      = (cum / cum.cummax() - 1)
    mdd     = dd.min()
    wr      = float((rets > 0).mean())

    spy_tot = 0.0
    if "SPY_Value" in df.columns and not df["SPY_Value"].dropna().empty:
        spy_tot = df["SPY_Value"].dropna().iloc[-1] / STARTING_CASH - 1

    return dict(
        end_val=end_val, total=total, ann_r=ann_r, ann_v=ann_v,
        sharpe=sharpe, mdd=mdd, wr=wr, spy_tot=spy_tot, dd_ser=dd,
    )


def print_summary(df: pd.DataFrame, trade_log: pd.DataFrame, m: dict):
    sep      = "═" * 60
    n_buys   = len(trade_log[trade_log["Action"] == "BUY"])  if len(trade_log) > 0 else 0
    n_sells  = len(trade_log[trade_log["Action"] == "SELL"]) if len(trade_log) > 0 else 0
    total_tc = (n_buys + n_sells) * COMMISSION
    alpha    = m["total"] - m["spy_tot"]

    if m["total"] > 0.10 and alpha > 0.02:
        verdict = "✅  PROFITABLE & BEATING SPY - Strategy is working"
    elif m["total"] > 0.02:
        verdict = "⚠️  SMALL PROFIT - Paper trade 1 more quarter to confirm"
    elif m["total"] > -0.05:
        verdict = "⚠️  NEAR BREAKEVEN - Do NOT use real money yet"
    else:
        verdict = "❌  LOSING - Retrain model. Do NOT go live."

    print(f"\n{sep}")
    print(f"  PAPER TRADING SIMULATION")
    print(f"  Period : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Capital: ${STARTING_CASH:,.0f}  →  ${m['end_val']:,.0f}")
    print(f"{sep}")
    print(f"  Total Return        : {m['total']:>+.2%}")
    print(f"  SPY Return (period) : {m['spy_tot']:>+.2%}")
    print(f"  Alpha vs SPY        : {alpha:>+.2%}")
    print(f"  Annualised Return   : {m['ann_r']:>+.2%}")
    print(f"  Annualised Vol      : {m['ann_v']:>.2%}")
    print(f"  Sharpe Ratio        : {m['sharpe']:>.3f}")
    print(f"  Max Drawdown        : {m['mdd']:>.2%}")
    print(f"  Weekly Win Rate     : {m['wr']:>.1%}")
    print(f"  Total Trades        : {len(trade_log)}  ({n_buys} buys / {n_sells} sells)")
    print(f"  Total Commission    : ${total_tc:,.2f}")
    print(f"{sep}")
    print(f"  VERDICT: {verdict}")
    print(f"{sep}")

    print(f"\n  LAST 8 WEEKS:")
    print(f"  {'Date':<12} {'Return':>8} {'P&L':>10} {'Portfolio Value':>16}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*16}")
    for date, row in df.tail(8).iterrows():
        flag = "▲" if row["Weekly_Return"] >= 0 else "▼"
        print(f"  {str(date.date()):<12} {flag}{abs(row['Weekly_Return']):>6.2%}"
              f" {row['PnL']:>+9.2f}  ${row['Portfolio_Value']:>13,.2f}")


def plot_paper_dashboard(df: pd.DataFrame, m: dict, trade_log: pd.DataFrame):
    plt.rcParams.update({
        "figure.facecolor": BG, "text.color": FG,
        "font.family": "monospace", "font.size": 8,
    })
    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    fig.suptitle(
        f"Paper Trading - ${STARTING_CASH:,.0f} → ${m['end_val']:,.0f}  "
        f"({PAPER_START.date()} to {PAPER_END.date()})",
        color=FG, fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.08, right=0.97, top=0.93, bottom=0.06)

    def _ax(r, c, title=""):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e1e30")
        ax.tick_params(colors=DIM, labelsize=8)
        ax.grid(True, color="#1a1a2e", lw=0.5, ls="--", alpha=0.6)
        if title: ax.set_title(title, color=FG, fontsize=9, fontweight="bold", pad=6)
        return ax

    # Panel 1: Equity curve vs SPY
    ax1 = _ax(0, 0, "Paper Portfolio Equity vs SPY")
    ax1.plot(df.index, df["Portfolio_Value"], color=ACC, lw=2.0, label="Strategy")
    if "SPY_Value" in df.columns:
        ax1.plot(df.index, df["SPY_Value"], color=DIM, lw=1.2, ls="--", label="SPY")
        ax1.fill_between(df.index, df["Portfolio_Value"], df["SPY_Value"],
                         where=df["Portfolio_Value"] >= df["SPY_Value"],
                         alpha=0.10, color=GRN)
        ax1.fill_between(df.index, df["Portfolio_Value"], df["SPY_Value"],
                         where=df["Portfolio_Value"] < df["SPY_Value"],
                         alpha=0.08, color=RED)
    ax1.axhline(STARTING_CASH, color="#444", lw=0.6, ls=":")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(facecolor="#111", fontsize=7)

    # Panel 2: Weekly returns bar chart
    ax2 = _ax(0, 1, "Weekly Returns")
    wr  = df["Weekly_Return"] * 100
    ax2.bar(range(len(wr)), wr.values,
            color=[GRN if v >= 0 else RED for v in wr.values],
            alpha=0.8, edgecolor="none")
    ax2.axhline(0, color="#444", lw=0.6)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax2.set_xticks([])
    win_pct = float((wr > 0).mean() * 100)
    ax2.text(0.02, 0.96, f"Win rate: {win_pct:.1f}%", transform=ax2.transAxes,
             color=ACC, fontsize=9, va="top", fontweight="bold")

    # Panel 3: Drawdown
    ax3 = _ax(1, 0, "Drawdown from Peak")
    dd  = m["dd_ser"] * 100
    ax3.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.55)
    ax3.plot(dd.index, dd.values, color=RED, lw=0.8)
    ax3.invert_yaxis()
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    if len(dd) > 0:
        ax3.text(0.02, 0.04, f"Max DD: {dd.min():.1f}%", transform=ax3.transAxes,
                 color=RED, fontsize=9, va="bottom")

    # Panel 4: Key metrics text (no axhline transform - use ax.plot instead)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(CARD)
    ax4.axis("off")
    alpha_v = m["total"] - m["spy_tot"]
    if m["total"] > 0.10 and alpha_v > 0.02:
        verdict = "✅ PROFITABLE\n   BEATING SPY"
        vc = GRN
    elif m["total"] > 0.02:
        verdict = "⚠️ SMALL PROFIT\n   Watch 1 more quarter"
        vc = YEL
    elif m["total"] > -0.05:
        verdict = "⚠️ NEAR BREAKEVEN\n   Not ready for real $"
        vc = YEL
    else:
        verdict = "❌ LOSING\n   Do NOT go live"
        vc = RED

    ax4.text(0.5, 0.96, verdict, ha="center", va="top", color=vc,
             fontsize=12, fontweight="bold", transform=ax4.transAxes)
    stats = [
        ("Total Return",  f"{m['total']:>+.2%}"),
        ("Alpha vs SPY",  f"{alpha_v:>+.2%}"),
        ("SPY Return",    f"{m['spy_tot']:>+.2%}"),
        ("Ann. Return",   f"{m['ann_r']:>+.2%}"),
        ("Sharpe Ratio",  f"{m['sharpe']:>.3f}"),
        ("Max Drawdown",  f"{m['mdd']:>.2%}"),
        ("Weekly Win %",  f"{m['wr']:>.1%}"),
        ("Commission $",  f"${(len(trade_log) * COMMISSION):,.0f}"),
    ]
    for i, (label, value) in enumerate(stats):
        y   = 0.72 - i * 0.085
        neg = value.startswith("-") or (label == "Max Drawdown")
        col = RED if neg else GRN
        ax4.text(0.05, y, label, color=DIM, fontsize=9, transform=ax4.transAxes)
        ax4.text(0.95, y, value, color=col, fontsize=9, ha="right",
                 fontweight="bold", transform=ax4.transAxes)
        # Use ax4.plot instead of axhline so transform is not needed
        ax4.plot([0.02, 0.98], [y - 0.025, y - 0.025],
                 color="#1a1a2e", lw=0.4, transform=ax4.transAxes)

    config.PAPER_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.PAPER_PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Paper trading chart saved → {config.PAPER_PLOT_PATH}")


if __name__ == "__main__":
    print(f"Running paper trade simulation: {PAPER_START.date()} → {PAPER_END.date()}")
    print(f"Starting capital: ${STARTING_CASH:,.0f}  |  Top {TOP_N} stocks  |  ${COMMISSION:.0f}/trade commission\n")
    weekly, trade_log = run_simulation()
    metrics = compute_paper_metrics(weekly)
    print_summary(weekly, trade_log, metrics)
    plot_paper_dashboard(weekly, metrics, trade_log)
