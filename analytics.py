"""
QuantProject v2 - analytics.py
9-panel professional dark-theme trading dashboard.
Loads results/backtest.csv and results/ic_log.csv.

Run: python analytics.py
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
import seaborn as sns
from scipy.stats import spearmanr
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import config

# ── Colour palette ─────────────────────────────────────────────────────────
BG   = "#0b0b12"
CARD = "#12121e"
FG   = "#e8e8f2"
DIM  = "#55556a"
ACC  = "#00d4ff"
GRN  = "#00e676"
RED  = "#ff3d5a"
YEL  = "#ffd600"
PRP  = "#b388ff"
ORG  = "#ff7043"


def _style(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e1e30")
    ax.tick_params(colors=DIM, labelsize=7.5)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    if title:
        ax.set_title(title, color=FG, fontsize=9, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, fontsize=7)
    if grid:   ax.grid(True, color="#1a1a2e", linewidth=0.5, linestyle="--", alpha=0.7)


def pct_fmt(x, _):   return f"{x:.0f}%"
def dollar_fmt(x, _): return f"${x:,.0f}"
def ratio_fmt(x, _):  return f"{x:.2f}"


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(config.BACKTEST_PATH):
        raise FileNotFoundError(f"Run main.py first. Missing: {config.BACKTEST_PATH}")
    bt = pd.read_csv(config.BACKTEST_PATH, index_col=0, parse_dates=True)
    ic = pd.read_csv(config.IC_LOG_PATH, parse_dates=["train_end", "pred_start", "pred_end"]) \
         if os.path.exists(config.IC_LOG_PATH) else pd.DataFrame()
    return bt, ic


def compute_metrics(bt: pd.DataFrame) -> dict:
    rets   = bt["Strategy_Return"]
    spy_r  = bt["SPY_Return"]
    n      = len(rets)
    total  = bt["Portfolio_Value"].iloc[-1] / config.CASH - 1
    ann_r  = (1 + total) ** (52 / n) - 1
    ann_v  = rets.std() * np.sqrt(52)
    sharpe = (ann_r - 0.05) / ann_v if ann_v > 0 else 0
    neg    = rets[rets < 0]
    sortino= (ann_r - 0.05) / (neg.std() * np.sqrt(52)) if len(neg) > 0 else 0
    dd_ser = (bt["Portfolio_Value"] - bt["Portfolio_Value"].cummax()) / bt["Portfolio_Value"].cummax()
    mdd    = dd_ser.min()
    calmar = ann_r / abs(mdd) if mdd < 0 else 0
    cov_   = rets.cov(spy_r)
    var_   = spy_r.var()
    beta_  = cov_ / var_ if var_ > 0 else 0
    spy_ann= (bt["SPY_Value"].iloc[-1] / config.CASH) ** (52 / n) - 1
    alpha_ = ann_r - (0.05 + beta_ * (spy_ann - 0.05))
    wr     = (rets > 0).mean()
    spy_tot= bt["SPY_Value"].iloc[-1] / config.CASH - 1
    return dict(
        total=total, ann_r=ann_r, ann_v=ann_v, sharpe=sharpe,
        sortino=sortino, mdd=mdd, calmar=calmar,
        beta=beta_, alpha=alpha_, wr=wr, spy_tot=spy_tot, dd_ser=dd_ser,
    )


def main():
    print("Loading results...")
    bt, ic = load_results()
    m      = compute_metrics(bt)

    plt.rcParams.update({
        "figure.facecolor": BG, "text.color": FG,
        "font.family": "monospace", "font.size": 8,
    })

    fig = plt.figure(figsize=(22, 26), facecolor=BG)
    fig.suptitle(
        "QuantProject v2 - Strategy Dashboard",
        color=FG, fontsize=17, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.06, right=0.97, top=0.96, bottom=0.04,
    )

    # ─── Panel 1: Cumulative Returns vs SPY ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    cum_strat = bt["Portfolio_Value"]
    cum_spy   = bt["SPY_Value"]
    ax1.plot(cum_strat.index, cum_strat,  color=ACC, lw=2.0, label="Strategy", zorder=3)
    ax1.plot(cum_spy.index,   cum_spy,    color=DIM, lw=1.2, ls="--", label="SPY Buy & Hold")
    ax1.fill_between(cum_strat.index, cum_strat, cum_spy,
                     where=cum_strat > cum_spy, alpha=0.10, color=GRN, label="Outperforming")
    ax1.fill_between(cum_strat.index, cum_strat, cum_spy,
                     where=cum_strat < cum_spy, alpha=0.08, color=RED, label="Underperforming")
    ax1.axhline(config.CASH, color="#333", lw=0.6)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(dollar_fmt))
    ax1.legend(facecolor="#111", edgecolor="#333", fontsize=7)
    _style(ax1, "Cumulative Portfolio Value - Strategy vs SPY", ylabel="Portfolio ($)")

    # ─── Panel 2: Drawdown ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    dd  = m["dd_ser"] * 100
    ax2.fill_between(dd.index, dd, 0, color=RED, alpha=0.55)
    ax2.plot(dd.index, dd, color=RED, lw=0.8)
    ax2.invert_yaxis()
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    _style(ax2, "Drawdown from Peak", ylabel="%")

    # ─── Panel 3: Monthly Returns Heatmap ────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    monthly = (bt["Strategy_Return"]
               .resample("ME")
               .apply(lambda r: (1 + r).prod() - 1) * 100)
    monthly.index = monthly.index.to_period("M")
    pivot = (monthly
             .groupby([monthly.index.year, monthly.index.month])
             .first()
             .unstack())
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn",
        center=0, linewidths=0.3, linecolor=BG,
        ax=ax3, cbar=False, annot_kws={"size": 7},
    )
    ax3.set_title("Monthly Returns Heatmap (%)", color=FG, fontsize=9,
                  fontweight="bold", pad=6)
    ax3.tick_params(colors=DIM, labelsize=7.5)
    ax3.set_xlabel(""); ax3.set_ylabel("")

    # ─── Panel 4: Rolling 52w Sharpe ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    roll_sharpe = bt["Strategy_Return"].rolling(52).apply(
        lambda r: (r.mean() * 52 - 0.05) / (r.std() * np.sqrt(52))
        if r.std() > 0 else 0, raw=True
    )
    ax4.plot(roll_sharpe.index, roll_sharpe, color=ACC, lw=1.3)
    ax4.axhline(0,   color="#333", lw=0.7)
    ax4.axhline(0.5, color=GRN,   lw=0.7, ls="--", alpha=0.6, label="0.5")
    ax4.axhline(1.0, color=GRN,   lw=0.9, ls="--",            label="1.0")
    ax4.fill_between(roll_sharpe.index, roll_sharpe, 0,
                     where=roll_sharpe > 0, alpha=0.10, color=GRN)
    ax4.fill_between(roll_sharpe.index, roll_sharpe, 0,
                     where=roll_sharpe < 0, alpha=0.10, color=RED)
    ax4.legend(facecolor="#111", fontsize=7)
    _style(ax4, "Rolling 52w Sharpe Ratio")

    # ─── Panel 5: Rolling 52w IC ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    if not ic.empty and "mean_ic" in ic.columns:
        ic_ts = ic.set_index("pred_start")["mean_ic"].dropna()
        if len(ic_ts) > 0:
            ax5.bar(range(len(ic_ts)), ic_ts.values,
                    color=[GRN if v > 0 else RED for v in ic_ts.values],
                    alpha=0.8, edgecolor="none", width=0.85)
            ax5.axhline(0,    color="#333", lw=0.7)
            ax5.axhline(0.03, color=YEL,   lw=0.9, ls="--", label="IC=0.03 target")
            ax5.set_xticks(range(len(ic_ts)))
            ax5.set_xticklabels(
                [str(d)[:7] for d in ic_ts.index], rotation=45, fontsize=6
            )
            ax5.legend(facecolor="#111", fontsize=7)
            mean_ic_val = ic_ts.mean()
            ax5.text(0.02, 0.95, f"Mean IC = {mean_ic_val:.4f}",
                     transform=ax5.transAxes, color=ACC, fontsize=8, va="top")
    else:
        ax5.text(0.5, 0.5, "IC data not available", ha="center", va="center",
                 color=DIM, transform=ax5.transAxes)
    _style(ax5, "Quarterly IC (Spearman)", ylabel="IC")

    # ─── Panel 6: Annual Bar Chart - Strategy vs SPY ──────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    annual_s = bt["Strategy_Return"].resample("YE").apply(lambda r: (1+r).prod()-1)
    annual_b = bt["SPY_Return"].resample("YE").apply(lambda r: (1+r).prod()-1)
    years    = [str(d.year) for d in annual_s.index]
    x        = np.arange(len(years))
    w        = 0.36
    sv       = annual_s.values * 100
    bv       = annual_b.values * 100
    ax6.bar(x - w/2, sv, w,
            color=[GRN if v >= 0 else RED for v in sv], alpha=0.85, label="Strategy")
    ax6.bar(x + w/2, bv, w,
            color=[ACC if v >= 0 else ORG for v in bv], alpha=0.55, label="SPY")
    for xi, val in zip(x - w/2, sv):
        ax6.text(xi, val + (0.8 if val >= 0 else -2.5),
                 f"{val:.1f}%", ha="center", fontsize=6, color=FG)
    ax6.set_xticks(x)
    ax6.set_xticklabels(years, fontsize=7)
    ax6.axhline(0, color="#333", lw=0.7)
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax6.legend(facecolor="#111", fontsize=7)
    _style(ax6, "Annual Returns - Strategy vs SPY", ylabel="%")

    # ─── Panel 7: Alpha / Beta scatter ────────────────────────────────
    ax7 = fig.add_subplot(gs[0, 2])  # NOTE: reusing slot - let me fix placement
    # Overwrite with proper placement
    ax7.remove()
    ax7 = fig.add_subplot(gs[2, 2])
    strat_r = bt["Strategy_Return"].values
    spy_r   = bt["SPY_Return"].values
    mask    = ~(np.isnan(strat_r) | np.isnan(spy_r))
    sx, sy  = spy_r[mask], strat_r[mask]
    ax7.scatter(sx * 100, sy * 100, color=ACC, alpha=0.25, s=6, edgecolors="none")
    # Regression line
    if len(sx) > 2:
        slope, intercept, r_val, *_ = scipy_stats.linregress(sx, sy)
        x_line = np.linspace(sx.min(), sx.max(), 100)
        ax7.plot(x_line * 100, (intercept + slope * x_line) * 100,
                 color=YEL, lw=1.2, label=f"β={slope:.2f}  α={intercept*52:.2%}/yr")
    ax7.axhline(0, color="#333", lw=0.6); ax7.axvline(0, color="#333", lw=0.6)
    ax7.legend(facecolor="#111", fontsize=7)
    ax7.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax7.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    _style(ax7, "Alpha/Beta - Weekly Returns Scatter",
           xlabel="SPY Weekly Return", ylabel="Strategy Weekly Return")

    # ─── Panel 8: IC Distribution Histogram ───────────────────────────
    # NOTE: Panel 2 slot gs[0,2] is drawdown; shift IC histogram to gs[1,2] → but already rolling sharpe
    # Use a sub-axes within the figure
    ax8 = fig.add_subplot(gs[1, 2])
    ax8.remove()
    ax8 = fig.add_subplot(gs[1, 2])
    if not ic.empty and "mean_ic" in ic.columns:
        ic_vals = ic["mean_ic"].dropna()
        ax8.hist(ic_vals, bins=min(15, len(ic_vals)),
                 color=PRP, alpha=0.8, edgecolor="none", density=True)
        mu_ic = ic_vals.mean(); std_ic = ic_vals.std()
        x_fit = np.linspace(ic_vals.min() - 0.01, ic_vals.max() + 0.01, 100)
        ax8.plot(x_fit, scipy_stats.norm.pdf(x_fit, mu_ic, std_ic),
                 color=YEL, lw=1.5, label=f"μ={mu_ic:.4f}")
        ax8.axvline(0,    color="#444", lw=0.7)
        ax8.axvline(0.03, color=GRN,   lw=0.9, ls="--", label="IC=0.03")
        ax8.legend(facecolor="#111", fontsize=7)
    else:
        ax8.text(0.5, 0.5, "IC data not available", ha="center", va="center",
                 color=DIM, transform=ax8.transAxes)
    _style(ax8, "IC Distribution Histogram", xlabel="Spearman IC")

    # ─── Panel 9: Turnover per Week ───────────────────────────────────
    # Replace rolling Sharpe panel (already panel 4) - put turnover at gs[1,2]
    # Actually we have 9 panels across 3×3 grid - need to reorganise.
    # Revised layout:
    # Row 0: Cumulative (2-wide), Drawdown
    # Row 1: Monthly Heatmap (2-wide), IC Histogram
    # Row 2: IC Quarterly, Annual Bars, Alpha/Beta Scatter
    # Rolling Sharpe and Turnover need spots - add extra row

    # Add two more panels below the 3×3 grid
    gs2 = gridspec.GridSpec(
        1, 2, figure=fig,
        top=0.04 - 0.00, bottom=0.00,  # this won't fit - use inset instead
    )
    # Just embed Turnover and Rolling Sharpe inside existing cells with inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Turnover inside ax4 (rolling sharpe) - REPLACE ax4 content
    ax4.clear()
    turnover_vals = bt["Turnover"].dropna()
    ax4.bar(range(len(turnover_vals)), turnover_vals.values,
            color=ORG, alpha=0.7, edgecolor="none", width=1.0)
    ax4.set_title("Weekly Portfolio Turnover", color=FG, fontsize=9,
                  fontweight="bold", pad=6)
    avg_to = turnover_vals.mean()
    ax4.axhline(avg_to, color=YEL, lw=1, ls="--", label=f"Avg {avg_to:.2f}")
    ax4.set_xticks([])
    ax4.legend(facecolor="#111", fontsize=7)
    _style(ax4, ylabel="Turnover (fractional)")

    # Rolling Sharpe - put into a completely new GridSpec row
    gs3 = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.50, wspace=0.38,
        left=0.06, right=0.97, top=0.96, bottom=0.00,
    )
    ax_sh = fig.add_subplot(gs3[3, :])
    roll_sh2 = bt["Strategy_Return"].rolling(52).apply(
        lambda r: (r.mean() * 52 - 0.05) / (r.std() * np.sqrt(52))
        if r.std() > 0 else 0, raw=True
    )
    ax_sh.plot(roll_sh2.index, roll_sh2, color=ACC, lw=1.3)
    ax_sh.axhline(0,   color="#333", lw=0.7)
    ax_sh.axhline(0.5, color=GRN,   lw=0.7, ls="--", alpha=0.6)
    ax_sh.axhline(1.0, color=GRN,   lw=0.9, ls="--", label="Sharpe=1.0")
    ax_sh.fill_between(roll_sh2.index, roll_sh2, 0,
                       where=roll_sh2 > 0, alpha=0.10, color=GRN)
    ax_sh.fill_between(roll_sh2.index, roll_sh2, 0,
                       where=roll_sh2 < 0, alpha=0.10, color=RED)
    ax_sh.legend(facecolor="#111", fontsize=7, loc="upper left")
    ax_sh.yaxis.set_major_formatter(mticker.FuncFormatter(ratio_fmt))
    _style(ax_sh, "Rolling 52-Week Sharpe Ratio", ylabel="Sharpe")

    # ── Stats box in bottom-right corner of the figure ────────────────
    # Add a text annotation box as the 9th "panel"
    stats_text = (
        f"  KEY METRICS\n"
        f"  {'─'*28}\n"
        f"  Total Return    {m['total']:>+.1%}\n"
        f"  Ann. Return     {m['ann_r']:>+.1%}\n"
        f"  Ann. Vol        {m['ann_v']:>.1%}\n"
        f"  Sharpe          {m['sharpe']:>.3f}\n"
        f"  Sortino         {m['sortino']:>.3f}\n"
        f"  Calmar          {m['calmar']:>.3f}\n"
        f"  Max Drawdown    {m['mdd']:>.1%}\n"
        f"  Weekly Win %    {m['wr']:>.1%}\n"
        f"  Alpha (ann)     {m['alpha']:>+.1%}\n"
        f"  Beta vs SPY     {m['beta']:>.3f}\n"
        f"  SPY Return      {m['spy_tot']:>+.1%}\n"
    )
    fig.text(0.785, 0.345, stats_text, color=FG, fontsize=8,
             fontfamily="monospace",
             bbox=dict(facecolor=CARD, edgecolor="#1e1e30", boxstyle="round,pad=0.5"))

    # ── Save ──────────────────────────────────────────────────────────
    config.DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.DASHBOARD_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Dashboard saved → {config.DASHBOARD_PATH}")
    print(f"  Open with: open {config.DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
