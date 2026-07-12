# The Mathematics of QuantProject v2

A complete mathematical specification of a market-neutral, walk-forward validated
long/short equity strategy on the S&P 500. Every formula below is implemented
verbatim in `main.py` and parameterised in `config.py`; nothing here is
idealised after the fact.

**Author:** Nedim Yıldırım

---

## Contents

1. [Notation and setup](#1-notation-and-setup)
2. [The 30 features](#2-the-30-features)
3. [Cross-sectional rank normalisation](#3-cross-sectional-rank-normalisation)
4. [The prediction target](#4-the-prediction-target)
5. [The three models and the ensemble](#5-the-three-models-and-the-ensemble)
6. [Walk-forward cross-validation](#6-walk-forward-cross-validation)
7. [The Information Coefficient](#7-the-information-coefficient)
8. [Portfolio construction](#8-portfolio-construction)
9. [Backtest accounting](#9-backtest-accounting)
10. [Performance metrics](#10-performance-metrics)
11. [Known biases, stated honestly](#11-known-biases-stated-honestly)

---

## 1. Notation and setup

The universe is $N = 459$ S&P 500 constituents observed weekly (Friday close)
from 2017 to 2025. For stock $i$ at week $t$:

- $P_{i,t}$ is the closing price, $H_{i,t}$ the weekly high, $L_{i,t}$ the weekly low, $V_{i,t}$ the share volume
- $S_t$ is the SPY (S&P 500 ETF) closing price, used as the market proxy

Simple and log returns:

$$
r_{i,t} = \frac{P_{i,t}}{P_{i,t-1}} - 1,
\qquad
\tilde r_{i,t} = \ln\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right),
\qquad
\tilde r_{m,t} = \ln\!\left(\frac{S_t}{S_{t-1}}\right)
$$

Log returns are used inside rolling statistical estimators (volatility, beta,
skewness, correlation) because they are time-additive; simple returns are used
for P&L, which compounds multiplicatively.

All rolling windows are **strictly backward-looking**: a feature at week $t$
uses data up to and including the close of week $t$, never after.

---

## 2. The 30 features

Each week, every stock is described by a vector
$x_{i,t} \in \mathbb{R}^{30}$: 25 technical features and 5 fundamental
features. Grouped by the market effect they try to capture:

### 2.1 Momentum (features 1 to 4)

Classical price momentum over four horizons, $k \in \{4, 13, 26, 52\}$ weeks:

$$
\text{mom}_{i,t}^{(k)} = \frac{P_{i,t}}{P_{i,t-k}} - 1
$$

Motivated by the empirical persistence of relative returns (Jegadeesh and
Titman, 1993): recent relative winners tend to keep winning over 3 to 12 month
horizons.

### 2.2 RSI (feature 5)

Wilder's Relative Strength Index over 14 weekly bars. With price change
$\Delta_{i,t} = P_{i,t} - P_{i,t-1}$, split into gains and losses:

$$
G_{i,t} = \max(\Delta_{i,t}, 0), \qquad D_{i,t} = \max(-\Delta_{i,t}, 0)
$$

Both are smoothed with an exponentially weighted moving average with
$\alpha = \tfrac{1}{14}$ (Wilder smoothing):

$$
\bar G_{i,t} = \alpha\, G_{i,t} + (1-\alpha)\, \bar G_{i,t-1},
\qquad
\bar D_{i,t} = \alpha\, D_{i,t} + (1-\alpha)\, \bar D_{i,t-1}
$$

$$
\text{RSI}_{i,t} = 100 - \frac{100}{1 + \bar G_{i,t} / \bar D_{i,t}}
$$

RSI near 100 means recent gains dominate (overbought); near 0, oversold.

### 2.3 Volume z-score (feature 6)

How unusual this week's volume is relative to its own recent 4-week history:

$$
z^{V}_{i,t} = \frac{V_{i,t} - \mu_4(V_{i,t})}{\sigma_4(V_{i,t})}
$$

where $\mu_k(\cdot)$ and $\sigma_k(\cdot)$ denote the rolling $k$-week mean and
standard deviation. Abnormal volume often accompanies information arrival.

### 2.4 52-week high and low anchors (features 7 and 19)

$$
\text{high52}_{i,t} = \frac{P_{i,t}}{\max_{s \in [t-51,\, t]} P_{i,s}},
\qquad
\text{low52}_{i,t} = \frac{P_{i,t}}{\min_{s \in [t-51,\, t]} P_{i,s}}
$$

The 52-week-high effect (George and Hwang, 2004): stocks near their yearly
high tend to continue outperforming, an anchoring bias in how investors
process news.

### 2.5 Idiosyncratic volatility (feature 8)

The volatility a stock has **beyond** what its market exposure explains.
First estimate the rolling 52-week beta by the least-squares formula:

$$
\beta_{i,t} = \frac{\operatorname{Cov}_{52}(\tilde r_{i}, \tilde r_{m})}{\operatorname{Var}_{52}(\tilde r_{m})}
$$

Then decompose total variance into systematic and idiosyncratic parts
(from the single-factor model $\tilde r_i = \beta_i \tilde r_m + \varepsilon_i$,
where the independence of $\varepsilon$ and $\tilde r_m$ gives
$\operatorname{Var}(\tilde r_i) = \beta_i^2 \operatorname{Var}(\tilde r_m) + \operatorname{Var}(\varepsilon_i)$):

$$
\sigma^2_{\varepsilon,i,t} = \max\!\Big( \operatorname{Var}_{52}(\tilde r_i) - \beta_{i,t}^2 \operatorname{Var}_{52}(\tilde r_m),\; 0 \Big)
$$

$$
\text{idiovol}_{i,t} = \sqrt{52\, \sigma^2_{\varepsilon,i,t}}
$$

The $\sqrt{52}$ annualises weekly volatility (variance scales linearly in
time under independent increments, so volatility scales with $\sqrt{\text{periods}}$).
This is the "low-volatility anomaly" factor (Ang, Hodrick, Xing, Zhang, 2006):
high idiosyncratic volatility has historically predicted **lower** future
returns.

### 2.6 Relative strength vs the market (features 9 and 10)

Momentum net of the market's own momentum, for $k \in \{4, 13\}$:

$$
\text{relstr}_{i,t}^{(k)} = \left(\frac{P_{i,t}}{P_{i,t-k}} - 1\right) - \left(\frac{S_t}{S_{t-k}} - 1\right)
$$

This isolates stock-specific strength from a rising or falling tide.

### 2.7 Short-term reversal (feature 11)

$$
\text{rev}_{i,t} = -\left(\frac{P_{i,t}}{P_{i,t-1}} - 1\right)
$$

The sign flip encodes the one-week reversal effect (Jegadeesh, 1990): the
previous week's extreme movers tend to partially revert, likely from
liquidity-driven price pressure.

### 2.8 Realised volatility (features 12 and 13)

Annualised rolling volatility of log returns for $k \in \{4, 13\}$:

$$
\sigma_{i,t}^{(k)} = \sqrt{52}\; \sigma_k(\tilde r_{i})
$$

### 2.9 MACD histogram (feature 14)

With the exponential moving average defined recursively as
$\text{EMA}^{(n)}_t = \lambda P_t + (1 - \lambda)\, \text{EMA}^{(n)}_{t-1}$,
$\lambda = \tfrac{2}{n+1}$, using spans adapted to weekly bars (4/9/4 instead
of the daily-chart 12/26/9):

$$
\text{MACD}_{i,t} = \text{EMA}^{(4)}_{i,t} - \text{EMA}^{(9)}_{i,t}
$$

$$
\text{hist}_{i,t} = \text{MACD}_{i,t} - \text{EMA-of-MACD}^{(4)}_{i,t}
$$

The histogram measures the **acceleration** of a trend: it is positive when
the short-term trend is pulling away from the medium-term trend.

### 2.10 Bollinger Band position (feature 15)

Where the price sits inside a 2-standard-deviation envelope around its
20-week mean:

$$
\text{bb}_{i,t} = \frac{P_{i,t} - \big(\mu_{20}(P_i) - 2\sigma_{20}(P_i)\big)}{4\,\sigma_{20}(P_i)}
$$

0 at the lower band, 1 at the upper band; values outside $[0,1]$ mark
statistically stretched prices.

### 2.11 Average True Range as a percentage of price (feature 16)

True range, the largest of the intra-week range and the gaps versus last
week's close:

$$
\text{TR}_{i,t} = \max\!\big( H_{i,t} - L_{i,t},\; |H_{i,t} - P_{i,t-1}|,\; |L_{i,t} - P_{i,t-1}| \big)
$$

$$
\text{ATR\%}_{i,t} = 100 \cdot \frac{\mu_4(\text{TR}_{i})}{P_{i,t}}
$$

A price-scale-free measure of trading range width.

### 2.12 Moving-average crossovers (features 17, 18, 22)

With $\text{MA}^{(k)}_{i,t} = \mu_k(P_i)$:

$$
\text{cross}^{(4,13)}_{i,t} = \frac{\text{MA}^{(4)}_{i,t}}{\text{MA}^{(13)}_{i,t}} - 1,
\qquad
\text{cross}^{(13,26)}_{i,t} = \frac{\text{MA}^{(13)}_{i,t}}{\text{MA}^{(26)}_{i,t}} - 1,
\qquad
\text{pma}^{(26)}_{i,t} = \frac{P_{i,t}}{\text{MA}^{(26)}_{i,t}}
$$

Positive crossover values indicate an established uptrend at that pair of
horizons.

### 2.13 Rolling beta (feature 20)

The 52-week beta from section 2.5, lagged one week
($\beta_{i,t-1}$) so the feature never contains contemporaneous information.

### 2.14 Volume trend (feature 21)

$$
\text{voltrend}_{i,t} = \frac{\mu_4(V_i)}{\mu_{13}(V_i)}
$$

Ratio above 1 means volume is building relative to its own quarter.

### 2.15 Return skewness (feature 23)

Rolling 13-week sample skewness of log returns:

$$
\text{skew}_{i,t} = \frac{\tfrac{1}{13}\sum_{s=t-12}^{t} \big(\tilde r_{i,s} - \bar{\tilde r}_i\big)^3}{\Big(\tfrac{1}{13}\sum_{s=t-12}^{t} \big(\tilde r_{i,s} - \bar{\tilde r}_i\big)^2\Big)^{3/2}}
$$

Investors systematically overpay for lottery-like positive skew, which
depresses the subsequent returns of positively skewed stocks.

### 2.16 Rolling correlation with the market (feature 24)

$$
\rho_{i,t} = \operatorname{Corr}_{13}(\tilde r_i, \tilde r_m)
= \frac{\operatorname{Cov}_{13}(\tilde r_i, \tilde r_m)}{\sigma_{13}(\tilde r_i)\, \sigma_{13}(\tilde r_m)}
$$

### 2.17 Upside/downside volatility ratio (feature 25)

Split each return into its positive and negative part,
$\tilde r^{+} = \max(\tilde r, 0)$ and $\tilde r^{-} = \min(\tilde r, 0)$, then:

$$
\text{updown}_{i,t} = \frac{\sigma_{13}(\tilde r^{+}_i)}{\sigma_{13}(\tilde r^{-}_i)}
$$

A ratio above 1 means the stock's variability comes disproportionately from
up-moves, an asymmetry plain volatility cannot see.

### 2.18 Fundamentals (features 26 to 30)

Five accounting ratios per company: trailing P/E, price-to-book, net profit
margin, year-over-year revenue growth, and debt-to-equity. Missing values are
filled with the cross-sectional median of each metric. These are **static**
current values broadcast across all dates (see section 11 for the honest
caveat this creates).

---

## 3. Cross-sectional rank normalisation

Raw features live on wildly different scales (RSI in $[0,100]$, momentum in
$[-1, \infty)$, P/E anywhere). Worse, their absolute levels drift across time
(volatility in 2020 dwarfs 2017). Both problems are removed by replacing every
feature value with its **percentile rank among all stocks that same week**:

$$
\hat x^{(j)}_{i,t} = \frac{\operatorname{rank}\big(x^{(j)}_{i,t}\big)}{N_t} \in (0, 1]
$$

where the rank is taken within week $t$ across the $N_t$ stocks with valid
data, separately for each feature $j$. After this transform every feature has
an identical (uniform) marginal distribution every single week, so the model
can only learn from **relative ordering**, which is exactly the quantity a
cross-sectional long/short portfolio trades on.

---

## 4. The prediction target

The target is the **4-week forward return**, converted to the same
cross-sectional rank scale:

$$
y_{i,t} = \operatorname{rank}_{\,i \in \text{week } t}\left( \frac{P_{i,t+4}}{P_{i,t}} - 1 \right) \Big/ N_t \in (0, 1]
$$

Two deliberate choices:

- **4 weeks, not 1**: a longer horizon averages out microstructure noise, so the signal-to-noise ratio of the target is higher.
- **Rank, not raw return**: the portfolio only needs the ordering. Predicting ranks makes the regression robust to the heavy tails of raw returns and matches the loss the strategy actually cares about.

The prediction problem is then: learn $f: [0,1]^{30} \to [0,1]$ with
$y_{i,t} \approx f(\hat x_{i,t})$.

---

## 5. The three models and the ensemble

Three models of deliberately different flavours are trained on identical data
each fold, then averaged. Diversity of functional form is the point: their
errors are imperfectly correlated, and averaging reduces error variance.

### 5.1 Ridge regression (linear baseline)

Ordinary least squares with an $L_2$ penalty. On standardised features
(z-scored using training-set statistics), Ridge solves:

$$
\hat\theta = \arg\min_{\theta}\; \sum_{(i,t) \in \text{train}} \big( y_{i,t} - \theta_0 - \theta^\top \hat x_{i,t} \big)^2 + \lambda \lVert \theta \rVert_2^2,
\qquad \lambda = 1
$$

The penalty term shrinks coefficients toward zero, trading a little bias for
a large variance reduction, which matters when features are correlated (and
momentum features certainly are). It has the closed form
$\hat\theta = (X^\top X + \lambda I)^{-1} X^\top y$.

### 5.2 Gradient-boosted trees: LightGBM and XGBoost

Both build an additive model of $M = 300$ regression trees:

$$
f_M(x) = \sum_{m=1}^{M} \nu\, h_m(x), \qquad \nu = 0.02
$$

fit by **gradient boosting**: at stage $m$, the tree $h_m$ is fit to the
negative gradient of the loss at the current prediction, which for squared
error is simply the residual:

$$
h_m \approx \arg\min_h \sum_{(i,t)} \Big( \big(y_{i,t} - f_{m-1}(\hat x_{i,t})\big) - h(\hat x_{i,t}) \Big)^2
$$

Each stage is a small correction ($\nu = 0.02$ shrinks every tree's
contribution) to the errors of everything built so far, so the ensemble
descends the loss surface in function space. Trees capture non-linear
effects and feature interactions (for example "momentum only works when
volatility is low") that a linear model cannot express.

Overfitting is controlled by: at most 15 leaves per tree (LightGBM) or depth
4 (XGBoost), a minimum of 50 samples per leaf, 70% row subsampling and 70%
feature subsampling per tree, plus $L_1$ ($\alpha = 0.1$) and $L_2$
($\lambda = 1.0$) penalties on leaf values. LightGBM and XGBoost implement the
same idea with different tree-growth policies (leaf-wise vs level-wise),
which is exactly the kind of decorrelated diversity an ensemble wants.

### 5.3 The ensemble

Raw scores are averaged with equal weights and then re-ranked
cross-sectionally so the final score is scale-free:

$$
s_{i,t} = \operatorname{rank}_{\,i \in \text{week } t}\!\left( \frac{f^{\text{LGB}}(\hat x_{i,t}) + f^{\text{XGB}}(\hat x_{i,t}) + f^{\text{Ridge}}(\hat x_{i,t})}{3} \right) \Big/ N_t
$$

Measured result: the ensemble's mean out-of-sample IC (0.0506) exceeds each
individual model's (LightGBM 0.0491, XGBoost 0.0498, Ridge 0.0387).

---

## 6. Walk-forward cross-validation

The single most important design decision in the project. Standard
cross-validation shuffles data randomly, which leaks future information into
training and produces meaninglessly optimistic results on time series.
Instead, training expands strictly forward in time:

$$
\text{fold } q: \quad \text{train on } \{t < T_q\}, \quad \text{predict } \{T_q \le t < T_{q+1}\}
$$

where $T_1, T_2, \dots, T_{27}$ are quarter-start dates from April 2018 to
December 2024 (after a minimum 52-week warmup). All three models are
**re-trained from scratch 27 times**, once per fold, and every backtested
prediction is genuinely out-of-sample: made by a model that has never seen
that quarter, exactly as live trading would have experienced it.

---

## 7. The Information Coefficient

The IC is the quality metric for the prediction step, before any portfolio
logic. For each week, it is the **Spearman rank correlation** between
predicted scores and realised forward-return ranks:

$$
\text{IC}_t = 1 - \frac{6 \sum_{i=1}^{N_t} d_{i}^2}{N_t (N_t^2 - 1)},
\qquad d_i = \operatorname{rank}(s_{i,t}) - \operatorname{rank}(y_{i,t})
$$

Spearman rather than Pearson because only ordering matters to the portfolio,
and rank correlation is immune to outliers and any monotone rescaling of the
scores.

Measured across 27 out-of-sample folds: **mean IC = 0.051, positive in 85.2%
of folds**. An IC of 0.05 sounds tiny, but the "fundamental law of active
management" (Grinold) says the information ratio scales like
$\text{IR} \approx \text{IC} \cdot \sqrt{\text{breadth}}$; with roughly 459
stocks ranked 52 times a year, a small but *consistent* IC is exactly what a
cross-sectional strategy needs. Consistency, not magnitude, is the evidence
of real signal.

---

## 8. Portfolio construction

The score vector $s_t$ becomes a weight vector $w_t$ through five
transformations, applied in order.

### 8.1 Soft power-law weights

Centre the score ranks and apply a signed power transform with
$\alpha = 3$:

$$
u_{i,t} = s_{i,t} - \tfrac{1}{2},
\qquad
w^{\text{raw}}_{i,t} = \operatorname{sign}(u_{i,t}) \cdot |u_{i,t}|^{3}
$$

then normalise to unit gross exposure:

$$
w_{i,t} = \frac{w^{\text{raw}}_{i,t}}{\sum_j |w^{\text{raw}}_{j,t}|}
\quad\Longrightarrow\quad
\lVert w_t \rVert_1 = 1
$$

The cubic transform concentrates capital in the extreme ranks (the strongest
convictions) while keeping middling stocks at small weights. Because the map
from rank to weight is **continuous**, a stock whose rank drifts from 0.93 to
0.89 sees only a small weight change, whereas a hard "top-25" rule would eject
it entirely. Measured effect: average turnover fell from 96% to 29% per week.
Symmetry of the transform around zero makes the portfolio dollar-neutral by
construction: $\sum_i w_{i,t} \approx 0$.

### 8.2 Sector neutralisation

For each GICS sector $\mathcal{S}$, subtract the sector's net weight equally
across its members:

$$
w_{i,t} \leftarrow w_{i,t} - \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} w_{j,t}
\qquad \text{for } i \in \mathcal{S}
$$

after which $\sum_{i \in \mathcal{S}} w_{i,t} = 0$ for every sector: the
portfolio cannot profit from "tech beats energy", only from picking the right
stocks *within* each sector. Gross exposure is rescaled to its
pre-adjustment level afterwards.

### 8.3 Regime filter

Define the bull-market indicator from SPY's 40-week moving average
(the weekly-bar equivalent of the 200-day MA):

$$
\mathbb{1}^{\text{bull}}_t = \mathbb{1}\!\left[ S_t > \tfrac{1}{40} \textstyle\sum_{s=t-39}^{t} S_s \right]
$$

If $\mathbb{1}^{\text{bull}}_t = 0$ the entire weight vector is set to zero:
the strategy holds cash through bear markets. Both legs are cut, not just the
long leg, because a live short book during a bear-market rally is a net-short
bet the model never intended to make.

### 8.4 Volatility targeting

Let $\hat\sigma_t = \sqrt{52} \cdot \sigma_{52}(r^p)$ be the annualised
realised volatility of the strategy's own trailing 52 weekly returns. Weights
are scaled toward a constant 12% risk budget:

$$
w_t \leftarrow w_t \cdot \min\!\left( \frac{0.12}{\hat\sigma_t},\; 1.5 \right)
$$

When markets are quiet the book levers up (capped at 1.5x); when they are
turbulent it shrinks. The portfolio's risk, rather than its notional size, is
held roughly constant, which is what makes its Sharpe ratio interpretable
across regimes.

### 8.5 Stop-loss and position cap

Two per-position overrides applied every week:

$$
\text{if } -\operatorname{sign}(w_{i,t-1}) \cdot r_{i,t} > 0.08
\;\Longrightarrow\; w_{i,t} = 0
\qquad\text{(stop-loss)}
$$

$$
w_{i,t} \leftarrow \operatorname{clip}(w_{i,t},\, -0.05,\, +0.05)
\qquad\text{(position cap)}
$$

The stop-loss expression covers both directions: a long that fell 8% and a
short that rose 8% both register a positive position loss and are exited the
following week regardless of the model's opinion.

Rebalancing to fresh model weights happens only every 4 weeks; between
rebalances the previous weights are held (with stop-losses still enforced
weekly), cutting transaction costs roughly fourfold.

---

## 9. Backtest accounting

The weekly loop applies events in an order that avoids the classic
look-ahead accounting error:

**Step 1: earn P&L on the weights held coming into the week.**

$$
r^{p}_t = \sum_i w_{i,t-1} \, r_{i,t}
$$

**Step 2: only then rebalance to the new weights**, paying transaction costs
proportional to one-way turnover at $c = 5$ basis points per side:

$$
\text{turnover}_t = \tfrac{1}{2} \sum_i \big| w_{i,t} - w_{i,t-1} \big|,
\qquad
\text{TC}_t = c \cdot \text{turnover}_t
$$

**Step 3: compound.**

$$
\Pi_t = \Pi_{t-1} \cdot (1 + r^p_t) \cdot (1 - \text{TC}_t), \qquad \Pi_0 = \$10{,}000
$$

Computing P&L with the *new* weights instead of the old ones (the v1 bug of
this project) silently grants the portfolio returns it never held; fixing
that single line moved the backtest from an absurd result to an honest one.

One refinement: scores are smoothed through time with an exponential moving
average, $\bar s_{i,t} = 0.4\, s_{i,t} + 0.6\, \bar s_{i,t-1}$, before weights
are built, further damping noise-driven churn.

---

## 10. Performance metrics

With weekly strategy returns $r^p_t$, $T = 349$ weeks, and annualisation
factor 52:

**Annualised return (CAGR)**

$$
R_{\text{ann}} = \left( \frac{\Pi_T}{\Pi_0} \right)^{52/T} - 1
$$

**Sharpe ratio** (risk-free rate taken as zero over the sample):

$$
\text{Sharpe} = \frac{52 \cdot \bar r^p}{\sqrt{52} \cdot \sigma(r^p)} = \sqrt{52}\, \frac{\bar r^p}{\sigma(r^p)}
$$

**Sortino ratio**: identical, but the denominator uses only downside
deviations, $\sigma_- = \sqrt{\tfrac{1}{T} \sum_t \min(r^p_t, 0)^2}$, so
upside volatility is not punished.

**Maximum drawdown**: the worst peak-to-trough loss,

$$
\text{MDD} = \min_{t} \left( \frac{\Pi_t}{\max_{s \le t} \Pi_s} - 1 \right)
$$

**Calmar ratio**: $R_{\text{ann}} / |\text{MDD}|$, return per unit of worst
historical pain.

**Alpha and beta** come from the CAPM time-series regression against SPY:

$$
r^p_t = \alpha + \beta\, r^{m}_t + \varepsilon_t
$$

estimated by OLS: $\hat\beta = \operatorname{Cov}(r^p, r^m) / \operatorname{Var}(r^m)$
and $\hat\alpha = \bar r^p - \hat\beta\, \bar r^m$ (annualised by
multiplying by 52).

### Measured results (6 Apr 2018 to 6 Dec 2024, all out-of-sample)

| Metric | Strategy | SPY buy and hold |
|---|---|---|
| Annualised return | +8.2% | +15.3% |
| Annualised volatility | 10.7% | 18.1% |
| Sharpe ratio | 0.30 | 0.55 |
| Maximum drawdown | **-21.7%** | -32.2% |
| Alpha (annualised) | **+2.9%** | 0 by definition |
| Beta vs SPY | **0.03** | 1.00 |

The right reading: with $\beta = 0.03$ the strategy is statistically almost
orthogonal to the market. Its +2.9% annual alpha is stock-selection skill,
not disguised market exposure, and it delivered its best relative years
precisely when the market fell (2018: +12.9% over SPY; 2022: +6.2% over SPY).
It is a diversifier, not an index-beater, and the mathematics above is
honest about which one it is.

---

## 11. Known biases, stated honestly

1. **Survivorship bias.** The data source (yfinance) only lists *current*
   S&P 500 members. Companies delisted or removed during 2018 to 2024 are
   invisible, and those are disproportionately the losers. Backtest returns
   are therefore inflated relative to what live trading would have achieved.
2. **Fundamental features are not point-in-time.** The five accounting ratios
   are today's values broadcast across all historical dates, a mild lookahead
   in the backtest (a company's *current* P/E was not knowable in 2019).
   The 25 technical features are fully point-in-time; the fundamental block
   is most valid for live signal generation, where current values are exactly
   what one wants.
3. **Transaction cost model is simple.** A flat 5 bps per side approximates
   costs for liquid large caps but ignores market impact, borrow fees on the
   short leg, and overnight financing.

These are disclosed rather than patched over because the honest error bars
are part of the result: the defensible claim is not "+8.2% a year" but
"a consistently positive out-of-sample rank correlation (IC 0.051, positive
in 85% of folds) that survives realistic costs and strict temporal
validation, in a portfolio nearly uncorrelated with the market."

---

*All formulas correspond line-for-line to `main.py` (features: section 2;
models: section 3; portfolio and accounting: section 4) with parameters in
`config.py`. Results tables are generated by the pipeline into `RESULTS.md`.*
