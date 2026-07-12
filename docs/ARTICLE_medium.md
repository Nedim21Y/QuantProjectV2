# I Built a Market-Neutral Trading System at 17. Here Is the Mathematics That Makes It Honest.

*How 30 features, three machine learning models, and 27 walk-forward validation folds produce a small but real predictive edge on S&P 500 stocks, and why the most important number in the project is not the return.*

---

Most amateur trading strategies die the same death: they look brilliant on historical data and lose money the moment they go live. The cause is almost never bad ideas. It is bad mathematics, specifically the quiet ways future information leaks into a backtest and flatters it.

I spent this year building a systematic long/short equity strategy on the S&P 500, and I made one rule for myself from the start: every design decision would be justified mathematically, and every result would be reported with its flaws attached. This article is the full story of that mathematics. The complete code and formal write-up are on GitHub: https://github.com/Nedim21Y/QuantProjectV2.

## The idea: trade the ordering, not the market

The strategy never bets on whether the market goes up. Every week it ranks 459 S&P 500 stocks from most to least attractive, buys the top of the ranking, and short sells the bottom, in equal dollar amounts. If the longs beat the shorts, it profits, whether the market rose or crashed.

This one choice reshapes the whole problem. I do not need to predict returns. I only need to predict *ordering*, which is statistically a much easier target. It also means the right measure of skill is not profit but rank correlation, a point I will come back to, because it is the heart of the project.

## Thirty features, one crucial transform

Each week, every stock is described by 30 numbers. Twenty-five are technical, built from price and volume history: momentum over four horizons (4, 13, 26 and 52 weeks), Wilder's RSI, MACD histogram, Bollinger Band position, average true range, distance from the 52-week high and low, moving-average crossovers, realised volatility, return skewness, rolling beta and correlation against SPY, an upside-to-downside volatility ratio, a one-week reversal term, and idiosyncratic volatility, which I estimate by regressing each stock's returns on the market and taking the volatility the regression cannot explain: idio variance = total variance minus beta squared times market variance. Five more are fundamentals: P/E, price-to-book, profit margin, revenue growth, and debt-to-equity.

None of these is original. They are the classic anomalies of the asset-pricing literature (Jegadeesh and Titman's momentum, George and Hwang's 52-week-high effect, Ang's low-volatility anomaly). The engineering that matters is what happens next.

Every feature, every week, is replaced by its percentile rank across all stocks that week. Apple's momentum is no longer 0.14; it is "83rd percentile this week". After this transform, every feature has an identical uniform distribution every single week. Level effects, regime shifts in volatility, units, outliers: all gone. The model physically cannot learn anything except relative ordering, which is the only thing a cross-sectional portfolio can trade.

The prediction target gets the same treatment: the 4-week forward return, rank-normalised within its week. Ranks in, ranks out.

## Three models that disagree productively

I train three models per period: LightGBM and XGBoost, two gradient-boosted tree ensembles that build 300 small regression trees where each tree corrects the residual errors of everything before it, and Ridge regression, a linear model with an L2 penalty that shrinks coefficients toward zero to survive the heavy correlation between momentum features.

Why three? Because their errors are imperfectly correlated. Trees capture interactions a line cannot ("momentum works better when volatility is low"); the linear model is stable where trees overfit. The final score is the equal-weight average of the three predictions, re-ranked. The ensemble's out-of-sample rank correlation (0.0506) beats every individual model (0.0491, 0.0498, 0.0387). Textbook variance reduction, visible in real data.

## The part that makes it honest: walk-forward validation

Here is where most backtests cheat, usually by accident. If you train a model on 2018 to 2024 data and evaluate it on those same years, it has effectively seen the answers. Random cross-validation does not fix this for time series; it shuffles future weeks into the training set.

So the system retrains from scratch 27 times, once per quarter from April 2018 to December 2024. Each fold trains only on data strictly before its quarter, then predicts that quarter blind. Every single backtested prediction is out-of-sample, made by a model that had never seen the period it was scoring, exactly as live trading would experience it.

The metric I track is the Information Coefficient: the Spearman rank correlation between predicted and realised return rankings, per week. Across 27 folds the mean IC is 0.051, and it is positive in 85% of folds.

An IC of 0.05 sounds laughably small. It is not. Grinold's fundamental law of active management says the information ratio grows with IC times the square root of breadth, and ranking 459 stocks 52 times a year is a lot of breadth. A tiny edge, applied consistently across thousands of independent bets, compounds into a real strategy. Consistency is the evidence of signal; magnitude alone never is.

## From scores to a portfolio: five layers of discipline

A prediction is not a portfolio. Five mathematical controls sit between them.

**Soft cubic weights.** Instead of "buy the top 25", each stock's weight is sign(rank minus 0.5) times |rank minus 0.5| cubed, normalised. Conviction concentrates capital at the extremes, but the mapping is continuous, so small rank changes cause small trades. This single change cut weekly turnover from 96% to 29%, and turnover is money: at 5 basis points per trade, the cubic weighting saved roughly 29 percentage points of return over the backtest versus weekly hard cutoffs.

**Sector neutrality.** Within each of the 11 GICS sectors, weights are demeaned so every sector nets to zero. The portfolio cannot make money by "tech beats energy"; it can only make money picking stocks within sectors. This closes a subtle escape hatch through which sector bets masquerade as stock-picking skill.

**A regime filter.** When SPY sits below its 40-week moving average, the historical signature of a bear market, the entire book goes to cash. Both legs, not just the longs, because holding shorts through a bear-market rally is a directional bet the model never chose to make.

**Volatility targeting.** The book is scaled each week by the ratio of a 12% annual volatility target to its own trailing realised volatility, capped at 1.5x leverage. Risk, not notional size, is held constant, which is what makes a Sharpe ratio comparable across calm and violent markets.

**Stop-losses and caps.** Any position losing more than 8% in a week is exited the following week regardless of the model's opinion, and no stock may exceed 5% of the book.

One more accounting detail that deserves its own paragraph, because getting it wrong destroyed the first version of this project. Each week the backtest earns profit and loss on the weights held *coming into* the week, and only then rebalances. Version one computed P&L on the new weights, a one-line error that granted the portfolio returns it never actually held and produced a nonsense result. The fix is trivial; noticing it is the education.

## The results, read correctly

From April 2018 to December 2024, all out-of-sample: the strategy returned 8.2% annually at 10.7% volatility, against SPY's 15.3% at 18.1%. Its worst drawdown was 21.7% against the index's 32.2%. Its beta to the market is 0.03, and its annualised alpha is +2.9%.

So it underperformed buy-and-hold. Why publish that?

Because with a beta of 0.03, the strategy is nearly orthogonal to the market. It is not competing with SPY; it is uncorrelated with it. In 2018 it beat the index by 12.9 points. In 2022, by 6.2. Its job is to produce returns that do not depend on the market's direction, and the regression says it did. The +2.9% alpha is stock-selection skill by construction, not disguised market exposure.

And the caveats are part of the result. The data source only lists current S&P 500 members, so companies that were delisted during the period, disproportionately the losers, are invisible: survivorship bias inflates these numbers. The five fundamental features use current values rather than point-in-time history, a mild lookahead I disclose rather than hide. The cost model ignores borrow fees and market impact.

The defensible claim is therefore narrow and, I think, more valuable than a bold one: a consistently positive out-of-sample rank correlation, IC 0.051 and positive in 85% of quarterly folds, that survives realistic transaction costs and strict temporal validation, inside a portfolio nearly uncorrelated with the market.

## What I actually learned

That the mathematics of *not fooling yourself* is harder, and more interesting, than the mathematics of prediction. Rank normalisation, walk-forward folds, correct P&L timing, sector demeaning: none of it is glamorous, and all of it is the difference between a result and an artifact.

The full mathematical specification, every formula implemented line-for-line in the code, is in the repository: https://github.com/Nedim21Y/QuantProjectV2.

*I am a high school student in Turkey. This project is for research and education, not financial advice.*
