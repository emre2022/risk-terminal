# Institutional Risk Terminal

A multi-asset market-risk workstation for a trading desk. Parametric and historical VaR, risk decomposition down to the individual position, category-level stress testing, portfolio performance analytics, and a grid-search strategy backtester — in one Streamlit application.

Built and used by a working FX/CFD dealer, so the workflow follows how a desk actually operates: paste the position blotter, get exposure, VaR and risk contribution in one pass.

```
Python · Streamlit · pandas · NumPy · SciPy · Plotly · yfinance
```

---

## Screenshots

### Risk breakdown — exposure, VaR and per-position risk contribution
<img width="1471" alt="breakdown" src="https://github.com/user-attachments/assets/82d8f387-9405-489b-947d-fc0ec63b50ad" />

### Correlation matrix and historical P&L distribution against the VaR threshold
<img width="1437" alt="simulation" src="https://github.com/user-attachments/assets/4a205130-3d83-48ee-9f76-d0dc98b3ba4a" />

### Backtest optimizer — net P&L across SMA period × risk:reward ratio
<img width="1439" alt="heatmap" src="https://github.com/user-attachments/assets/baf66db7-7e24-4fbc-a28e-573857c1ab24" />

---

## What it does

| Module | Output |
|---|---|
| **Risk breakdown** | Gross exposure, 99% VaR, diversification benefit, expected annualised volatility. Allocation and per-position risk contribution charts. |
| **Simulation & correlation** | Historical P&L distribution with the parametric VaR threshold overlaid, scenario impact, correlation matrix. |
| **Detailed report** | Position-level signed exposure, marginal VaR, component VaR, percentage risk contribution. |
| **Backtest optimizer** | Grid search over SMA 2–200 × selectable risk:reward ratios. Net P&L, win rate, trade count per combination, with a performance heatmap. |
| **Data cleaner** | Regex parser that turns a pasted broker blotter into a structured frame and auto-populates the portfolio. |

---

## Risk methodology

**Parametric VaR.** Annualised implied volatility per instrument is converted to a daily figure (σ_annual / √252). The covariance matrix is the correlation matrix element-wise multiplied by the outer product of daily volatilities. Portfolio variance is wᵀΣw on signed dollar exposures; VaR is the portfolio standard deviation scaled by the normal z-score at the chosen confidence level — 99% headline, 95% also reported.

**Diversification benefit.** Standalone VaR is computed per position and summed to give undiversified VaR. The difference against diversified portfolio VaR is reported in dollars and as a percentage — the number a risk committee actually asks for.

**Marginal and component VaR.** Marginal VaR is (Σw / σ_p) × z, the sensitivity of portfolio VaR to a unit increase in each exposure. Component VaR is marginal VaR multiplied by that position's exposure, and the components sum back to total diversified VaR. Percentage risk contribution follows directly — which is what identifies the position actually driving the book's risk, often not the largest one.

**Historical VaR.** Computed independently as the 1st percentile of the portfolio P&L series reconstructed from two years of daily returns. A non-parametric cross-check on the model-based number.

**Stress testing.** Shocks applied at asset-category level (equity, FX, commodity, crypto) and revalued through position size and contract multiplier into a dollar P&L impact.

**Performance analytics.** Sharpe at a 4% risk-free rate, Sortino on downside deviation only, beta against the S&P 500.

---

## Backtest engine

The strategy under test is a daily-SMA retest. Hourly OHLC is resampled to daily bars to compute the moving average, then execution is simulated back on the hourly series — so the signal is daily but the fill is intraday.

A long triggers when the previous daily close was above the SMA and price trades back down to it; a short is the mirror. Stop loss is a fixed dollar amount, take profit is that amount multiplied by the risk:reward ratio under test, and both convert to price points via the contract multiplier.

Same-bar entry and exit are resolved explicitly rather than assumed, and concurrent positions are tracked independently.

The optimizer sweeps roughly 200 SMA periods against each selected R:R ratio and surfaces the whole parameter space as a heatmap. The question it answers is not *"does this setting work"* but *"is there a stable region where it works"* — an isolated profitable cell is overfitting, a broad profitable zone is signal.

---

## Instrument coverage

| Class | Instruments |
|---|---|
| Indices | Nasdaq 100 E-mini, S&P 500 E-mini, Dow E-mini, DAX 40 |
| FX | EUR/USD, GBP/USD, USD/JPY, USD/TRY, Euro FX futures |
| Commodities & energy | Gold, Silver, WTI crude, Natural gas, Platinum, Palladium |
| Crypto | Bitcoin, Ethereum |

Each carries its contract multiplier and a default implied-volatility input, both overridable per position.

---

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Market data is pulled from Yahoo Finance via `yfinance` — no API key required.

---

## Assumptions and limitations

Stated deliberately, since a risk tool that hides its assumptions is worth less than one that does not.

- **Normality.** Parametric VaR assumes normally distributed returns and therefore understates tail risk. The historical VaR figure is included as a deliberate counterweight.
- **Volatility input.** Implied volatility is user-supplied rather than read from a live surface. Garbage in, garbage out applies directly to the VaR number.
- **Static correlations.** Estimated from two years of daily returns and held fixed. They are not stressed — and in practice they converge toward one precisely when it hurts most.
- **Backtest costs.** Spread, commission, slippage and overnight financing are not modelled, so absolute P&L is optimistic. The engine is built to compare parameter regions against each other, not to forecast returns.
- **Linear revaluation.** Positions are revalued linearly; there is no second-order (gamma) treatment for optionality.
