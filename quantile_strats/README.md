# Quantile-Based Long–Short Equity Strategies

This project implements and analyzes quantile-based long–short equity strategies using accounting-derived financial ratios. The objective is to evaluate the profitability, risk characteristics, and implementation choices of simple “quantamental” signals using a systematic backtesting framework.

## Files

- `analysis_complete.ipynb` — End-to-end analysis and results
- `backtest_framework.py` — Core backtesting engine
- `strategy_definitions.py` — Strategy and signal definitions
- `build_investment_universe.ipynb` — Universe construction and filtering
- `financial_ratios_cleaned.ipynb` — Data cleaning and ratio construction
- `daily_ratios.xlsx` — Final daily ratio dataset
- `investment_universe.csv` — Filtered equity universe
- `spy_returns.csv` — Market benchmark data
- `Data_export/` — Exported tables and figures
- `Data_export.zip` — Zipped outputs for submission

## Methodology

**Universe**
- ~500 U.S. equities
- Coverage: **2018-03-09 to 2023-06-30**
- Excludes financial, insurance, and automotive sectors
- Market cap ≥ **$100MM** throughout the sample

**Ratios**
- Return on Investment (ROI)
- Price-to-Earnings (P/E)
- Debt-to-Market Capitalization

Accounting data is aligned to filing dates and forward-filled between filings to avoid look-ahead bias.

## Strategies

- Base quantile strategies (top/bottom decile): ROI, P/E, Debt-to-MktCap
- Combined strategy: **Quality–Value** (rank-based combination of ROI, Debt-to-MktCap, and P/E)
- Momentum variants: ranking on **changes** in ratios rather than levels
- Position sizing variants: equal-weight vs. vigintile overweight/underweight

## Portfolio Construction

- Dollar-neutral long–short portfolios
- Equal gross exposure on long and short sides
- Initial capital set to **10× gross notional**
- Monthly rebalancing by default; weekly rebalancing tested
- Zero transaction costs assumed

## Key Findings

- The **Quality–Value** strategy is the best base strategy on a risk-adjusted basis.
- **Equal-weight sizing** is optimal; vigintile overweight/underweight reduces Sharpe and PL/Notional.
- **Momentum (ratio-change) strategies** perform poorly, indicating signal noise/instability.
- **Weekly rebalancing** improves Quality–Value performance without materially worsening drawdowns.
- Downside beta analysis indicates limited crash sensitivity across strategies.

## Notes

This project is for research and educational purposes only and does not constitute investment advice. Results depend on modeling assumptions, signal construction, and rebalancing frequency.
