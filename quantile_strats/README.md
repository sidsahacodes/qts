# Quantile-Based Long–Short Equity Strategies

This project implements and analyzes quantile-based long–short equity strategies using accounting-derived financial ratios. The goal is to evaluate the profitability, risk characteristics, and implementation choices of simple “quantamental” signals, following the assignment framework for Financial Ratio Quantile Strategies.

---

## Project Structure

quantile_strats/
│
├── analysis_complete.ipynb # End-to-end analysis and results
├── backtest_framework.py # Core backtesting engine
├── strategy_definitions.py # Strategy and signal definitions
│
├── build_investment_universe.ipynb # Universe construction and filtering
├── financial_ratios_cleaned.ipynb # Data cleaning and ratio construction
│
├── daily_ratios.xlsx # Final daily ratio dataset
├── investment_universe.csv # Filtered equity universe
├── spy_returns.csv # Market benchmark data
│
├── Data_export/ # Exported tables and figures
├── Data_export.zip # Zipped outputs for submission



## Methodology Overview

### Investment Universe
- ~500 U.S. equities
- Continuous price and ratio coverage from **2018-03-09 to 2023-06-30**
- Excludes financial, insurance, and automotive sectors
- Market capitalization ≥ $100MM throughout the sample

### Financial Ratios
- **Return on Investment (ROI)**
- **Price-to-Earnings (P/E)**
- **Debt-to-Market Capitalization**

Ratios are constructed using filing-date–aware accounting data and forward-filled between filings to avoid look-ahead bias.

---

## Strategies Tested

### Base Quantile Strategies (Top–Bottom Decile)
- Long high ROI / short low ROI  
- Long low P/E / short high P/E  
- Long low Debt-to-Market-Cap / short high Debt-to-Market-Cap  

### Combined Strategy
- **Quality–Value**: weighted combination of ROI, Debt/MktCap, and P/E using cross-sectional ranks

### Momentum Variants
- Ranking on changes in ratios rather than levels

---

## Portfolio Construction

- Dollar-neutral long–short portfolios
- Equal gross exposure on long and short sides
- Initial capital set to **10× gross notional**
- Monthly rebalancing by default, with weekly rebalancing tested for robustness
- Zero transaction costs assumed

---

## Key Findings

- The **Quality–Value combined strategy** delivers the strongest performance among base strategies, outperforming all single-ratio signals on a risk-adjusted basis.
- **Equal weighting** across positions is optimal; rank-based sizing (vigintile overweight/underweight) reduces Sharpe and trading efficiency.
- Strategies based on **changes in accounting ratios** perform poorly, indicating high noise and instability.
- **Weekly rebalancing** materially improves performance for the Quality–Value strategy without increasing drawdowns, suggesting faster signal decay than monthly horizons.
- Downside beta estimates indicate limited exposure to market crashes across strategies.

---
