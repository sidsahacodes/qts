# Trade Flow Return Predictions in Crypto Markets

Quantitative analysis of high-frequency trade flow as a predictor of short-term returns in BTC-USDT markets.

## Overview

**Research Question:** Does short-term trade flow predict future returns in cryptocurrency markets?

**Methodology:** Strict in-sample/out-of-sample validation with no data snooping.

## Mathematical Framework

**Trade Flow** $F_i^{(\tau)}$: Net signed quantity over $(t_i - \tau, t_i)$

$$F_i^{(\tau)} = \sum_{j: t_i - \tau < t_j < t_i} q_j \cdot \mathbb{1}_{\text{buy}_j} - q_j \cdot \mathbb{1}_{\text{sell}_j}$$

**Forward Return** $r_i^{(T)}$: Price change from $t_i$ to $t_i + T$

$$r_i^{(T)} = \frac{P(t_i + T) - P(t_i)}{P(t_i)}$$

**Model:** Through-origin OLS regression

$$r_i^{(T)} = \beta \cdot F_i^{(\tau)} + \epsilon_i \quad \Rightarrow \quad \beta = \frac{\sum F_i r_i}{\sum F_i^2}$$

## Workflow

### Phase 1: In-Sample Discovery (40% of data)
- Estimate β coefficient
- Determine trading threshold
- Analyze performance metrics
- Discover volatility patterns

### Phase 2: Out-of-Sample Validation (60% of data)
- Apply trained parameters (no re-optimization)
- Validate on unseen data
- Compare in-sample vs out-of-sample

## Installation

```bash
pip install pandas numpy matplotlib seaborn scipy pyarrow
```

## Usage

```bash
jupyter notebook flow_analysis_proper_workflow.ipynb
```

**Parameters:**
- τ = 1.0s (lookback window)
- T = 1.0s (forward window)
- Transaction costs = 5 bps

## Data Format

| Column | Description |
|--------|-------------|
| `ts` | Trade timestamp |
| `side` | 'B' (buy) or 'A' (sell) |
| `qty` | Trade quantity |
| `trade_price` | Execution price |
| `Exchange` | Exchange identifier |

## Key Results

### In-Sample
- β = 1.32 × 10⁻⁵ (CV = 51%)
- Gross P&L: +$10,144 (Sharpe = 0.077)
- Net P&L: -$25,647 (Sharpe = -0.134)

### Out-of-Sample
- Gross P&L: +$20,941 (Sharpe = 0.050)
- Net P&L: -$42,723 (Sharpe = -0.088)

### Findings

1. **Transaction costs dominate**: 5 bps costs eliminate profitability
2. **Volatility dependence**: Strategy works better in low-volatility regimes
3. **Unstable β**: 51% CV indicates time-varying relationship
4. **Limited viability**: Not deployable as-is; illustrates microstructure effects

## Project Structure

```
.
├── flow_analysis_proper_workflow.ipynb    # Main analysis
├── WORKFLOW_EXPLANATION.md                # Methodology details
├── README.md                              # This file
└── data/
    └── BTC-USDT_fewer_trades.parquet     # Trade data
```

## Extensions

- Test on additional exchanges (OKX)
- Optimize τ and T parameters
- Implement volatility filters
- Add latency modeling
- Try adaptive β estimation
- Explore ML alternatives
