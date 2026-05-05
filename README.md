# Portfolio Optimization Using Mean-Variance Model with 7 Realistic Constraints

## Overview

This project implements a comprehensive portfolio optimization system using the Mean-Variance (M-V) model with real stock data from Yahoo Finance. The program automatically searches for the best constraint configuration and risk-return trade-off (λ) using grid experiments, then applies those settings to produce the final optimized portfolio.

## Key Features

### 1. Real Data Integration (yfinance)
- Historical price data for the past 2 years (~500 trading days)
- Real bid-ask spreads used as variable transaction costs per stock
- Per-stock minimum lot sizes (minimum purchasable shares) fetched from yfinance
- Universe: 100 US large-cap stocks; optimizer selects final holdings automatically

### 2. Objective Function
```
Minimize: Z = [λ × σ_p² - (1 - λ) × E_p] + Penalty_Total
```
- **σ_p²** — Portfolio variance (risk)
- **E_p** — Expected annual return (net of transaction costs)
- **λ** — Risk aversion coefficient, automatically optimized in [0, 1]
- **Penalty_Total** — Weighted sum of all constraint violation penalties

### 3. Seven Realistic Constraints

| # | Constraint | Description |
|---|---|---|
| 1 | **No Short Selling** | All weights w_i ≥ 0 |
| 2 | **Boundary** | Each w_i within [L_i, U_i]; U_i searched over 5%–20% |
| 3 | **Cardinality** | Maximum K active assets; K searched over 5–20 |
| 4 | **Transaction Costs** | Real bid-ask spreads (variable) + broker commission (fixed) |
| 5 | **Transaction Lots** | Per-stock lot step = (min_shares × price) / portfolio_value |
| 6 | **Sector Constraint** | Sector weight ≤ sector_max; searched over 10%–25% |
| 7 | **Minimum Investment** | If invested, allocation ≥ 5% of portfolio |

### 4. Optimization Pipeline

```
Data Fetch & Clean
       ↓
Unconstrained Baseline (SLSQP)
       ↓
Constraint Combination Experiment
  Grid: max_weight (5%–20%, step 1%) × cardinality (5–20) × sector_max (10%–25%, step 5%)
  Up to 1,024 combinations; PSO with 12 particles × 50 iterations each
       ↓
Lambda Sweep (0.00 → 1.00, step 0.01)
  101 values; PSO with 10 particles × 35 iterations each
  Uses best constraints found above
       ↓
Final Portfolio
  Best constraints + best λ → reported results & visualizations
```

### 5. Output Metrics
- **Annual Return** — Expected return net of transaction costs
- **Gross Return** — Return before cost deduction
- **Transaction Cost Deduction** — Cost impact
- **Annual Volatility** — Portfolio standard deviation
- **Sharpe Ratio** — (Return − Risk-free Rate) / Volatility

---

## Project Structure

```
Ci_project/
├── portfolio_optimization.py          # Main implementation
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── trading_costs.csv                  # Real bid-ask data per stock (generated)
├── lot_sizes.csv                      # Real minimum lot sizes per stock (generated)
├── TRADING_PARAMETERS.md              # Trading parameter documentation
├── constraint_combination_results.csv # All combination experiment results (generated)
├── lambda_sweep_results.csv           # Lambda sweep results (generated)
├── constraint_combination_analysis.png# Constraint experiment charts (generated)
└── portfolio_optimization_analysis.png# Final portfolio analysis charts (generated)
```

---

## Installation

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the optimization

```bash
# Full run — all 1,024 combinations + 101 lambda values (may take several hours)
python portfolio_optimization.py

# Scale mode — run only a fraction for faster testing
python portfolio_optimization.py --scale 1   # 10%  (~4 combos,  10 lambda values)
python portfolio_optimization.py --scale 3   # 30%  (~27 combos, 30 lambda values)
python portfolio_optimization.py --scale 5   # 50%  (~128 combos, 50 lambda values)
python portfolio_optimization.py --scale 10  # 100% — same as no flag (default)
```

`--scale` accepts **1–10** (default: `10`).  
Scale 1 = 10 % of each grid dimension sampled evenly, scale 10 = full grid.

### Customize the Stock Universe

Edit `TOP_100_US_LARGE_CAPS` at the top of `portfolio_optimization.py`, or replace the list passed to `PortfolioDataFetcher`:

```python
stocks = ['AAPL', 'MSFT', 'GOOGL', ...]
```

---

## Output

### Console Output
- Universe size and data period
- Real trading parameters (bid-ask spreads, lot sizes, cost per trade)
- Per-stock lot weight summary
- Sector coverage breakdown
- Unconstrained baseline portfolio weights and metrics
- Live progress: `Running combination X/1024 ...` (single overwriting line)
- Top-10 constraint combinations by Sharpe ratio
- Summary sentence (Thai) of the best combination found
- Live progress: `Running lambda X/101 ...` (single overwriting line)
- Best λ and its portfolio metrics
- Final PSO portfolio weights, metrics, and sector breakdown
- Side-by-side comparison table

### Visualizations

**`constraint_combination_analysis.png`** — 6-panel chart:
1. Sharpe ratio heatmap (max_weight × cardinality)
2. Annual return heatmap
3. Risk-Return scatter (all combinations)
4. Sharpe ratio by cardinality (line chart)
5. Sharpe ratio vs combination ID (all tested combinations)
6. Top-10 combinations bar chart

**`portfolio_optimization_analysis.png`** — 9-panel chart:
1. Top portfolio weights comparison (Unconstrained vs PSO)
2. Risk-Return profile scatter
3. Top-holdings correlation heatmap
4. Historical cumulative performance
5. Sharpe ratio comparison
6. Return vs Volatility comparison
7. **Lambda vs Sharpe ratio** (sweep result curve)
8. Portfolio weight distribution (pie chart)
9. Sector allocation comparison

---

## Mathematical Details

### Transaction Cost Model
```
Net Return = Gross Return − (fixed_cost × active_assets) − (variable_cost × portfolio_turnover)
variable_cost = bid-ask spread / 2   (one-way cost)
```

### Per-Stock Lot Size Constraint
```
lot_weight_i = (min_shares_i × price_i) / reference_portfolio_value
weight_i rounded to nearest multiple of lot_weight_i
```
Reference portfolio value: **$100,000**

### Penalty Approach
```
Total Objective = Objective + Σ(penalty_coefficient × violation²)
penalty_coefficient = 1000
```

---

## Requirements

- Python 3.8+
- `numpy` — numerical computations
- `pandas` — data manipulation
- `yfinance` — real stock data (prices, bid-ask, lot sizes, sectors)
- `matplotlib` — visualization
- `scipy` — unconstrained baseline optimization (SLSQP)
- `pyswarms` — PSO implementation

---

## Data Source

**Yahoo Finance via `yfinance`**
- Historical OHLCV prices (2-year window, ~500 trading days)
- Real-time bid/ask prices for transaction cost estimation
- Sector and industry metadata for constraint building
- Minimum lot size (`regularMarketDayLow` and `info` fields)

---

## Key Insights

1. **Automated constraint tuning** — the program finds the best (max_weight, cardinality, sector_max) triple automatically instead of manual guessing
2. **Automated λ tuning** — the lambda sweep eliminates the need to manually set the risk-return trade-off
3. **Real transaction costs** — using live bid-ask spreads grounds the optimization in actual market friction
4. **Per-stock lot rounding** — each stock's minimum purchase unit is respected independently
5. **Scale flag** — `--scale 1` lets you validate the full pipeline end-to-end in minutes
3. **Practical Implementation**: Transaction lots ensure realistic portfolio construction
4. **Risk Management**: Cardinality constraint limits tracking error from too many positions

## Future Enhancements

- Differential Evolution (DE) algorithm comparison
- Black-Litterman model for return estimation
- Real-time portfolio rebalancing
- Machine learning for constraint parameter optimization
- Multi-objective optimization (Pareto frontier)

## License

This project is provided for educational and research purposes.

## Contact & Support

For questions or issues, please refer to the code documentation and comments.

---

**Created by**: Financial Data Scientist and Optimization Specialist
**Last Updated**: 2026
