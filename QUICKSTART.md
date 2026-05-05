# Quick Start Guide - Portfolio Optimization Project

## Setup (5 minutes)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python test_setup.py
```

You should see all tests passing ✓

---

## Run Your First Optimization

### Option A: Default Configuration
```bash
python portfolio_optimization.py
```

This will:
- Download 2 years of data for 100 US large-cap stocks
- Run unconstrained optimization
- Run PSO with 7 realistic constraints
- Let the algorithm choose the final subset of holdings automatically
- Generate visualization (`portfolio_optimization_analysis.png`)
- Print detailed results to console

**Expected Output:**
```
PORTFOLIO OPTIMIZATION - Mean-Variance Model with 7 Constraints
======================================================

Candidate universe size: 100 US large-cap stocks
Data shape: (500, 99)

OPTIMIZATION RESULTS
======================================================

1. UNCONSTRAINED OPTIMIZATION:
Portfolio Expected Return: 15.42%
Portfolio Volatility: 18.76%
Sharpe Ratio: 0.7298

2. PSO OPTIMIZATION (with 7 constraints):
Portfolio Expected Return: 9.81%
Portfolio Volatility: 14.43%
Sharpe Ratio: 0.5415
```

### Option B: Run Examples
```bash
python examples.py
```

This demonstrates 6 different scenarios:
1. Default Configuration
2. Conservative Portfolio (High Risk Aversion)
3. Aggressive Growth (Low Risk Aversion)
4. Tech-Focused Portfolio
5. Different Historical Periods
6. Constraint Sensitivity Analysis

---

## Customize Your Portfolio

### Edit `portfolio_optimization.py` to change the default universe or constraints:
```ini
[DATA]
# Your custom stocks
stocks = IBM, AMZN, NFLX, FB, TWTR

[OPTIMIZATION]
# Risk preference: 0 (return) to 1 (safety)
lambda_param = 0.5

[CONSTRAINTS]
# Maximum per-stock allocation
max_weight_per_asset = 0.25

# Maximum number of positions
max_num_assets = 8

# Minimum if invested
min_investment_threshold = 0.03
```

---

## Python API Usage

### Quick Example
```python
from portfolio_optimization import PortfolioDataFetcher, PortfolioOptimizer
import numpy as np

# Step 1: Fetch data
fetcher = PortfolioDataFetcher(['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
data = fetcher.fetch_data()
mean_returns, cov_matrix, returns = fetcher.calculate_statistics(data)

# Step 2: Create optimizer
optimizer = PortfolioOptimizer(mean_returns, cov_matrix, 4)

# Step 3: Set constraints
params = {
    'lower_bounds': np.zeros(4),
    'upper_bounds': np.ones(4) * 0.3,  # Max 30% per stock
    'max_assets': 3,                    # Max 3 active positions
    'min_invest': 0.05,                 # Min 5% if invested
}

# Step 4: Optimize
weights, cost = optimizer.optimize_pso(
    lambda_param=0.5,  # Balanced risk-return
    params=params,
    n_particles=30,
    n_iterations=100
)

# Step 5: Get metrics
metrics = optimizer.calculate_portfolio_metrics(weights)
print(f"Return: {metrics['return']*100:.2f}%")
print(f"Volatility: {metrics['volatility']*100:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
```

---

## Understanding the Results

### Key Metrics

| Metric | Description | Good Range |
|--------|-------------|-----------|
| **Annual Return** | Expected portfolio return | Higher is better |
| **Annual Volatility** | Portfolio risk (std dev) | Lower is better |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 is good |

### The 7 Constraints

1. **No Short Selling**: All weights ≥ 0
2. **Boundary Limits**: Min/max per asset
3. **Cardinality**: Max number of holdings
4. **Transaction Costs**: Realistic trading fees
5. **Transaction Lots**: Shares in multiples
6. **Sector Limits**: Per-industry allocation caps
7. **Minimum Investment**: Meaningful positions only

---

## Visualization Guide

The generated `portfolio_optimization_analysis.png` contains 6 subplots:

1. **Weight Comparison**: Portfolio allocation differences
2. **Risk-Return Profile**: Efficiency frontier comparison
3. **Correlation Heatmap**: Asset relationships (-1 to +1)
4. **Historical Performance**: Cumulative returns over time
5. **Sharpe Ratio**: Risk-adjusted performance
6. **Return vs Volatility**: Performance trade-off

---

## Common Use Cases

### Conservative Portfolio (Retirees)
```python
# Set high risk aversion
lambda_param = 0.8  # Prioritize safety
params['max_weight_per_asset'] = 0.2  # Lower concentration
params['max_num_assets'] = 10          # More diversification
```

### Growth Portfolio (Young Investors)
```python
# Set low risk aversion
lambda_param = 0.2  # Prioritize returns
params['max_weight_per_asset'] = 0.4  # Higher concentration
params['max_num_assets'] = 5           # Fewer holdings
```

### Sector-Balanced Portfolio
```python
# Add sector constraints
params['sector_limits'] = {
    'tech': [0, 1, 2, 6],
    'finance': [3],
    'consumer': [4],
    'energy': [5]
}
params['sector_max'] = 0.35  # No sector > 35%
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Install missing packages
pip install -r requirements.txt
```

### Issue: "No data for ticker"
- Check stock symbols are valid (e.g., AAPL, not APPLE)
- Verify internet connection (files fetched from Yahoo Finance)
- Try a more recent date range

### Issue: "Optimization is slow"
- Reduce `n_particles` in PSO (default: 30)
- Reduce `n_iterations` (default: 100)
- Use a shorter historical period (e.g., 365 days)

### Issue: "Weights are all zeros"
- Check constraint parameters aren't too restrictive
- Increase PSO iterations
- Verify data statistics are reasonable

---

## Data Source & Updates

All data comes from **Yahoo Finance (yfinance)**
- Historical: Up to 60+ years available
- Real-time: Updated daily
- Global: Supports all major exchanges

### Update Data
To refresh data for recent analysis:
```python
fetcher = PortfolioDataFetcher(
    ['AAPL', 'MSFT'],
    start_date='2024-01-01',  # Recent start
    end_date='2025-12-31'     # Recent end
)
```

---

## Next Steps

1. **Try the examples**: `python examples.py`
2. **Experiment with parameters**: Edit `config.ini`
3. **Customize stocks**: Add your favorites
4. **Adjust constraints**: Make them realistic for your use case
5. **Compare algorithms**: Extend with Differential Evolution (DE)
6. **Real trading**: Integrate with your brokerage API

---

## Additional Resources

- **Portfolio Theory**: https://en.wikipedia.org/wiki/Modern_portfolio_theory
- **PSO Algorithm**: https://github.com/ljvmiranda921/pyswarms
- **Yahoo Finance**: https://finance.yahoo.com
- **Mean-Variance Model**: Markowitz, H. M. (1952). Portfolio Selection

---

## Support

For issues:
1. Check test output: `python test_setup.py`
2. Review console error messages
3. Verify data is available for your stocks
4. Check constraint parameters are reasonable

---

**Happy optimizing!** 📈

