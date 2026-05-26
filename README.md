# Portfolio Optimization with Mean-Variance + Practical Constraints

This project builds a realistic portfolio optimization workflow using a Mean-Variance objective, Particle Swarm Optimization (PSO), and real market data from Yahoo Finance.

Historical prices and reusable market metadata can now be cached locally in CSV files so the optimizer does not need to call Yahoo Finance on every run.

The pipeline does not just optimize portfolio weights once. It also searches for the best constraint set and the best risk-aversion parameter (lambda) before running the final constrained optimization.

## What This Project Solves

Given a universe of US large-cap stocks, the program finds portfolio weights that balance risk and return under practical trading constraints such as:

1. no short selling
2. position bounds
3. max number of holdings (cardinality)
4. transaction costs
5. lot-size rounding
6. sector exposure cap
7. minimum position size

## Objective Function

The optimizer minimizes:

```text
Z = lambda * variance - (1 - lambda) * expected_return + penalties
```

Where:

1. `variance` is annualized portfolio variance
2. `expected_return` is annualized expected return net of transaction costs
3. `lambda` is risk-aversion in `[0, 1]`
4. `penalties` enforce constraint compliance

## Optimization Workflow

```text
Fetch and clean data
       -> Baseline unconstrained solve (SLSQP)
       -> Constraint combination sweep
                      max_weight x cardinality x sector_max
       -> Lambda sweep on top combinations
       -> Final PSO portfolio with best constraints + best lambda
       -> Reports + CSV exports + plots
```

## Repository Contents

```text
Ci_project/
       portfolio_optimization.py
       examples.py
       test_setup.py
       requirements.txt
       QUICKSTART.md
       TRADING_PARAMETERS.md
       config.ini

       # Generated/analysis artifacts
       constraint_combination_results.csv
       lambda_sweep_results.csv
       trading_costs.csv
       lot_sizes.csv
       constraint_combination_analysis.png
       portfolio_optimization_analysis.png
```

## Requirements

1. Python 3.9+
2. Internet connection (for Yahoo Finance API calls)
3. Dependencies from `requirements.txt`

Install:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python test_setup.py
```

Build the reusable local cache once:

```bash
python build_sp500_cache.py
```

## How To Run

Default full run:

```bash
python portfolio_optimization.py
```

If [sp500_daily_prices_10y.csv](c:/Users/pram/OneDrive/Desktop/devops/djrepo_wk4_slide34/CI_project/sp500_daily_prices_10y.csv), [sp500_asset_metadata.csv](c:/Users/pram/OneDrive/Desktop/devops/djrepo_wk4_slide34/CI_project/sp500_asset_metadata.csv), [sp500_trading_costs.csv](c:/Users/pram/OneDrive/Desktop/devops/djrepo_wk4_slide34/CI_project/sp500_trading_costs.csv), and [sp500_lot_sizes.csv](c:/Users/pram/OneDrive/Desktop/devops/djrepo_wk4_slide34/CI_project/sp500_lot_sizes.csv) exist, the program loads them first and only falls back to `yfinance` when the cache is missing or does not cover the requested slice.

Fast run with reduced grid via scale:

```bash
python portfolio_optimization.py --scale 1
python portfolio_optimization.py --scale 3
python portfolio_optimization.py --scale 5
python portfolio_optimization.py --scale 10
```

Scale behavior:

1. accepts integers `1..10`
2. `1` = 10% sampled density per grid dimension
3. `10` = full search space

## Main Parameters (Current Defaults)

Important defaults inside `portfolio_optimization.py`:

1. Universe: 100 US large-cap stocks
2. Price window: last 2 years
3. `max_weight` search: 0.05 to 0.20 (step 0.01)
4. `cardinality` search: 5 to 20
5. `sector_max` search: 0.10 to 0.25 (step 0.05)
6. lot-size reference portfolio value: `100000`
7. min investment threshold: `0.05`

## Produced Outputs

CSV files:

1. `constraint_combination_results.csv`: all tested constraint combinations and metrics
2. `lambda_sweep_results.csv`: lambda sweep performance across top combinations
3. `trading_costs.csv`: fetched spread-related trading-cost inputs
4. `lot_sizes.csv`: fetched lot-size assumptions per ticker
5. `sp500_daily_prices_10y.csv`: reusable 10-year daily price cache for the current S&P 500 constituents
6. `sp500_asset_metadata.csv`: cached sector metadata
7. `sp500_trading_costs.csv`: cached spread/current-price inputs
8. `sp500_lot_sizes.csv`: cached lot-size inputs

Plots:

1. `constraint_combination_analysis.png`
2. `portfolio_optimization_analysis.png`

Console sections:

1. data diagnostics and market parameter summary
2. unconstrained baseline metrics
3. best constraint-combination leaderboard
4. best lambda summary
5. final constrained portfolio and comparative table

## Notes On Realism

1. Trading costs are estimated from bid-ask spread data and fixed trade cost assumptions.
2. Lot-size handling uses per-asset weight steps derived from lot size and current price.
3. Sector limits are built dynamically from fetched metadata when available.
4. The repair-and-penalty approach keeps candidate solutions feasible enough for stochastic search.

## Troubleshooting

`ModuleNotFoundError`:

```bash
pip install -r requirements.txt
```

No data returned from yfinance:

1. verify internet access
2. verify ticker symbols
3. reduce universe size to test quickly
4. rerun later in case of temporary data-provider issues

Runtime too long:

1. run with `--scale 1` or `--scale 3`
2. reduce PSO particles/iterations in code
3. shorten historical period

## Suggested Next Enhancements

1. Add Differential Evolution baseline for apples-to-apples comparison.
2. Add walk-forward or rolling-window validation.
3. Add explicit turnover constraint across rebalancing dates.
4. Export selected portfolio as broker-ready order sheet.

## Disclaimer

This project is for education and research only. It is not investment advice.
