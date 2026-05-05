"""
Portfolio Optimization using Mean-Variance Model with Multiple Constraints
Role: Financial Data Scientist and Optimization Specialist

This program implements portfolio optimization using:
- Mean-Variance (M-V) Model
- 7 Realistic Constraints
- PSO and DE algorithms for comparison
- Real stock data from yfinance
"""

import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pyswarms.single.global_best import GlobalBestPSO
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

TOP_100_US_LARGE_CAPS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'TSLA', 'JPM', 'V',
    'LLY', 'XOM', 'UNH', 'MA', 'COST', 'NFLX', 'WMT', 'JNJ', 'PG', 'ORCL',
    'HD', 'BAC', 'ABBV', 'KO', 'CVX', 'AMD', 'CRM', 'MRK', 'ADBE', 'PEP',
    'TMO', 'LIN', 'ACN', 'AVGO', 'MCD', 'CSCO', 'DHR', 'ABT', 'WFC', 'DIS',
    'QCOM', 'TXN', 'PM', 'IBM', 'INTC', 'GE', 'CAT', 'GS', 'RTX', 'NOW',
    'MS', 'BKNG', 'BLK', 'AMGN', 'SPGI', 'ISRG', 'SCHW', 'MDT', 'GILD', 'BA',
    'SYK', 'HON', 'TJX', 'ADP', 'AMT', 'DE', 'T', 'LOW', 'PGR', 'C',
    'ETN', 'MMC', 'PLD', 'LMT', 'CB', 'UPS', 'MO', 'NEE', 'SBUX', 'ELV',
    'MDLZ', 'CI', 'USB', 'BMY', 'SO', 'CME', 'DUK', 'ICE', 'CL', 'MMM',
    'FIS', 'APD', 'ZTS', 'CVS', 'WM', 'TMUS', 'EQIX', 'PYPL', 'COP', 'UNP'
]

# ========================
# 1. DATA EXTRACTION
# ========================

class PortfolioDataFetcher:
    """Fetch and prepare portfolio data from yfinance"""
    
    def __init__(self, ticker_symbols, start_date=None, end_date=None):
        """
        Initialize data fetcher
        
        Args:
            ticker_symbols: List of stock ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        self.ticker_symbols = ticker_symbols
        self.end_date = end_date or datetime.now().date()
        self.start_date = start_date or (self.end_date - timedelta(days=365*2))
        
    def fetch_data(self):
        """Fetch adjusted close prices from yfinance"""
        print(f"Fetching data for {len(self.ticker_symbols)} stocks...")
        print(f"Period: {self.start_date} to {self.end_date}")
        
        # Download data
        raw_data = yf.download(
            self.ticker_symbols,
            start=self.start_date,
            end=self.end_date,
            progress=False,
            auto_adjust=False
        )

        if raw_data.empty:
            raise ValueError("No price data was returned by yfinance for the selected tickers and date range.")

        if isinstance(raw_data.columns, pd.MultiIndex):
            price_field = 'Adj Close' if 'Adj Close' in raw_data.columns.get_level_values(0) else 'Close'
            data = raw_data[price_field].copy()
        else:
            price_field = 'Adj Close' if 'Adj Close' in raw_data.columns else 'Close'
            data = raw_data[[price_field]].copy()
        
        if isinstance(data, pd.Series):
            data = data.to_frame(name=self.ticker_symbols[0])
        elif len(self.ticker_symbols) == 1:
            data.columns = self.ticker_symbols

        available_tickers = [ticker for ticker in self.ticker_symbols if ticker in data.columns]
        missing_tickers = [ticker for ticker in self.ticker_symbols if ticker not in data.columns]
        if missing_tickers:
            print(f"Warning: yfinance returned no usable price series for {len(missing_tickers)} tickers.")

        cleaned_data = data[available_tickers].dropna(axis=1, how='all').dropna()
        if cleaned_data.shape[1] == 0:
            raise ValueError("No valid price series remained after cleaning the downloaded data.")

        return cleaned_data
    
    def calculate_statistics(self, data):
        """Calculate mean returns and covariance matrix"""
        # Daily returns
        returns = data.pct_change().dropna()
        
        # Annualized statistics
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        return mean_returns, cov_matrix, returns

    @staticmethod
    def get_top_100_us_stocks():
        """Return a diversified universe of 100 US large-cap stocks."""
        return TOP_100_US_LARGE_CAPS.copy()

    @staticmethod
    def _fetch_single_asset_metadata(ticker_symbol):
        """Fetch sector metadata for one ticker with safe fallbacks."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            sector = info.get('sector') or info.get('sectorDisp') or info.get('industry') or 'Unknown'
        except Exception:
            sector = 'Unknown'

        return {'ticker': ticker_symbol, 'sector': sector}

    def fetch_asset_metadata(self, ticker_symbols=None, max_workers=10):
        """Fetch sector metadata for all tickers in parallel."""
        symbols = ticker_symbols or self.ticker_symbols
        metadata_rows = []

        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            futures = {
                executor.submit(self._fetch_single_asset_metadata, ticker_symbol): ticker_symbol
                for ticker_symbol in symbols
            }
            for future in as_completed(futures):
                metadata_rows.append(future.result())

        metadata = pd.DataFrame(metadata_rows).set_index('ticker').reindex(symbols)
        metadata['sector'] = metadata['sector'].fillna('Unknown')
        return metadata

    @staticmethod
    def build_sector_constraints(ticker_symbols, asset_metadata):
        """Create sector-to-index mapping used by the sector exposure constraint."""
        sector_limits = {}
        for sector_name, sector_frame in asset_metadata.groupby('sector'):
            indices = [ticker_symbols.index(ticker_symbol) for ticker_symbol in sector_frame.index]
            if indices:
                sector_limits[sector_name] = indices
        return sector_limits

    @staticmethod
    def _fetch_trading_costs_for_ticker(ticker_symbol):
        """Fetch bid-ask spread and estimate transaction costs for one ticker."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 100.0
            bid = info.get('bid') or current_price * 0.999
            ask = info.get('ask') or current_price * 1.001

            current_price = max(current_price, 0.01)
            bid = max(bid, 0.01)
            ask = max(ask, bid)

            spread = (ask - bid) / current_price if current_price > 0 else 0.001
            spread = max(spread, 0.0001)

            return {
                'ticker': ticker_symbol,
                'current_price': current_price,
                'bid': bid,
                'ask': ask,
                'spread_pct': spread
            }
        except Exception:
            return {
                'ticker': ticker_symbol,
                'current_price': 100.0,
                'bid': 99.9,
                'ask': 100.1,
                'spread_pct': 0.002
            }

    def fetch_trading_costs(self, ticker_symbols=None, max_workers=10):
        """Fetch trading cost estimates (bid-ask spread) for all tickers in parallel."""
        symbols = ticker_symbols or self.ticker_symbols
        cost_rows = []

        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            futures = {
                executor.submit(self._fetch_trading_costs_for_ticker, ticker_symbol): ticker_symbol
                for ticker_symbol in symbols
            }
            for future in as_completed(futures):
                cost_rows.append(future.result())

        trading_costs = pd.DataFrame(cost_rows).set_index('ticker').reindex(symbols)
        return trading_costs

    @staticmethod
    def _fetch_lot_size_for_ticker(ticker_symbol):
        """Fetch minimum lot size for one ticker."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice') or 100.0
            # yfinance usually does not expose broker-specific min quantity.
            # Use ticker lotSize when available, otherwise use market-standard minimum 1 share.
            lot_size = info.get('lotSize') or 1
            lot_size = max(int(lot_size), 1)

            return {
                'ticker': ticker_symbol,
                'lot_size': lot_size,
                'price_point': current_price
            }
        except Exception:
            return {
                'ticker': ticker_symbol,
                'lot_size': 1,
                'price_point': 100.0
            }

    def fetch_lot_sizes(self, ticker_symbols=None, max_workers=10):
        """Fetch lot size constraints for all tickers in parallel."""
        symbols = ticker_symbols or self.ticker_symbols
        lot_rows = []

        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as executor:
            futures = {
                executor.submit(self._fetch_lot_size_for_ticker, ticker_symbol): ticker_symbol
                for ticker_symbol in symbols
            }
            for future in as_completed(futures):
                lot_rows.append(future.result())

        lot_sizes = pd.DataFrame(lot_rows).set_index('ticker').reindex(symbols)
        return lot_sizes
    
    @staticmethod
    def get_sample_data():
        """Backward-compatible alias for the default 100-stock universe."""
        return PortfolioDataFetcher.get_top_100_us_stocks()


# ========================
# 2. PORTFOLIO OPTIMIZER
# ========================

class PortfolioOptimizer:
    """Portfolio optimization with M-V model and constraints"""
    
    def __init__(self, mean_returns, cov_matrix, num_assets):
        """
        Initialize optimizer
        
        Args:
            mean_returns: Vector of mean annual returns
            cov_matrix: Covariance matrix of returns
            num_assets: Number of assets
        """
        self.mean_returns = mean_returns.values
        self.cov_matrix = cov_matrix.values
        self.num_assets = num_assets
        
    def objective_function(self, weights, lambda_param=0.5):
        """
        M-V Objective Function with Penalty Approach
        
        Minimize: Z = [λ * σ_p² - (1 - λ) * E_p] + Penalty_Total
        
        Args:
            weights: Portfolio weights
            lambda_param: Risk aversion coefficient (0 to 1)
        """
        # Portfolio variance (risk)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, self.mean_returns)
        
        # Base objective
        objective = lambda_param * portfolio_variance - (1 - lambda_param) * portfolio_return
        
        return objective

    def estimate_transaction_cost(self, weights, params):
        """Estimate fixed and variable transaction costs for active positions."""
        fixed_cost = params.get('fixed_cost', 0.001)
        variable_cost = params.get('variable_cost', 0.0005)
        active_positions = np.sum(weights > 1e-8)
        turnover = np.sum(np.abs(weights))
        return active_positions * fixed_cost + variable_cost * turnover

    def _get_lot_weight_vector(self, params):
        """Return lot-size step in portfolio weights for each asset."""
        lot_weights = params.get('lot_size_weights')
        if lot_weights is not None:
            lot_weights = np.asarray(lot_weights, dtype=float)
            if lot_weights.shape[0] != self.num_assets:
                raise ValueError("lot_size_weights length must match number of assets")
            return np.maximum(lot_weights, 1e-8)

        scalar_lot = float(params.get('lot_size', 0.0))
        if scalar_lot <= 0:
            return np.full(self.num_assets, 1e-8)
        return np.full(self.num_assets, max(scalar_lot, 1e-8))

    def repair_weights(self, weights, params=None):
        """Project candidate weights into a portfolio that better respects practical constraints."""
        if params is None:
            params = {}

        repaired = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        repaired = np.maximum(repaired, 0.0)

        lower_bounds = np.asarray(params.get('lower_bounds', np.zeros(self.num_assets)), dtype=float)
        upper_bounds = np.asarray(params.get('upper_bounds', np.ones(self.num_assets)), dtype=float)
        upper_bounds = np.maximum(upper_bounds, lower_bounds)

        max_assets = min(int(params.get('max_assets', self.num_assets)), self.num_assets)
        min_invest = float(params.get('min_invest', 0.0))
        lot_weight_vector = self._get_lot_weight_vector(params)
        sector_limits = params.get('sector_limits', {})
        sector_max = float(params.get('sector_max', 1.0))

        if np.sum(repaired) <= 0:
            seed_count = max(1, max_assets)
            seed_indices = np.argsort(self.mean_returns)[::-1][:seed_count]
            repaired[seed_indices] = 1.0

        if max_assets < self.num_assets:
            keep_indices = np.argsort(repaired)[::-1][:max_assets]
            keep_mask = np.zeros(self.num_assets, dtype=bool)
            keep_mask[keep_indices] = True
            repaired[~keep_mask] = 0.0

        repaired = np.minimum(repaired, upper_bounds)
        active_mask = repaired > 0
        repaired[active_mask] = np.maximum(repaired[active_mask], lower_bounds[active_mask])

        if min_invest > 0:
            repaired[(repaired > 0) & (repaired < min_invest)] = 0.0

        if max_assets < self.num_assets:
            keep_indices = np.argsort(repaired)[::-1][:max_assets]
            keep_mask = np.zeros(self.num_assets, dtype=bool)
            keep_mask[keep_indices] = True
            repaired[~keep_mask] = 0.0

        if np.sum(repaired) <= 0:
            fallback_count = max(1, max_assets)
            fallback_indices = np.argsort(self.mean_returns)[::-1][:fallback_count]
            repaired[fallback_indices] = 1.0 / fallback_count

        repaired = repaired / np.sum(repaired)

        if np.any(lot_weight_vector > 1e-8):
            repaired = np.round(repaired / lot_weight_vector) * lot_weight_vector
            repaired = np.maximum(repaired, 0.0)

        if sector_limits:
            for sector_indices in sector_limits.values():
                sector_weight = np.sum(repaired[sector_indices])
                if sector_weight > sector_max and sector_weight > 0:
                    repaired[sector_indices] *= sector_max / sector_weight

        if np.sum(repaired) <= 0:
            repaired[np.argmax(self.mean_returns)] = min(1.0, upper_bounds[np.argmax(self.mean_returns)])

        for _ in range(10):
            repaired = np.minimum(repaired, upper_bounds)
            repaired[(repaired > 0) & (repaired < min_invest)] = 0.0

            portfolio_sum = np.sum(repaired)
            if portfolio_sum >= 1.0 - 1e-8:
                break

            remaining_capacity = np.maximum(upper_bounds - repaired, 0.0)
            if sector_limits:
                for sector_indices in sector_limits.values():
                    sector_weight = np.sum(repaired[sector_indices])
                    sector_room = max(sector_max - sector_weight, 0.0)
                    sector_capacity = np.sum(remaining_capacity[sector_indices])
                    if sector_capacity <= 0:
                        continue
                    if sector_room <= 0:
                        remaining_capacity[sector_indices] = 0.0
                    elif sector_capacity > sector_room:
                        remaining_capacity[sector_indices] *= sector_room / sector_capacity

            total_capacity = np.sum(remaining_capacity)
            if total_capacity <= 0:
                break

            repaired += remaining_capacity * ((1.0 - portfolio_sum) / total_capacity)

        repaired = np.minimum(repaired, upper_bounds)
        repaired[(repaired > 0) & (repaired < min_invest)] = 0.0

        if np.sum(repaired) <= 0:
            fallback_index = int(np.argmax(self.mean_returns))
            repaired[fallback_index] = min(1.0, upper_bounds[fallback_index])

        repaired = repaired / np.sum(repaired)
        return repaired
    
    def penalty_function(self, weights, params):
        """
        Calculate total penalty for constraint violations
        
        Constraints:
        1. No Short Sell: w_i >= 0
        2. Boundary: L_i <= w_i <= U_i
        3. Cardinality: Number of non-zero weights <= K
        4. Transaction Costs: Fixed + Variable costs
        5. Transaction Lots: Weights as multiples of lot size
        6. Sector Constraint: Sector weights within limits
        7. Minimum Investment: If invested, minimum allocation
        """
        penalty = 0
        penalty_weight = 1000  # Large penalty coefficient
        
        # 1. No Short Sell
        short_sell_penalty = np.sum(np.maximum(-weights, 0))
        penalty += penalty_weight * short_sell_penalty
        
        # 2. Boundary Constraint
        lower_bounds = params.get('lower_bounds', np.zeros(self.num_assets))
        upper_bounds = params.get('upper_bounds', np.ones(self.num_assets))
        boundary_penalty = np.sum(np.maximum(lower_bounds - weights, 0)) + \
                          np.sum(np.maximum(weights - upper_bounds, 0))
        penalty += penalty_weight * boundary_penalty
        
        # 3. Cardinality Constraint
        max_assets = params.get('max_assets', self.num_assets)
        min_invest = params.get('min_invest', 0.05)
        num_active = np.sum(weights >= max(1e-4, min_invest))
        if num_active > max_assets:
            cardinality_penalty = (num_active - max_assets) ** 2
            penalty += penalty_weight * cardinality_penalty
        
        # 4. Sum to 1 constraint
        sum_penalty = np.abs(np.sum(weights) - 1.0)
        penalty += penalty_weight * sum_penalty
        
        # 5. Transaction Costs
        penalty += penalty_weight * self.estimate_transaction_cost(weights, params)

        # 6. Transaction Lots (rounding to nearest lot size per asset)
        lot_weight_vector = self._get_lot_weight_vector(params)
        lot_penalty = np.sum(np.abs(weights - np.round(weights / lot_weight_vector) * lot_weight_vector))
        penalty += penalty_weight * lot_penalty
        
        # 7. Sector Constraint
        sector_limits = params.get('sector_limits', {})
        if sector_limits:
            sector_max = params.get('sector_max', 0.4)
            for sector, indices in sector_limits.items():
                sector_weight = np.sum(weights[indices])
                if sector_weight > sector_max:
                    penalty += penalty_weight * (sector_weight - sector_max)
        
        # 8. Minimum Investment
        for i, w in enumerate(weights):
            if 0 < w < min_invest:
                penalty += penalty_weight * (min_invest - w)
        
        return penalty
    
    def combined_objective(self, weights, lambda_param=0.5, params=None):
        """Combined objective with penalty"""
        if params is None:
            params = {}

        repaired_weights = self.repair_weights(weights, params)
        transaction_cost = self.estimate_transaction_cost(repaired_weights, params)
        portfolio_variance = np.dot(repaired_weights, np.dot(self.cov_matrix, repaired_weights))
        portfolio_return = np.dot(repaired_weights, self.mean_returns) - transaction_cost
        objective = lambda_param * portfolio_variance - (1 - lambda_param) * portfolio_return
        penalty = self.penalty_function(repaired_weights, params)
        
        return objective + penalty
    
    def optimize_pso(self, lambda_param=0.5, params=None, n_particles=30, n_iterations=100):
        """
        Optimize using Particle Swarm Optimization
        
        Args:
            lambda_param: Risk aversion coefficient
            params: Constraint parameters
            n_particles: Number of particles
            n_iterations: Number of iterations
        """
        if params is None:
            params = {}
        
        # PSO options
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        # Initialize PSO
        optimizer = GlobalBestPSO(
            n_particles=n_particles,
            dimensions=self.num_assets,
            options=options,
            bounds=(np.zeros(self.num_assets), np.ones(self.num_assets))
        )
        
        # Define objective for PSO
        def pso_objective(particles):
            objectives = np.array([
                self.combined_objective(p, lambda_param, params)
                for p in particles
            ])
            return objectives
        
        # Optimize
        best_cost, best_weights = optimizer.optimize(
            pso_objective,
            iters=n_iterations,
            verbose=False
        )
        
        best_weights = self.repair_weights(best_weights, params)
        
        return best_weights, best_cost
    
    def optimize_unconstrained(self, lambda_param=0.5):
        """Optimize without constraints (baseline)"""
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        x0 = np.array([1/self.num_assets] * self.num_assets)
        result = minimize(
            lambda x: self.objective_function(x, lambda_param),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x, result.fun
    
    def calculate_portfolio_metrics(self, weights, params=None):
        """Calculate portfolio metrics"""
        params = params or {}
        weights = self.repair_weights(weights, params)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        gross_return = np.dot(weights, self.mean_returns)
        transaction_cost = self.estimate_transaction_cost(weights, params)
        portfolio_return = gross_return - transaction_cost
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - params.get('risk_free_rate', 0.02)) / portfolio_std
        
        return {
            'return': portfolio_return,
            'gross_return': gross_return,
            'transaction_cost': transaction_cost,
            'volatility': portfolio_std,
            'variance': portfolio_variance,
            'sharpe_ratio': sharpe_ratio
        }


def evaluate_constraint_combinations(
    optimizer,
    base_params,
    lambda_param,
    max_weight_grid,
    cardinality_grid,
    sector_max_grid=None,
    n_particles=12,
    n_iterations=50
):
    """Evaluate PSO performance across constraint combinations and return the best set."""
    records = []
    num_assets = optimizer.num_assets
    sector_grid = sector_max_grid or [base_params.get('sector_max', 0.25)]
    best_sharpe = -np.inf
    best_weights = None

    total_runs = len(max_weight_grid) * len(cardinality_grid) * len(sector_grid)
    run_count = 0

    print("\n" + "=" * 70)
    print("CONSTRAINT COMBINATION EXPERIMENT")
    print("=" * 70)
    print(f"Total combinations: {total_runs}")

    for max_weight in max_weight_grid:
        for max_assets in cardinality_grid:
            for sector_max in sector_grid:
                run_count += 1
                print(
                    (
                        f"Running combination {run_count}/{total_runs} "
                        f"(max_weight={max_weight:.2f}, cardinality={max_assets}, sector_max={sector_max:.2f})"
                    ),
                    end='\r',
                    flush=True
                )

                test_params = dict(base_params)
                test_params['upper_bounds'] = np.ones(num_assets) * max_weight
                test_params['max_assets'] = int(max_assets)
                test_params['sector_max'] = float(sector_max)

                weights, objective_value = optimizer.optimize_pso(
                    lambda_param=lambda_param,
                    params=test_params,
                    n_particles=n_particles,
                    n_iterations=n_iterations
                )
                metrics = optimizer.calculate_portfolio_metrics(weights, test_params)
                selected_assets = int(np.sum(weights >= test_params['min_invest'] - 1e-8))

                if float(metrics['sharpe_ratio']) > best_sharpe:
                    best_sharpe = float(metrics['sharpe_ratio'])
                    best_weights = weights.copy()

                records.append(
                    {
                        'combination_id': int(run_count),
                        'max_weight': float(max_weight),
                        'cardinality': int(max_assets),
                        'sector_max': float(sector_max),
                        'objective': float(objective_value),
                        'annual_return': float(metrics['return']),
                        'annual_volatility': float(metrics['volatility']),
                        'sharpe_ratio': float(metrics['sharpe_ratio']),
                        'selected_assets': selected_assets
                    }
                )

    print(' ' * 120, end='\r', flush=True)
    print(f"Completed {total_runs}/{total_runs} combinations")

    results_df = pd.DataFrame(records)
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_row = results_df.loc[best_idx]
    best_params = dict(base_params)
    best_params['upper_bounds'] = np.ones(num_assets) * float(best_row['max_weight'])
    best_params['max_assets'] = int(best_row['cardinality'])
    best_params['sector_max'] = float(best_row['sector_max'])
    return results_df, best_row, best_params, best_weights


def evaluate_lambda_sweep(
    optimizer,
    best_params,
    lambda_grid,
    n_particles=10,
    n_iterations=35,
    run_label=None,
    show_header=True,
    show_lambda_progress=True,
    show_summary=True
):
    """Sweep lambda in [0,1] and find the lambda that maximizes Sharpe ratio."""
    records = []
    best_sharpe = -np.inf
    best_weights = None

    total_lambdas = len(lambda_grid)

    if show_header:
        print("\n" + "=" * 70)
        if run_label:
            print(f"LAMBDA SWEEP EXPERIMENT ({run_label})")
        else:
            print("LAMBDA SWEEP EXPERIMENT")
        print("=" * 70)
        print(f"Total lambda values: {total_lambdas}")

    for idx, lambda_value in enumerate(lambda_grid, start=1):
        if show_lambda_progress:
            print(
                f"Running lambda {idx}/{total_lambdas} (lambda={lambda_value:.2f})",
                end='\r',
                flush=True
            )

        weights, objective_value = optimizer.optimize_pso(
            lambda_param=float(lambda_value),
            params=best_params,
            n_particles=n_particles,
            n_iterations=n_iterations
        )
        metrics = optimizer.calculate_portfolio_metrics(weights, best_params)

        if float(metrics['sharpe_ratio']) > best_sharpe:
            best_sharpe = float(metrics['sharpe_ratio'])
            best_weights = weights.copy()

        records.append(
            {
                'lambda': float(lambda_value),
                'objective': float(objective_value),
                'annual_return': float(metrics['return']),
                'annual_volatility': float(metrics['volatility']),
                'sharpe_ratio': float(metrics['sharpe_ratio'])
            }
        )

    if show_summary:
        print(' ' * 100, end='\r', flush=True)
        print(f"Completed {total_lambdas}/{total_lambdas} lambda values")

    lambda_results_df = pd.DataFrame(records)
    best_idx = lambda_results_df['sharpe_ratio'].idxmax()
    best_lambda_row = lambda_results_df.loc[best_idx]
    return lambda_results_df, best_lambda_row, best_weights


def create_constraint_combination_visualizations(results_df, best_row):
    """Visualize constraint-combination experiment outcomes."""
    viz_df = results_df.copy()
    # Keep heatmaps 2D by taking the best Sharpe over sector_max for each (max_weight, cardinality).
    viz_df = (
        viz_df.sort_values('sharpe_ratio', ascending=False)
        .groupby(['cardinality', 'max_weight'], as_index=False)
        .first()
    )

    fig = plt.figure(figsize=(18, 14))

    # Heatmap 1: Sharpe ratio
    ax1 = plt.subplot(3, 2, 1)
    sharpe_pivot = viz_df.pivot(index='cardinality', columns='max_weight', values='sharpe_ratio')
    im1 = ax1.imshow(sharpe_pivot.values, cmap='viridis', aspect='auto')
    ax1.set_title('Sharpe Ratio by Constraint Combination')
    ax1.set_xlabel('Max Weight per Stock')
    ax1.set_ylabel('Cardinality')
    ax1.set_xticks(np.arange(len(sharpe_pivot.columns)))
    ax1.set_xticklabels([f"{x:.2f}" for x in sharpe_pivot.columns])
    ax1.set_yticks(np.arange(len(sharpe_pivot.index)))
    ax1.set_yticklabels([str(int(y)) for y in sharpe_pivot.index])
    plt.colorbar(im1, ax=ax1)

    # Heatmap 2: Annual return
    ax2 = plt.subplot(3, 2, 2)
    return_pivot = viz_df.pivot(index='cardinality', columns='max_weight', values='annual_return')
    im2 = ax2.imshow(return_pivot.values, cmap='YlGnBu', aspect='auto')
    ax2.set_title('Annual Return by Constraint Combination')
    ax2.set_xlabel('Max Weight per Stock')
    ax2.set_ylabel('Cardinality')
    ax2.set_xticks(np.arange(len(return_pivot.columns)))
    ax2.set_xticklabels([f"{x:.2f}" for x in return_pivot.columns])
    ax2.set_yticks(np.arange(len(return_pivot.index)))
    ax2.set_yticklabels([str(int(y)) for y in return_pivot.index])
    plt.colorbar(im2, ax=ax2)

    # Scatter: risk-return and best combination
    ax3 = plt.subplot(3, 2, 3)
    scatter = ax3.scatter(
        results_df['annual_volatility'],
        results_df['annual_return'],
        c=results_df['sharpe_ratio'],
        s=(results_df['cardinality'] * 4),
        cmap='plasma',
        alpha=0.8
    )
    ax3.scatter(
        best_row['annual_volatility'],
        best_row['annual_return'],
        s=220,
        marker='*',
        color='red',
        label='Best Sharpe Combination'
    )
    ax3.set_title('Risk vs Return Across Combinations')
    ax3.set_xlabel('Annual Volatility')
    ax3.set_ylabel('Annual Return')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='Sharpe Ratio')

    # Line chart: Sharpe by max weight for each cardinality
    ax4 = plt.subplot(3, 2, 4)
    best_sector = float(best_row['sector_max'])
    line_df = results_df[np.isclose(results_df['sector_max'], best_sector)]
    for cardinality in sorted(line_df['cardinality'].unique()):
        card_frame = line_df[line_df['cardinality'] == cardinality].sort_values('max_weight')
        ax4.plot(card_frame['max_weight'], card_frame['sharpe_ratio'], marker='o', label=f'K={cardinality}')
    ax4.set_title(f'Sharpe Trend at sector_max={best_sector:.2f}')
    ax4.set_xlabel('Max Weight per Stock')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best')

    # Required chart: y=sharpe ratio, x=each combination
    ax5 = plt.subplot(3, 2, 5)
    combo_line = results_df.sort_values('combination_id')
    ax5.plot(combo_line['combination_id'], combo_line['sharpe_ratio'], color='teal', linewidth=1.4)
    ax5.scatter(
        best_row['combination_id'],
        best_row['sharpe_ratio'],
        color='red',
        s=80,
        marker='*',
        label='Best Combination'
    )
    ax5.set_title('Sharpe Ratio by Each Combination')
    ax5.set_xlabel('Combination ID')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best')

    # Top combinations bar chart
    ax6 = plt.subplot(3, 2, 6)
    top10 = results_df.sort_values('sharpe_ratio', ascending=False).head(10)
    ax6.bar(np.arange(len(top10)), top10['sharpe_ratio'], color='slateblue', alpha=0.8)
    ax6.set_title('Top 10 Combination Sharpe Ratios')
    ax6.set_xlabel('Top Combination Rank')
    ax6.set_ylabel('Sharpe Ratio')
    ax6.set_xticks(np.arange(len(top10)))
    ax6.set_xticklabels([str(i + 1) for i in range(len(top10))])
    ax6.grid(axis='y', alpha=0.3)

    fig.suptitle(
        (
            f"Best constraints -> max_weight={best_row['max_weight']:.2f}, "
            f"cardinality={int(best_row['cardinality'])}, sector_max={best_row['sector_max']:.2f}, "
            f"Sharpe={best_row['sharpe_ratio']:.4f}"
        ),
        fontsize=11,
        y=1.02
    )

    plt.tight_layout()
    plt.savefig('constraint_combination_analysis.png', dpi=300, bbox_inches='tight')
    print('Saved: constraint_combination_analysis.png')
    plt.close(fig)


# ========================
# 3. MAIN EXECUTION
# ========================

def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(description='Portfolio Optimization')
    parser.add_argument(
        '--scale',
        type=int,
        default=10,
        choices=range(1, 11),
        metavar='1-10',
        help='Run scale: 1=10%% of combinations, 10=100%% (default: 10)'
    )
    args = parser.parse_args()
    scale = args.scale
    scale_pct = scale * 10

    print("="*70)
    print("PORTFOLIO OPTIMIZATION - Mean-Variance Model with 7 Constraints")
    print("="*70)
    if scale < 10:
        print(f"[Scale mode: {scale}/10 — running {scale_pct}% of combinations and lambda values]")
    
    # Build the optimization universe from 100 US large-cap stocks.
    stocks = PortfolioDataFetcher.get_top_100_us_stocks()
    print(f"\nCandidate universe size: {len(stocks)} US large-cap stocks")
    
    # Fetch data
    fetcher = PortfolioDataFetcher(stocks)
    price_data = fetcher.fetch_data()
    stocks = list(price_data.columns)
    mean_returns, cov_matrix, returns = fetcher.calculate_statistics(price_data)
    
    # Fetch real trading parameters
    print("\nFetching real trading costs and lot sizes...")
    asset_metadata = fetcher.fetch_asset_metadata(stocks)
    trading_costs = fetcher.fetch_trading_costs(stocks)
    lot_sizes = fetcher.fetch_lot_sizes(stocks)
    sector_limits = fetcher.build_sector_constraints(stocks, asset_metadata)
    
    # Calculate average trading costs
    avg_bid_ask_spread = trading_costs['spread_pct'].mean()
    avg_lot_price = lot_sizes['price_point'].mean()
    
    # Commission assuming modern broker (e.g., Robinhood, Fidelity = free, but interactive brokers ~ $1/trade for stocks)
    commission_per_trade = 0.0  # Free commissions on most modern platforms
    
    # Calculate variable cost as average bid-ask spread per transaction
    # Bid-ask spread is paid each way (half on entry, half on exit for round-trip)
    variable_cost_pct = (avg_bid_ask_spread / 2.0) * 0.01  # Convert to percentage and take half
    variable_cost_pct = max(variable_cost_pct, 0.0001)  # Minimum 0.01% per transaction
    
    # Fixed cost is commission per position or minimum fee
    fixed_cost_pct = commission_per_trade / 10000 if avg_lot_price > 0 else 0.0  # As percentage of $10k position
    
    print(f"\nData shape: {price_data.shape}")
    print(f"Usable stocks after data cleaning: {len(stocks)}")
    print("\nTrading Parameters (Real Data):")
    print(f"  Average Bid-Ask Spread: {avg_bid_ask_spread*100:.4f}%")
    print(f"  Average Variable Cost per Trade: {variable_cost_pct*100:.4f}%")
    print(f"  Fixed Commission Cost: ${commission_per_trade:.2f}")
    print(f"  Average Lot Price Point: ${avg_lot_price:.2f}")
    print(f"  Minimum Lot Size (shares): {lot_sizes['lot_size'].min():.0f}")
    print(f"\nAll candidate stocks with usable data ({len(stocks)}):")
    print(stocks)
    print("\nMean Annual Returns:")
    print(mean_returns.sort_values(ascending=False).head(15))
    print("\nAnnualized Volatility:")
    print(pd.Series(np.sqrt(np.diag(cov_matrix)), index=stocks).sort_values(ascending=False).head(15))
    print("\nSector coverage:")
    print(asset_metadata['sector'].value_counts().to_string())
    print("\nBid-Ask Spread by Stock (Top 10):")
    print(trading_costs['spread_pct'].sort_values(ascending=False).head(10))
    
    # Initialize optimizer
    num_assets = len(stocks)
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
    
    # Define constraints parameters with real trading data
    # Per-stock lot size converted into weight using a reference portfolio value.
    # weight_step_i = (min_shares_i * price_i) / reference_portfolio_value
    reference_portfolio_value = 100000.0
    lot_weights = (lot_sizes['lot_size'] * lot_sizes['price_point']) / reference_portfolio_value
    lot_weights = lot_weights.clip(lower=0.0001, upper=0.02)
    lot_weights = lot_weights.reindex(stocks).fillna(0.0001)

    print("\nPer-stock lot weight summary:")
    print(f"  Min lot weight: {lot_weights.min()*100:.4f}%")
    print(f"  Mean lot weight: {lot_weights.mean()*100:.4f}%")
    print(f"  Max lot weight: {lot_weights.max()*100:.4f}%")
    
    params = {
        'lower_bounds': np.zeros(num_assets),  # No short selling
        'upper_bounds': np.ones(num_assets) * 0.12,  # Max 12% per stock
        'max_assets': 12,  # Maximum active positions chosen from 100 stocks
        'lot_size': float(lot_weights.mean()),  # Backward-compatible scalar fallback
        'lot_size_weights': lot_weights.values,  # Per-stock lot step from real minimum shares
        'reference_portfolio_value': reference_portfolio_value,
        'min_invest': 0.05,  # Minimum 5% if invested
        'sector_max': 0.25,  # Max 25% per sector
        'fixed_cost': fixed_cost_pct,  # Real commission cost
        'variable_cost': variable_cost_pct,  # Real bid-ask spread cost
        'risk_free_rate': 0.02,
        'sector_limits': sector_limits,
        'trading_costs_data': trading_costs,
        'lot_sizes_data': lot_sizes
    }
    
    # Lambda parameter for risk-return tradeoff
    lambda_param = 0.5  # Equal weight on risk and return
    
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    
    # 1. Unconstrained optimization
    print("\n1. UNCONSTRAINED OPTIMIZATION:")
    w_unconstrained, cost_unconstrained = optimizer.optimize_unconstrained(lambda_param)
    metrics_unconstrained = optimizer.calculate_portfolio_metrics(w_unconstrained)
    
    print("Optimal Weights:")
    for stock, weight in zip(stocks, w_unconstrained):
        if weight > 0.01:
            print(f"  {stock}: {weight:.4f} ({weight*100:.2f}%)")
    
    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {metrics_unconstrained['return']:.4f} ({metrics_unconstrained['return']*100:.2f}%)")
    print(f"  Volatility (Std Dev): {metrics_unconstrained['volatility']:.4f} ({metrics_unconstrained['volatility']*100:.2f}%)")
    print(f"  Sharpe Ratio: {metrics_unconstrained['sharpe_ratio']:.4f}")
    
    # 2. Expanded constraint-combination experiment (requested ranges)
    def _subsample_grid(grid, scale):
        """Pick evenly spaced items from a grid according to scale (1-10)."""
        n = max(1, round(len(grid) * scale / 10))
        indices = np.round(np.linspace(0, len(grid) - 1, n)).astype(int)
        return [grid[i] for i in indices]

    _mw_full = np.round(np.arange(0.05, 0.201, 0.01), 2).tolist()
    _card_full = list(range(5, 21))
    _sm_full = np.round(np.arange(0.10, 0.251, 0.05), 2).tolist()

    max_weight_grid = _subsample_grid(_mw_full, scale)
    cardinality_grid = _subsample_grid(_card_full, scale)
    sector_max_grid = _subsample_grid(_sm_full, scale)

    print(f"\nConstraint grid sizes: max_weight={len(max_weight_grid)}, cardinality={len(cardinality_grid)}, sector_max={len(sector_max_grid)}")
    print(f"Total combinations: {len(max_weight_grid) * len(cardinality_grid) * len(sector_max_grid)}")

    combo_results_df, best_combo, best_params, best_weights = evaluate_constraint_combinations(
        optimizer=optimizer,
        base_params=params,
        lambda_param=lambda_param,
        max_weight_grid=max_weight_grid,
        cardinality_grid=cardinality_grid,
        sector_max_grid=sector_max_grid,
        n_particles=12,
        n_iterations=50
    )

    combo_results_df = combo_results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    combo_results_df.to_csv('constraint_combination_results.csv', index=False)

    print("\nTop 10 combinations by Sharpe ratio:")
    print(
        combo_results_df[
            [
                'max_weight', 'cardinality', 'sector_max',
                'annual_return', 'annual_volatility', 'sharpe_ratio', 'selected_assets'
            ]
        ].head(10).to_string(index=False)
    )

    print("\nBest combination found from experiment:")
    print(f"  max_weight: {best_combo['max_weight']:.2f}")
    print(f"  cardinality: {int(best_combo['cardinality'])}")
    print(f"  sector_max: {best_combo['sector_max']:.2f}")
    print(f"  annual_return: {best_combo['annual_return']*100:.2f}%")
    print(f"  annual_volatility: {best_combo['annual_volatility']*100:.2f}%")
    print(f"  sharpe_ratio: {best_combo['sharpe_ratio']:.4f}")
    print(
        "\nจากการทดลอง combinations ทั้งหมดพบว่า "
        f"ค่า constraint max_weight={best_combo['max_weight']:.2f}, "
        f"cardinality={int(best_combo['cardinality'])}, "
        f"sector_max={best_combo['sector_max']:.2f} "
        "จะให้ผลดีที่สุดเมื่อนำมาใช้กับ PSO"
    )

    create_constraint_combination_visualizations(combo_results_df, best_combo)

    # 3. Lambda sweep from 0.00 to 1.00 with step 0.01 on Top-10 combinations
    _lambda_full = np.round(np.arange(0.0, 1.001, 0.01), 2).tolist()
    lambda_grid = _subsample_grid(_lambda_full, scale)
    print(f"\nLambda grid size: {len(lambda_grid)} values")

    top_k = min(10, len(combo_results_df))
    top_combo_rows = combo_results_df.head(top_k)
    all_lambda_results = []
    best_lambda_sharpe = -np.inf
    best_lambda_row = None
    best_lambda_weights = None
    best_params = None
    best_combo_for_lambda = None
    best_combo_lambda_curve_df = None

    print(f"\nRunning lambda sweep on Top {top_k} combinations...")
    for rank, (_, combo_row) in enumerate(top_combo_rows.iterrows(), start=1):
        print(f"Lambda combo rank {rank}/{top_k}", end='\r', flush=True)
        combo_params = dict(params)
        combo_params['upper_bounds'] = np.ones(num_assets) * float(combo_row['max_weight'])
        combo_params['max_assets'] = int(combo_row['cardinality'])
        combo_params['sector_max'] = float(combo_row['sector_max'])

        run_label = (
            f"combo_rank={rank}/{top_k}, "
            f"max_weight={combo_row['max_weight']:.2f}, "
            f"cardinality={int(combo_row['cardinality'])}, "
            f"sector_max={combo_row['sector_max']:.2f}"
        )
        combo_lambda_df, combo_best_lambda_row, combo_best_weights = evaluate_lambda_sweep(
            optimizer=optimizer,
            best_params=combo_params,
            lambda_grid=lambda_grid,
            n_particles=10,
            n_iterations=35,
            run_label=run_label,
            show_header=False,
            show_lambda_progress=False,
            show_summary=False
        )

        combo_lambda_df = combo_lambda_df.copy()
        combo_lambda_df['combo_rank'] = rank
        combo_lambda_df['combo_max_weight'] = float(combo_row['max_weight'])
        combo_lambda_df['combo_cardinality'] = int(combo_row['cardinality'])
        combo_lambda_df['combo_sector_max'] = float(combo_row['sector_max'])
        all_lambda_results.append(combo_lambda_df)

        combo_best_sharpe = float(combo_best_lambda_row['sharpe_ratio'])
        if combo_best_sharpe > best_lambda_sharpe:
            best_lambda_sharpe = combo_best_sharpe
            best_lambda_row = combo_best_lambda_row.copy()
            best_lambda_weights = combo_best_weights.copy()
            best_params = combo_params
            best_combo_for_lambda = combo_row.copy()
            best_combo_lambda_curve_df = combo_lambda_df.copy()

    print(' ' * 60, end='\r', flush=True)
    print(f"Completed lambda sweep combo ranks {top_k}/{top_k}")

    lambda_results_df = pd.concat(all_lambda_results, ignore_index=True)
    lambda_results_df.to_csv('lambda_sweep_results.csv', index=False)

    print("\nBest result from lambda sweep across Top 10 combinations:")
    print(f"  combo max_weight={best_combo_for_lambda['max_weight']:.2f}")
    print(f"  combo cardinality={int(best_combo_for_lambda['cardinality'])}")
    print(f"  combo sector_max={best_combo_for_lambda['sector_max']:.2f}")
    print(f"  best lambda={best_lambda_row['lambda']:.2f}")
    print(f"  annual_return={best_lambda_row['annual_return']*100:.2f}%")
    print(f"  annual_volatility={best_lambda_row['annual_volatility']*100:.2f}%")
    print(f"  sharpe_ratio={best_lambda_row['sharpe_ratio']:.4f}")

    # 4. Final PSO analysis with best dynamic constraints and best lambda
    print("\n4. PSO OPTIMIZATION (best dynamic constraints + best lambda):")
    w_pso = best_lambda_weights
    metrics_pso = optimizer.calculate_portfolio_metrics(w_pso, best_params)
    selected_stocks = [
        (stock, weight)
        for stock, weight in zip(stocks, w_pso)
        if weight >= best_params['min_invest'] - 1e-8
    ]

    print("Optimal Weights:")
    for stock, weight in selected_stocks:
        print(f"  {stock}: {weight:.4f} ({weight*100:.2f}%)")

    print(f"\nPortfolio Metrics:")
    print(f"  Expected Return: {metrics_pso['return']:.4f} ({metrics_pso['return']*100:.2f}%)")
    print(f"  Gross Return Before Costs: {metrics_pso['gross_return']:.4f} ({metrics_pso['gross_return']*100:.2f}%)")
    print(f"  Transaction Cost Deduction: {metrics_pso['transaction_cost']:.4f} ({metrics_pso['transaction_cost']*100:.2f}%)")
    print(f"  Volatility (Std Dev): {metrics_pso['volatility']:.4f} ({metrics_pso['volatility']*100:.2f}%)")
    print(f"  Sharpe Ratio: {metrics_pso['sharpe_ratio']:.4f}")
    print(f"  Sum of Weights: {np.sum(w_pso):.6f}")
    print(f"  Selected Assets: {len(selected_stocks)}")
    print("\nApplied best settings to final PSO run:")
    print(
        f"  max_weight={best_params['upper_bounds'][0]:.2f}, "
        f"max_assets={best_params['max_assets']}, "
        f"sector_max={best_params['sector_max']:.2f}, "
        f"lambda={best_lambda_row['lambda']:.2f}"
    )
    print("\nSelected sectors:")
    print(asset_metadata.loc[[stock for stock, _ in selected_stocks], 'sector'].value_counts().to_string())

    # 5. Create results comparison
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)

    results_df = pd.DataFrame({
        'Metric': ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Selected Assets'],
        'Unconstrained': [
            f"{metrics_unconstrained['return']*100:.2f}%",
            f"{metrics_unconstrained['volatility']*100:.2f}%",
            f"{metrics_unconstrained['sharpe_ratio']:.4f}",
            f"{np.sum(w_unconstrained > 0.01)}"
        ],
        'PSO (Constrained, Best Dynamic)': [
            f"{metrics_pso['return']*100:.2f}%",
            f"{metrics_pso['volatility']*100:.2f}%",
            f"{metrics_pso['sharpe_ratio']:.4f}",
            f"{len(selected_stocks)}"
        ]
    })

    print("\n" + results_df.to_string(index=False))
    
    # 6. Visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS...")
    print("="*70)
    
    create_visualizations(
        stocks,
        w_unconstrained,
        w_pso,
        returns,
        metrics_unconstrained,
        metrics_pso,
        best_params,
        best_combo_lambda_curve_df,
        best_lambda_row
    )
    
    print("\nVisualizations saved successfully!")
    print("\nProgram completed successfully!")
    
    return {
        'unconstrained': {'weights': w_unconstrained, 'metrics': metrics_unconstrained},
        'constrained_pso': {'weights': w_pso, 'metrics': metrics_pso},
        'best_constraints': {
            'max_weight': float(best_combo_for_lambda['max_weight']),
            'cardinality': int(best_combo_for_lambda['cardinality']),
            'sector_max': float(best_combo_for_lambda['sector_max'])
        },
        'best_lambda': float(best_lambda_row['lambda']),
        'stocks': stocks
    }
def create_visualizations(
    stocks,
    w_unconstrained,
    w_pso,
    returns,
    metrics_unconstrained,
    metrics_pso,
    params=None,
    lambda_results_df=None,
    best_lambda_row=None
):
    """Create visualization plots focused on the most relevant assets in a large universe."""
    params = params or {}
    top_display_count = min(15, len(stocks))
    top_indices = np.argsort(w_pso)[::-1][:top_display_count]
    top_stocks = [stocks[index] for index in top_indices]
    top_unconstrained = w_unconstrained[top_indices]
    top_pso = w_pso[top_indices]
    display_returns = returns[top_stocks]
    
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Weight Comparison
    ax1 = plt.subplot(3, 3, 1)
    x = np.arange(len(top_stocks))
    width = 0.35
    ax1.bar(x - width/2, top_unconstrained, width, label='Unconstrained', alpha=0.8)
    ax1.bar(x + width/2, top_pso, width, label='PSO (Constrained)', alpha=0.8)
    ax1.set_xlabel('Assets')
    ax1.set_ylabel('Weight')
    ax1.set_title('Top Portfolio Weights Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_stocks, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Risk-Return Trade-off
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(metrics_unconstrained['volatility'], metrics_unconstrained['return'],
               s=300, marker='o', label='Unconstrained', color='blue', alpha=0.7)
    ax2.scatter(metrics_pso['volatility'], metrics_pso['return'],
               s=300, marker='s', label='PSO (Constrained)', color='red', alpha=0.7)
    ax2.scatter(np.sqrt(np.diag(display_returns.cov()*252)), display_returns.mean()*252,
               s=100, marker='.', label='Individual Assets', color='gray', alpha=0.5)
    ax2.set_xlabel('Volatility (Risk)')
    ax2.set_ylabel('Expected Return')
    ax2.set_title('Risk-Return Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Correlation Heatmap
    ax3 = plt.subplot(3, 3, 3)
    corr_matrix = display_returns.corr()
    im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax3.set_xticks(range(len(top_stocks)))
    ax3.set_yticks(range(len(top_stocks)))
    ax3.set_xticklabels(top_stocks, rotation=45)
    ax3.set_yticklabels(top_stocks)
    ax3.set_title('Top Holdings Correlation Matrix')
    plt.colorbar(im, ax=ax3)
    
    # 4. Cumulative Returns (Price Series)
    ax4 = plt.subplot(3, 3, 4)
    normalized_returns = (1 + display_returns).cumprod()
    for stock in top_stocks[:10]:
        ax4.plot(normalized_returns.index, normalized_returns[stock], label=stock, linewidth=2)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Cumulative Return')
    ax4.set_title('Historical Performance of Top Candidates')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # 5. Sharpe Ratio Comparison
    ax5 = plt.subplot(3, 3, 5)
    sharpe_ratios = [metrics_unconstrained['sharpe_ratio'], metrics_pso['sharpe_ratio']]
    labels = ['Unconstrained', 'PSO (Constrained)']
    colors = ['blue', 'red']
    bars = ax5.bar(labels, sharpe_ratios, color=colors, alpha=0.7)
    ax5.set_ylabel('Sharpe Ratio')
    ax5.set_title('Risk-Adjusted Performance Comparison')
    ax5.grid(axis='y', alpha=0.3)
    for bar, ratio in zip(bars, sharpe_ratios):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.4f}', ha='center', va='bottom')
    
    # 6. Volatility and Return Comparison
    ax6 = plt.subplot(3, 3, 6)
    x_pos = np.arange(2)
    width = 0.35
    returns_vals = [metrics_unconstrained['return']*100, metrics_pso['return']*100]
    volatility_vals = [metrics_unconstrained['volatility']*100, metrics_pso['volatility']*100]
    
    ax6_twin = ax6.twinx()
    bars1 = ax6.bar(x_pos - width/2, returns_vals, width, label='Annual Return', color='green', alpha=0.7)
    bars2 = ax6_twin.bar(x_pos + width/2, volatility_vals, width, label='Annual Volatility', color='orange', alpha=0.7)
    
    ax6.set_ylabel('Annual Return (%)', color='green')
    ax6_twin.set_ylabel('Annual Volatility (%)', color='orange')
    ax6.set_xlabel('Portfolio Strategy')
    ax6.set_title('Return vs Volatility')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(labels)
    ax6.tick_params(axis='y', labelcolor='green')
    ax6_twin.tick_params(axis='y', labelcolor='orange')
    ax6.grid(axis='y', alpha=0.3)

    # 7. Lambda sweep chart: y=Sharpe ratio, x=Lambda
    ax7 = plt.subplot(3, 3, 7)
    if lambda_results_df is not None and not lambda_results_df.empty:
        ax7.plot(lambda_results_df['lambda'], lambda_results_df['sharpe_ratio'], color='purple', linewidth=1.8)
        if best_lambda_row is not None:
            ax7.scatter(
                best_lambda_row['lambda'],
                best_lambda_row['sharpe_ratio'],
                color='red',
                marker='*',
                s=180,
                label=(
                    f"Best lambda={best_lambda_row['lambda']:.2f}, "
                    f"Sharpe={best_lambda_row['sharpe_ratio']:.4f}"
                )
            )
            ax7.axvline(best_lambda_row['lambda'], color='red', linestyle='--', alpha=0.5)
            ax7.legend(loc='best')
    ax7.set_title('Lambda vs Sharpe Ratio')
    ax7.set_xlabel('Lambda')
    ax7.set_ylabel('Sharpe Ratio')
    ax7.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_optimization_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: portfolio_optimization_analysis.png")
    plt.close(fig)


if __name__ == "__main__":
    results = main()
