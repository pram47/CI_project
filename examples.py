"""
Example Usage of Portfolio Optimization Module
Demonstrates different scenarios and configurations
"""

import numpy as np
from portfolio_optimization import (
    PortfolioDataFetcher,
    PortfolioOptimizer,
    create_visualizations
)

def example_1_default():
    """Example 1: Default configuration with standard parameters"""
    print("\n" + "="*70)
    print("EXAMPLE 1: DEFAULT CONFIGURATION")
    print("="*70)
    
    # Get default stocks
    stocks = PortfolioDataFetcher.get_sample_data()
    
    # Fetch data
    fetcher = PortfolioDataFetcher(stocks)
    price_data = fetcher.fetch_data()
    mean_returns, cov_matrix, returns = fetcher.calculate_statistics(price_data)
    
    # Optimize
    num_assets = len(stocks)
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
    
    params = {
        'lower_bounds': np.zeros(num_assets),
        'upper_bounds': np.ones(num_assets) * 0.3,
        'max_assets': 5,
        'lot_size': 0.01,
        'min_invest': 0.05,
    }
    
    w_unconstrained, _ = optimizer.optimize_unconstrained(lambda_param=0.5)
    w_pso, _ = optimizer.optimize_pso(lambda_param=0.5, params=params)
    
    metrics_unconstrained = optimizer.calculate_portfolio_metrics(w_unconstrained)
    metrics_pso = optimizer.calculate_portfolio_metrics(w_pso)
    
    print(f"\nUnconstrained Portfolio Return: {metrics_unconstrained['return']*100:.2f}%")
    print(f"Constrained Portfolio Return: {metrics_pso['return']*100:.2f}%")
    print(f"Unconstrained Portfolio Volatility: {metrics_unconstrained['volatility']*100:.2f}%")
    print(f"Constrained Portfolio Volatility: {metrics_pso['volatility']*100:.2f}%")


def example_2_high_risk_aversion():
    """Example 2: High risk aversion (conservative portfolio)"""
    print("\n" + "="*70)
    print("EXAMPLE 2: HIGH RISK AVERSION (λ = 0.8)")
    print("="*70)
    
    stocks = PortfolioDataFetcher.get_sample_data()
    fetcher = PortfolioDataFetcher(stocks)
    price_data = fetcher.fetch_data()
    mean_returns, cov_matrix, returns = fetcher.calculate_statistics(price_data)
    
    num_assets = len(stocks)
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
    
    params = {
        'lower_bounds': np.zeros(num_assets),
        'upper_bounds': np.ones(num_assets) * 0.2,  # More conservative
        'max_assets': 3,  # Fewer assets
        'lot_size': 0.01,
        'min_invest': 0.05,
    }
    
    w_pso, _ = optimizer.optimize_pso(lambda_param=0.8, params=params)  # High lambda
    metrics = optimizer.calculate_portfolio_metrics(w_pso)
    
    print(f"\nPortfolio Return: {metrics['return']*100:.2f}%")
    print(f"Portfolio Volatility: {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print("\nWeights (Conservative Portfolio):")
    for stock, weight in zip(stocks, w_pso):
        if weight > 0.001:
            print(f"  {stock}: {weight*100:.2f}%")


def example_3_aggressive_growth():
    """Example 3: Low risk aversion (aggressive growth portfolio)"""
    print("\n" + "="*70)
    print("EXAMPLE 3: AGGRESSIVE GROWTH (λ = 0.2)")
    print("="*70)
    
    stocks = PortfolioDataFetcher.get_sample_data()
    fetcher = PortfolioDataFetcher(stocks)
    price_data = fetcher.fetch_data()
    mean_returns, cov_matrix, returns = fetcher.calculate_statistics(price_data)
    
    num_assets = len(stocks)
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
    
    params = {
        'lower_bounds': np.zeros(num_assets),
        'upper_bounds': np.ones(num_assets) * 0.4,  # More aggressive
        'max_assets': 7,  # All assets
        'lot_size': 0.01,
        'min_invest': 0.01,  # Lower threshold
    }
    
    w_pso, _ = optimizer.optimize_pso(lambda_param=0.2, params=params)  # Low lambda
    metrics = optimizer.calculate_portfolio_metrics(w_pso)
    
    print(f"\nPortfolio Return: {metrics['return']*100:.2f}%")
    print(f"Portfolio Volatility: {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print("\nWeights (Aggressive Portfolio):")
    for stock, weight in zip(stocks, w_pso):
        if weight > 0.001:
            print(f"  {stock}: {weight*100:.2f}%")


def example_4_custom_stocks():
    """Example 4: Custom stocks selection"""
    print("\n" + "="*70)
    print("EXAMPLE 4: CUSTOM STOCKS (Tech-focused)")
    print("="*70)
    
    # Custom tech stocks
    custom_stocks = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC']
    
    fetcher = PortfolioDataFetcher(custom_stocks)
    price_data = fetcher.fetch_data()
    mean_returns, cov_matrix, returns = fetcher.calculate_statistics(price_data)
    
    num_assets = len(custom_stocks)
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
    
    params = {
        'lower_bounds': np.zeros(num_assets),
        'upper_bounds': np.ones(num_assets) * 0.3,
        'max_assets': 4,
        'lot_size': 0.01,
        'min_invest': 0.05,
    }
    
    w_pso, _ = optimizer.optimize_pso(lambda_param=0.5, params=params)
    metrics = optimizer.calculate_portfolio_metrics(w_pso)
    
    print(f"\nCustom Stocks: {custom_stocks}")
    print(f"Portfolio Return: {metrics['return']*100:.2f}%")
    print(f"Portfolio Volatility: {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print("\nWeights (Tech Portfolio):")
    for stock, weight in zip(custom_stocks, w_pso):
        if weight > 0.001:
            print(f"  {stock}: {weight*100:.2f}%")


def example_5_different_periods():
    """Example 5: Different historical periods"""
    print("\n" + "="*70)
    print("EXAMPLE 5: DIFFERENT LOOKBACK PERIODS")
    print("="*70)
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'JPM']
    
    for days, period_name in [(365, "1 Year"), (730, "2 Years"), (1825, "5 Years")]:
        from datetime import datetime, timedelta
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        fetcher = PortfolioDataFetcher(stocks, start_date, end_date)
        price_data = fetcher.fetch_data()
        mean_returns, cov_matrix, returns = fetcher.calculate_statistics(price_data)
        
        num_assets = len(stocks)
        optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
        
        w_pso, _ = optimizer.optimize_pso(lambda_param=0.5)
        metrics = optimizer.calculate_portfolio_metrics(w_pso)
        
        print(f"\n{period_name}:")
        print(f"  Return: {metrics['return']*100:.2f}%")
        print(f"  Volatility: {metrics['volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")


def example_6_constraint_sensitivity():
    """Example 6: Sensitivity analysis for constraints"""
    print("\n" + "="*70)
    print("EXAMPLE 6: CONSTRAINT SENSITIVITY ANALYSIS")
    print("="*70)
    
    stocks = PortfolioDataFetcher.get_sample_data()
    fetcher = PortfolioDataFetcher(stocks)
    price_data = fetcher.fetch_data()
    mean_returns, cov_matrix, returns = fetcher.calculate_statistics(price_data)
    
    num_assets = len(stocks)
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
    
    # Test different maximum weights
    max_weights = [0.1, 0.2, 0.3, 0.4]
    
    print("\nEffect of Maximum Weight Constraint:")
    print("Max Weight | Return  | Volatility | Sharpe Ratio")
    print("-" * 50)
    
    for max_w in max_weights:
        params = {
            'lower_bounds': np.zeros(num_assets),
            'upper_bounds': np.ones(num_assets) * max_w,
            'max_assets': 5,
            'lot_size': 0.01,
            'min_invest': 0.05,
        }
        
        w_pso, _ = optimizer.optimize_pso(lambda_param=0.5, params=params)
        metrics = optimizer.calculate_portfolio_metrics(w_pso)
        
        print(f"{max_w*100:6.1f}%   | {metrics['return']*100:6.2f}% | {metrics['volatility']*100:9.2f}% | {metrics['sharpe_ratio']:9.4f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION - USAGE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_default()
    example_2_high_risk_aversion()
    example_3_aggressive_growth()
    example_4_custom_stocks()
    example_5_different_periods()
    example_6_constraint_sensitivity()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
