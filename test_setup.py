"""
Test script to verify portfolio optimization implementation
Run this to validate the setup and all components work correctly
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def test_imports():
    """Test if all required libraries are installed"""
    print("Testing imports...")
    try:
        import yfinance as yf
        print("  ✓ yfinance")
    except ImportError:
        print("  ✗ yfinance - NOT INSTALLED")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("  ✓ matplotlib")
    except ImportError:
        print("  ✗ matplotlib - NOT INSTALLED")
        return False
    
    try:
        import scipy
        print("  ✓ scipy")
    except ImportError:
        print("  ✗ scipy - NOT INSTALLED")
        return False
    
    try:
        import pyswarms
        print("  ✓ pyswarms")
    except ImportError:
        print("  ✗ pyswarms - NOT INSTALLED")
        return False
    
    print("All imports successful!\n")
    return True


def test_data_fetching():
    """Test data fetching from yfinance"""
    print("Testing data fetching from yfinance...")
    try:
        from portfolio_optimization import PortfolioDataFetcher
        
        # Test with 3 stocks
        fetcher = PortfolioDataFetcher(['AAPL', 'MSFT', 'GOOGL'])
        data = fetcher.fetch_data()
        
        print(f"  ✓ Successfully fetched data")
        print(f"    Shape: {data.shape}")
        print(f"    Columns: {list(data.columns)}")
        print(f"    Date range: {data.index[0]} to {data.index[-1]}\n")
        
        return True
    except Exception as e:
        print(f"  ✗ Data fetching failed: {e}\n")
        return False


def test_statistics():
    """Test statistics calculation"""
    print("Testing statistics calculation...")
    try:
        from portfolio_optimization import PortfolioDataFetcher
        
        fetcher = PortfolioDataFetcher(['AAPL', 'MSFT', 'GOOGL'])
        data = fetcher.fetch_data()
        mean_returns, cov_matrix, returns = fetcher.calculate_statistics(data)
        
        print(f"  ✓ Statistics calculated successfully")
        print(f"    Mean returns shape: {mean_returns.shape}")
        print(f"    Covariance matrix shape: {cov_matrix.shape}")
        print(f"    Sample mean return (AAPL): {mean_returns.iloc[0]*100:.2f}%\n")
        
        return True
    except Exception as e:
        print(f"  ✗ Statistics calculation failed: {e}\n")
        return False


def test_optimization():
    """Test portfolio optimization"""
    print("Testing portfolio optimization...")
    try:
        from portfolio_optimization import PortfolioDataFetcher, PortfolioOptimizer
        
        fetcher = PortfolioDataFetcher(['AAPL', 'MSFT', 'GOOGL'])
        data = fetcher.fetch_data()
        mean_returns, cov_matrix, returns = fetcher.calculate_statistics(data)
        
        num_assets = len(['AAPL', 'MSFT', 'GOOGL'])
        optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
        
        # Test unconstrained optimization
        w_unconstrained, cost = optimizer.optimize_unconstrained(lambda_param=0.5)
        
        print(f"  ✓ Unconstrained optimization successful")
        print(f"    Weights sum: {np.sum(w_unconstrained):.6f}")
        print(f"    Weights: {w_unconstrained}")
        
        # Test metrics calculation
        metrics = optimizer.calculate_portfolio_metrics(w_unconstrained)
        print(f"    Portfolio return: {metrics['return']*100:.2f}%")
        print(f"    Portfolio volatility: {metrics['volatility']*100:.2f}%")
        print(f"    Sharpe ratio: {metrics['sharpe_ratio']:.4f}\n")
        
        return True
    except Exception as e:
        print(f"  ✗ Optimization failed: {e}\n")
        return False


def test_pso_optimization():
    """Test PSO-based optimization"""
    print("Testing PSO optimization with constraints...")
    try:
        from portfolio_optimization import PortfolioDataFetcher, PortfolioOptimizer
        
        fetcher = PortfolioDataFetcher(['AAPL', 'MSFT'])  # Smaller dataset for faster test
        data = fetcher.fetch_data()
        mean_returns, cov_matrix, returns = fetcher.calculate_statistics(data)
        
        num_assets = 2
        optimizer = PortfolioOptimizer(mean_returns, cov_matrix, num_assets)
        
        # Test with constraints
        params = {
            'lower_bounds': np.zeros(num_assets),
            'upper_bounds': np.ones(num_assets) * 0.5,
            'max_assets': 2,
        }
        
        w_pso, cost = optimizer.optimize_pso(
            lambda_param=0.5,
            params=params,
            n_particles=10,
            n_iterations=20
        )
        
        print(f"  ✓ PSO optimization successful")
        print(f"    Weights sum: {np.sum(w_pso):.6f}")
        print(f"    Weights: {w_pso}")
        print(f"    All weights non-negative: {(w_pso >= -1e-6).all()}\n")
        
        return True
    except Exception as e:
        print(f"  ✗ PSO optimization failed: {e}\n")
        return False


def test_penalty_function():
    """Test penalty function"""
    print("Testing penalty function...")
    try:
        from portfolio_optimization import PortfolioDataFetcher, PortfolioOptimizer
        
        fetcher = PortfolioDataFetcher(['AAPL', 'MSFT'])
        data = fetcher.fetch_data()
        mean_returns, cov_matrix, returns = fetcher.calculate_statistics(data)
        
        optimizer = PortfolioOptimizer(mean_returns, cov_matrix, 2)
        
        # Test valid weights
        valid_weights = np.array([0.5, 0.5])
        params = {'lower_bounds': np.zeros(2), 'upper_bounds': np.ones(2)}
        penalty_valid = optimizer.penalty_function(valid_weights, params)
        
        # Test invalid weights (short selling)
        invalid_weights = np.array([-0.2, 1.2])
        penalty_invalid = optimizer.penalty_function(invalid_weights, params)
        
        print(f"  ✓ Penalty function working")
        print(f"    Penalty for valid weights [0.5, 0.5]: {penalty_valid:.2f}")
        print(f"    Penalty for invalid weights [-0.2, 1.2]: {penalty_invalid:.2f}\n")
        
        return penalty_invalid > penalty_valid
    except Exception as e:
        print(f"  ✗ Penalty function test failed: {e}\n")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("PORTFOLIO OPTIMIZATION - VERIFICATION TESTS")
    print("="*70 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Fetching", test_data_fetching),
        ("Statistics", test_statistics),
        ("Basic Optimization", test_optimization),
        ("PSO Optimization", test_pso_optimization),
        ("Penalty Function", test_penalty_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}\n")
            results.append((test_name, False))
    
    # Print summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    print("="*70)
    print(f"Total: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    if passed == total:
        print("✓ All tests passed! The environment is ready.\n")
        print("Next steps:")
        print("  1. Run: python portfolio_optimization.py")
        print("  2. Or explore examples: python examples.py")
        return True
    else:
        print("✗ Some tests failed. Please check your installation.\n")
        print("To install dependencies:")
        print("  pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
