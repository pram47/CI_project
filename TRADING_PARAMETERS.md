# Real Trading Parameters Used in Portfolio Optimization

## Overview
This document explains the real trading parameters fetched from Yahoo Finance and used in the portfolio optimization algorithm.

## Data Sources & Files

### Files Generated
- **trading_costs.csv** - Bid-ask spread data for each stock
- **lot_sizes.csv** - Minimum lot size (shares) for each stock

### Fetched From
- **Price Data**: Yahoo Finance (yfinance) historical prices
- **Bid-Ask Spreads**: Yahoo Finance ticker.info (current market quotes)
- **Lot Sizes**: Yahoo Finance ticker.info (assumes fractional shares = 1 share minimum)
- **Commission**: Modern brokers (Robinhood, Fidelity, etc.) = $0 commission

---

## Current Parameters

### Trading Costs Summary
```
Average Bid-Ask Spread:     1.2327%
Average Variable Cost:      0.0100% (half of bid-ask spread per transaction)
Fixed Commission Cost:      $0.00 (commission-free broker)
Average Stock Price:        $286.55
```

### Lot Size Summary
```
Minimum Lot Size:           1 share (fractional shares enabled)
Average Lot Price Point:    $286.55
Minimum Portfolio Weight:   ~0.1% per position ($100 investment in $100k portfolio)
```

### Bid-Ask Spread Distribution
Top 10 stocks with highest bid-ask spread:
```
TMUS                        0.1143%  (T-Mobile)
CSCO                        0.1044%  (Cisco)
TXN                         0.0101%  (Texas Instruments)
GILD                        0.0936%  (Gilead Sciences)
SBUX                        0.0907%  (Starbucks)
PEP                         0.0870%  (PepsiCo)
AMD                         0.0682%  (Advanced Micro Devices)
WMT                         0.0451%  (Walmart)
WM                          0.0443%  (Waste Management)
QCOM                        0.0356%  (Qualcomm)
```

---

## Constraint Parameters Used

| Constraint | Value | Description |
|-----------|-------|-------------|
| **No Short Sell** | w_i ≥ 0 | All weights non-negative |
| **Max Weight per Stock** | 12% | Each stock max 0.12 of portfolio |
| **Minimum Investment** | 5% | If holding, at least 5% or 0% |
| **Lot Size** | ~0.1% | Based on 1 share at average $286.55 price |
| **Max Active Positions** | 12 | Choose up to 12 holdings from 100 candidates |
| **Sector Cap** | 25% | No sector exceeds 25% portfolio weight |
| **Transaction Costs** | Variable | Based on real bid-ask spreads |

---

## How Costs Are Calculated

### Per-Trade Transaction Cost
```
Transaction Cost = (Fixed Commission) + (Bid-Ask Spread × Amount) × 0.5

Example for $10,000 investment:
- Fixed: $0 (commission-free)
- Bid-Ask (0.01% average): $10,000 × 0.0001 = $1
- Total cost: ~$1 per entry or exit
```

### Lot Size Impact
```
Since minimum is 1 share fractional:
- Can invest any amount ≥ $1 (fractional share)
- Portfolio weights rounded to 0.1% increments
- No "round lot" penalty on cost
```

---

## Updating Trading Parameters

### To Refresh Data
```bash
python -c "
from portfolio_optimization import PortfolioDataFetcher, TOP_100_US_LARGE_CAPS
import pandas as pd

fetcher = PortfolioDataFetcher(TOP_100_US_LARGE_CAPS)
stocks = list(fetcher.fetch_data().columns)

trading_costs = fetcher.fetch_trading_costs(stocks)
lot_sizes = fetcher.fetch_lot_sizes(stocks)

trading_costs.to_csv('trading_costs.csv')
lot_sizes.to_csv('lot_sizes.csv')

print('Data updated successfully!')
"
```

### To Use Different Broker Commissions
Edit **portfolio_optimization.py**:
```python
# Change line ~590
commission_per_trade = 1.00  # Interactive Brokers ~$1 per stock trade
# or
commission_per_trade = 0.0   # Robinhood/Fidelity = free
```

### To Use Different Minimum Lot Size
Edit **portfolio_optimization.py**:
```python
# Change line ~580
lot_size = 100  # Traditional round lot
# or
lot_size = 1    # Fractional shares (current default)
```

---

## Realistic Assumptions

### Modern US Broker Environment (as of March 2026)
- **Commission**: $0 (all major brokers are commission-free for US stocks)
- **Bid-Ask Spread**: 0.01% - 0.11% for large-cap stocks (very liquid)
- **Fractional Shares**: Enabled (can buy $1 worth of any stock)
- **Minimum Deposit**: Often $0 - $1 with most online brokers
- **Market Hours**: 9:30 AM - 4:00 PM EST (regular) + pre/after hours

---

## Example Portfolio Cost Analysis

### Scenario: $100,000 Portfolio with 12 Holdings (~$8,333 each)

**Entry Costs (Buy signal)**
```
Per position: $8,333 × 0.01% average bid-ask = $0.83
Total across 12: ~$10

Quarterly rebalancing (3-4 trades per holding):
Per position per trade: ~$0.83
Total quarterly: ~$40-50
Annual: ~$160-200
```

**Impact on Returns**
```
Before costs: 33.70% annual
Transaction costs ~0.15% annually
After costs net: 33.55% annual
```

---

## Data Quality Notes

- Bid-Ask spread data from end-of-day Yahoo Finance (not intraday average)
- Lot size assumes modern fractional share capability
- Commission assumes commission-free broker (Robinhood/Fidelity/IB fractional)
- Spreads tighten during US market open (9:30 AM - 4:00 PM EST)
- Spreads widen during after-hours trading

---

## References

- **CSV Files**: `trading_costs.csv`, `lot_sizes.csv`
- **Data Source**: Yahoo Finance yfinance
- **Broker Assumptions**: Modern commission-free brokers (2026)
- **Market Data**: US equity markets, regular hours
