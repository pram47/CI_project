"""Build a reusable local market-data cache for the portfolio optimization project."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf


WIKIPEDIA_SP500_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
PRICE_CACHE_PATH = 'sp500_daily_prices_10y.csv'
METADATA_CACHE_PATH = 'sp500_asset_metadata.csv'
TRADING_COSTS_CACHE_PATH = 'sp500_trading_costs.csv'
LOT_SIZES_CACHE_PATH = 'sp500_lot_sizes.csv'
TICKERS_CACHE_PATH = 'sp500_tickers.csv'


def normalize_ticker_symbol(ticker_symbol):
    return str(ticker_symbol).strip().upper().replace('.', '-')


def fetch_sp500_constituents():
    """Load the current S&P 500 constituents from Wikipedia."""
    request = Request(
        WIKIPEDIA_SP500_URL,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    with urlopen(request) as response:
        html_text = response.read().decode('utf-8')

    constituents_table = pd.read_html(StringIO(html_text), match='Symbol')[0]
    constituents_table['Symbol'] = constituents_table['Symbol'].map(normalize_ticker_symbol)
    constituents_table = constituents_table.drop_duplicates(subset='Symbol').sort_values('Symbol').reset_index(drop=True)
    return constituents_table


def extract_price_matrix(raw_data):
    """Select adjusted close when available, otherwise close."""
    if raw_data.empty:
        raise ValueError('No price data was returned by yfinance.')

    if isinstance(raw_data.columns, pd.MultiIndex):
        price_field = 'Adj Close' if 'Adj Close' in raw_data.columns.get_level_values(0) else 'Close'
        price_data = raw_data[price_field].copy()
    else:
        price_field = 'Adj Close' if 'Adj Close' in raw_data.columns else 'Close'
        price_data = raw_data[[price_field]].copy()

    if isinstance(price_data, pd.Series):
        price_data = price_data.to_frame()

    price_data.columns = [normalize_ticker_symbol(column) for column in price_data.columns]
    price_data.index = pd.to_datetime(price_data.index).tz_localize(None)
    price_data = price_data[~price_data.index.duplicated(keep='last')].sort_index()
    price_data = price_data.dropna(axis=1, how='all')
    return price_data


def clean_price_matrix(price_data, ticker_symbols):
    """Reorder columns and keep a dense, analyst-friendly daily price panel."""
    normalized_symbols = [normalize_ticker_symbol(ticker_symbol) for ticker_symbol in ticker_symbols]
    available_symbols = [ticker_symbol for ticker_symbol in normalized_symbols if ticker_symbol in price_data.columns]
    cleaned_data = price_data.reindex(columns=available_symbols)
    cleaned_data = cleaned_data.dropna(axis=1, how='all')
    cleaned_data.index.name = 'Date'
    return cleaned_data


def fetch_ticker_snapshot(ticker_symbol):
    """Fetch reusable sector and execution inputs for one ticker."""
    fallback_price = 100.0

    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
    except Exception:
        info = {}

    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or fallback_price
    bid = info.get('bid') or current_price * 0.999
    ask = info.get('ask') or current_price * 1.001

    current_price = max(float(current_price), 0.01)
    bid = max(float(bid), 0.01)
    ask = max(float(ask), bid)

    return {
        'ticker': ticker_symbol,
        'sector': info.get('sector') or info.get('sectorDisp') or info.get('industry') or 'Unknown',
        'current_price': current_price,
        'bid': bid,
        'ask': ask,
        'spread_pct': max((ask - bid) / current_price if current_price > 0 else 0.001, 0.0001),
        'lot_size': max(int(info.get('lotSize') or 1), 1),
        'price_point': current_price
    }


def fetch_snapshot_frame(ticker_symbols, max_workers=16):
    """Fetch per-ticker snapshot data in parallel."""
    snapshot_rows = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(ticker_symbols))) as executor:
        futures = {
            executor.submit(fetch_ticker_snapshot, ticker_symbol): ticker_symbol
            for ticker_symbol in ticker_symbols
        }
        for future in as_completed(futures):
            snapshot_rows.append(future.result())

    snapshot_frame = pd.DataFrame(snapshot_rows)
    snapshot_frame['ticker'] = snapshot_frame['ticker'].map(normalize_ticker_symbol)
    snapshot_frame = snapshot_frame.drop_duplicates(subset='ticker', keep='last').sort_values('ticker').reset_index(drop=True)
    return snapshot_frame


def main():
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365 * 10)

    print('Loading S&P 500 constituents...')
    constituents = fetch_sp500_constituents()
    ticker_symbols = constituents['Symbol'].tolist()
    print(f'Found {len(ticker_symbols)} current S&P 500 tickers')

    print('Downloading 10 years of daily price history from yfinance...')
    raw_data = yf.download(
        ticker_symbols,
        start=start_date,
        end=end_date + timedelta(days=1),
        interval='1d',
        progress=False,
        auto_adjust=False,
        threads=True
    )
    price_data = clean_price_matrix(extract_price_matrix(raw_data), ticker_symbols)
    price_data.reset_index().to_csv(PRICE_CACHE_PATH, index=False)
    print(f'Saved {PRICE_CACHE_PATH} with shape {price_data.shape}')

    print('Fetching reusable sector/trading metadata...')
    snapshot_frame = fetch_snapshot_frame(list(price_data.columns))
    snapshot_frame[['ticker', 'sector']].to_csv(METADATA_CACHE_PATH, index=False)
    snapshot_frame[['ticker', 'current_price', 'bid', 'ask', 'spread_pct']].to_csv(TRADING_COSTS_CACHE_PATH, index=False)
    snapshot_frame[['ticker', 'lot_size', 'price_point']].to_csv(LOT_SIZES_CACHE_PATH, index=False)

    constituents = constituents[constituents['Symbol'].isin(price_data.columns)].copy()
    constituents.to_csv(TICKERS_CACHE_PATH, index=False)
    print(f'Saved {METADATA_CACHE_PATH}, {TRADING_COSTS_CACHE_PATH}, {LOT_SIZES_CACHE_PATH}, and {TICKERS_CACHE_PATH}')

    completeness = price_data.notna().mean().sort_values(ascending=False)
    print(f'Tickers with usable price history: {price_data.shape[1]}')
    print(f'Daily rows saved: {price_data.shape[0]}')
    print(f'Average completeness: {completeness.mean() * 100:.2f}%')


if __name__ == '__main__':
    main()