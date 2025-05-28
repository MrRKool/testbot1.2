import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import time
from utils.trading.price_fetcher import PriceFetcher
from utils.env_loader import check_environment, get_api_keys, get_telegram_config

def fetch_historical_data(symbol: str, start_date: str, end_date: str, timeframes: List[str] = ['1m', '15m', '1h']):
    """
    Fetch historical data for a symbol across multiple timeframes.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timeframes: List of timeframes to fetch
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Load configuration
        config = get_telegram_config()
        
        # Initialize price fetcher
        price_fetcher = PriceFetcher(config)
        
        # Convert dates to datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Fetch data for each timeframe
        for tf in timeframes:
            logging.info(f"Fetching {tf} data for {symbol} from {start_date} to {end_date}")
            
            # Convert timeframe to Bybit format
            tf_map = {
                '1m': '1',
                '15m': '15',
                '1h': '60'
            }
            bybit_tf = tf_map.get(tf)
            if not bybit_tf:
                raise ValueError(f"Unsupported timeframe: {tf}")
            
            # Calculate number of candles needed
            if tf == '1m':
                minutes_per_candle = 1
            elif tf == '15m':
                minutes_per_candle = 15
            elif tf == '1h':
                minutes_per_candle = 60
            else:
                raise ValueError(f"Unsupported timeframe: {tf}")
            
            total_minutes = (end_dt - start_dt).total_seconds() / 60
            num_candles = int(total_minutes / minutes_per_candle)
            
            # Fetch data in chunks to avoid API limits
            chunk_size = 1000  # Maximum candles per request
            all_data = []
            
            for i in range(0, num_candles, chunk_size):
                chunk_end = min(i + chunk_size, num_candles)
                logging.info(f"Fetching candles {i} to {chunk_end} for {symbol} {tf}")
                try:
                    chunk_data = price_fetcher.get_klines(
                        symbol=symbol,
                        interval=bybit_tf,
                        limit=chunk_end - i
                    )
                    if chunk_data is not None:
                        logging.info(f"Chunk {i}-{chunk_end}: {len(chunk_data)} candles fetched.")
                    else:
                        logging.warning(f"No data returned for chunk {i}-{chunk_end}")
                except Exception as e:
                    logging.error(f"Error fetching chunk {i}-{chunk_end} for {symbol} {tf}: {e}")
                    continue
                
                if chunk_data is not None and not chunk_data.empty:
                    all_data.append(chunk_data)
                else:
                    logging.warning(f"No data returned for chunk {i}-{chunk_end} for {symbol} {tf}")
                
                # Add delay to avoid rate limits
                time.sleep(1)
            
            if all_data:
                # Combine all chunks
                df = pd.concat(all_data)
                
                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]
                
                # Sort by timestamp
                df.sort_index(inplace=True)
                
                # Save to CSV
                filename = f'data/{symbol}_{tf}.csv'
                df.to_csv(filename)
                logging.info(f"Saved {len(df)} candles to {filename}")
            else:
                logging.warning(f"No data fetched for {symbol} on {tf} timeframe")
                
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        raise

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fetch_historical.log', mode='w')
        ]
    )
    
    # Define parameters
    symbol = 'BTCUSDT'
    start_date = '2024-01-01'
    end_date = '2024-01-02'  # Verkort naar 1 dag
    timeframes = ['1h']  # Alleen 1h timeframe voor test
    
    logging.info(f"Starting data fetch for {symbol} from {start_date} to {end_date}")
    logging.info(f"Timeframes: {timeframes}")
    
    try:
        # Fetch data
        fetch_historical_data(symbol, start_date, end_date, timeframes)
    except Exception as e:
        logging.error(f"Error in main: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 