import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, lru_cache
import ccxt
from tqdm import tqdm
import time
import json
from typing import Dict, List, Tuple, Optional
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/data_download.log')
    ]
)

logger = logging.getLogger(__name__)

def get_exchange() -> ccxt.Exchange:
    """Get exchange instance"""
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True
        }
    })

def download_with_retry(func, max_retries: int = 3, delay: int = 1) -> Optional[pd.DataFrame]:
    """Retry mechanism for downloads"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(delay * (attempt + 1))
    return None

def download_chunk(exchange: ccxt.Exchange, symbol: str, timeframe: str, 
                  start_time: int, end_time: int) -> pd.DataFrame:
    """Download a chunk of historical data"""
    def _download():
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=start_time,
            limit=1000
        )
        
        if not ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Optimize data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype('float32')
        
        return df
    
    return download_with_retry(_download)

def validate_data(df: pd.DataFrame) -> bool:
    """Validate downloaded data"""
    try:
        if df.empty:
            return False
            
        # Check for required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return False
            
        # Check for null values
        if df[required_cols].isnull().any().any():
            return False
            
        # Check for negative values
        if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            return False
            
        # Check for price consistency
        if not (df['high'] >= df['low']).all():
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        return False

def download_historical_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical data for a symbol and timeframe"""
    try:
        # Initialize exchange
        exchange = get_exchange()
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # Calculate chunk size based on timeframe
        if timeframe == '1m':
            chunk_size = 1000 * 60 * 1000  # 1000 minutes
        elif timeframe == '5m':
            chunk_size = 1000 * 5 * 60 * 1000  # 5000 minutes
        elif timeframe == '15m':
            chunk_size = 1000 * 15 * 60 * 1000  # 15000 minutes
        elif timeframe == '1h':
            chunk_size = 1000 * 60 * 60 * 1000  # 1000 hours
        elif timeframe == '4h':
            chunk_size = 1000 * 4 * 60 * 60 * 1000  # 4000 hours
        elif timeframe == '1d':
            chunk_size = 1000 * 24 * 60 * 60 * 1000  # 1000 days
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Download data in chunks
        all_data = []
        current_timestamp = start_timestamp
        
        with tqdm(total=(end_timestamp - start_timestamp) // chunk_size + 1, 
                 desc=f"Downloading {symbol} {timeframe}") as pbar:
            while current_timestamp < end_timestamp:
                chunk_end = min(current_timestamp + chunk_size, end_timestamp)
                df = download_chunk(exchange, symbol, timeframe, current_timestamp, chunk_end)
                if not df.empty:
                    all_data.append(df)
                current_timestamp = chunk_end
                pbar.update(1)
                time.sleep(1)  # Rate limiting
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all chunks
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.drop_duplicates(subset=['timestamp'])
        final_df = final_df.sort_values('timestamp')
        final_df.set_index('timestamp', inplace=True)
        
        return final_df
        
    except Exception as e:
        logger.error(f"Error downloading historical data: {str(e)}")
        return pd.DataFrame()

def save_data(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save data with memory mapping and compression"""
    try:
        # Create data directory
        data_dir = Path('data') / symbol.lower()
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with correct index and headers
        file_path = data_dir / f"{symbol.lower()}_data_{timeframe}.csv"
        df.to_csv(
            file_path,
            index=True,   # Save timestamp as index
            header=True,  # Always write column headers
            float_format='%.8f',
            compression='infer'
        )
        
        # Create memory map for faster access
        mmap_path = file_path.with_suffix('.mmap')
        df.to_pickle(mmap_path, protocol=4)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'rows': len(df),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(file_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving data for {symbol} {timeframe}: {str(e)}")

def main():
    """Main function to download 5 years of historical data"""
    try:
        # Setup logging
        logger.info("Starting 5-year data download")
        
        # Define parameters
        symbol = 'BTCUSDT'
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        logger.info(f"Downloading data from {start_date} to {end_date}")
        logger.info(f"Timeframes: {timeframes}")
        
        # Download data for each timeframe
        for timeframe in timeframes:
            logger.info(f"Downloading {symbol} {timeframe}")
            
            # Download historical data
            df = download_historical_data(symbol, timeframe, start_date, end_date)
            
            if not df.empty:
                # Save data
                save_data(df, symbol, timeframe)
                logger.info(f"Downloaded {len(df)} candles for {symbol} {timeframe}")
            else:
                logger.error(f"Failed to download data for {symbol} {timeframe}")
        
        logger.info("Data download completed")
        
    except Exception as e:
        logger.error(f"Error in data download: {str(e)}")

if __name__ == "__main__":
    main() 