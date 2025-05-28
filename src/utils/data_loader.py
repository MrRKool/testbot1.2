import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import mmap
import os
import yaml
import time
import threading
from queue import Queue
import asyncio
import aiohttp
import json

class DataLoader:
    def __init__(self, config: dict):
        """Initialize the data loader with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.exchange_config = config['exchange']
        self.cache_config = config.get('cache', {})
        self.data_cache = {}
        self.last_update = {}
        self.update_queue = Queue()
        self.is_running = False
        self.update_thread = None
        self.session = None
        
        # Initialize cache for each symbol
        for symbol in self.config['symbols'].keys():
            self.data_cache[symbol] = {}
            self.last_update[symbol] = {}
            for timeframe in self.config['timeframes']:
                self.data_cache[symbol][timeframe] = None
                self.last_update[symbol][timeframe] = datetime.min
        
        self.logger.info("Data loader initialized")

    async def initialize(self):
        """Initialize async session and start update thread."""
        try:
            self.session = aiohttp.ClientSession()
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_worker)
            self.update_thread.daemon = True
            self.update_thread.start()
            self.logger.info("Data loader started")
        except Exception as e:
            self.logger.error(f"Error initializing data loader: {str(e)}")
            raise

    async def close(self):
        """Close async session and stop update thread."""
        try:
            self.is_running = False
            if self.update_thread:
                self.update_thread.join()
            if self.session:
                await self.session.close()
            self.logger.info("Data loader stopped")
        except Exception as e:
            self.logger.error(f"Error closing data loader: {str(e)}")

    def get_latest_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get latest data for symbol and timeframe."""
        try:
            if symbol not in self.data_cache or timeframe not in self.data_cache[symbol]:
                return None
            
            data = self.data_cache[symbol][timeframe]
            if data is None or data.empty:
                return None
            
            # Check if data is recent enough
            last_update = self.last_update[symbol][timeframe]
            if datetime.now() - last_update > self._get_timeframe_delta(timeframe) * 2:
                self.update_queue.put((symbol, timeframe))
                return None
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {str(e)}")
            return None

    def _update_worker(self):
        """Worker thread for updating data."""
        while self.is_running:
            try:
                symbol, timeframe = self.update_queue.get(timeout=1)
                asyncio.run(self._update_data(symbol, timeframe))
                self.update_queue.task_done()
            except Exception as e:
                if not isinstance(e, TimeoutError):
                    self.logger.error(f"Error in update worker: {str(e)}")
                time.sleep(1)

    async def _update_data(self, symbol: str, timeframe: str):
        """Update data for symbol and timeframe."""
        try:
            # Get data from exchange
            data = await self._fetch_data(symbol, timeframe)
            if data is not None and not data.empty:
                self.data_cache[symbol][timeframe] = data
                self.last_update[symbol][timeframe] = datetime.now()
                self.logger.debug(f"Updated data for {symbol} {timeframe}")
        except Exception as e:
            self.logger.error(f"Error updating data: {str(e)}")

    async def _fetch_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch data from exchange."""
        try:
            url = f"{self.exchange_config['api']['base_url']}/market/kline"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': 1000
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['ret_code'] == 0:
                        return self._process_exchange_data(data['result'])
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return None

    def _process_exchange_data(self, data: Dict) -> pd.DataFrame:
        """Process raw exchange data into DataFrame."""
        try:
            df = pd.DataFrame(data['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing exchange data: {str(e)}")
            return pd.DataFrame()

    def _get_timeframe_delta(self, timeframe: str) -> timedelta:
        """Convert timeframe string to timedelta."""
        try:
            unit = timeframe[-1]
            value = int(timeframe[:-1])
            if unit == 'm':
                return timedelta(minutes=value)
            elif unit == 'h':
                return timedelta(hours=value)
            elif unit == 'd':
                return timedelta(days=value)
            return timedelta(minutes=1)
        except Exception as e:
            self.logger.error(f"Error converting timeframe: {str(e)}")
            return timedelta(minutes=1)

    @lru_cache(maxsize=100)
    def load_historical_data(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting."""
        try:
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
            
            # Fetch data in chunks
            all_data = []
            current_ts = start_ts
            
            while current_ts < end_ts:
                url = f"{self.exchange_config['api']['base_url']}/market/kline"
                params = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'from': current_ts,
                    'limit': 1000
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data['ret_code'] == 0:
                        all_data.extend(data['result']['list'])
                        current_ts = int(data['result']['list'][-1][0]) + 1
                    else:
                        break
                else:
                    break
            
            # Process data
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            return pd.DataFrame()

    def _get_mmap_path(self, symbol: str, timeframe: str) -> Path:
        """Get memory map file path"""
        return self.data_dir / symbol / f"{symbol.lower()}_data_{timeframe}.mmap"
    
    def _create_mmap(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Create memory map for faster data access"""
        try:
            mmap_path = self._get_mmap_path(symbol, timeframe)
            mmap_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame to pickle with protocol 4 for better performance
            df.to_pickle(mmap_path, protocol=4)
            
            # Create memory map
            with open(mmap_path, 'rb') as f:
                self._mmap_cache[f"{symbol}_{timeframe}"] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
        except Exception as e:
            self.logger.error(f"Error creating memory map: {str(e)}")
    
    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data with parallel processing"""
        try:
            # Define timeframe mapping
            timeframe_map = {
                '1m': '1T',
                '5m': '5T',
                '15m': '15T',
                '30m': '30T',
                '1h': '1H',
                '4h': '4H',
                '1d': '1D'
            }
            
            if timeframe not in timeframe_map:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return df
            
            # Split data into chunks for parallel processing
            chunk_size = len(df) // os.cpu_count()
            chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            
            def process_chunk(chunk):
                return chunk.resample(timeframe_map[timeframe]).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
            
            # Process chunks in parallel
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
                results = [future.result() for future in as_completed(futures)]
            
            # Combine results
            resampled = pd.concat(results)
            resampled = resampled.sort_index()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {str(e)}", exc_info=True)
            return df
    
    def verify_resampled_data(self, original_df: pd.DataFrame, resampled_df: pd.DataFrame) -> bool:
        """Verify resampled data with parallel processing"""
        try:
            # Check row count
            if len(original_df) != len(resampled_df):
                self.logger.error(f"Row count mismatch: original={len(original_df)}, resampled={len(resampled_df)}")
                return False
            
            # Check timestamps
            if not original_df.index.equals(resampled_df.index):
                self.logger.error("Timestamp mismatch between original and resampled data")
                return False
            
            # Split data into chunks for parallel verification
            chunk_size = len(original_df) // os.cpu_count()
            chunks = [(original_df[i:i + chunk_size], resampled_df[i:i + chunk_size]) 
                     for i in range(0, len(original_df), chunk_size)]
            
            def verify_chunk(chunk_pair):
                orig_chunk, resamp_chunk = chunk_pair
                tolerance = 1e-6
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if not np.allclose(orig_chunk[col], resamp_chunk[col], rtol=tolerance, atol=tolerance):
                        return False
                return True
            
            # Verify chunks in parallel
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(verify_chunk, chunk) for chunk in chunks]
                results = [future.result() for future in as_completed(futures)]
            
            return all(results)
            
        except Exception as e:
            self.logger.error(f"Error verifying resampled data: {str(e)}", exc_info=True)
            return False

    def clear_cache(self):
        """Clear the data cache"""
        self._data_cache.clear()
        self._mmap_cache.clear()
        self.load_historical_data.cache_clear()
        
        # Close memory maps
        for mmap_obj in self._mmap_cache.values():
            try:
                mmap_obj.close()
            except Exception:
                pass 