import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data_loader import DataLoader

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_integrity.log')
        ]
    )

def check_data_file(file_path: Path) -> dict:
    """Check integrity of a single data file"""
    try:
        # Load data with optimized settings
        df = pd.read_csv(
            file_path,
            parse_dates=['timestamp'],
            index_col='timestamp',
            usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            dtype={
                'open': 'float32',
                'high': 'float32',
                'low': 'float32',
                'close': 'float32',
                'volume': 'float32'
            }
        )
        
        # Basic checks
        checks = {
            'file_exists': True,
            'row_count': len(df),
            'has_duplicates': df.index.duplicated().any(),
            'has_nulls': df.isnull().any().any(),
            'has_zeros': (df == 0).any().any(),
            'has_negative': (df < 0).any().any(),
            'timestamp_gaps': False,
            'price_anomalies': False,
            'volume_anomalies': False
        }
        
        # Check for timestamp gaps
        expected_timestamps = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='1T'  # 1 minute frequency
        )
        checks['timestamp_gaps'] = len(expected_timestamps) != len(df)
        
        # Check for price anomalies
        price_changes = df['close'].pct_change().abs()
        checks['price_anomalies'] = (price_changes > 0.1).any()  # 10% price change threshold
        
        # Check for volume anomalies
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        checks['volume_anomalies'] = (df['volume'] > volume_mean + 3 * volume_std).any()
        
        return checks
        
    except Exception as e:
        logging.error(f"Error checking file {file_path}: {str(e)}")
        return {
            'file_exists': False,
            'error': str(e)
        }

def check_data_integrity():
    """Main function to check data integrity"""
    try:
        # Setup logging
        setup_logging()
        logging.info("Starting data integrity check")
        
        # Get data directory
        data_dir = Path('data')
        if not data_dir.exists():
            logging.error("Data directory not found")
            return
        
        # Find all data files
        data_files = []
        for symbol_dir in data_dir.iterdir():
            if symbol_dir.is_dir():
                for file in symbol_dir.glob('*_data_*.csv'):
                    data_files.append(file)
        
        if not data_files:
            logging.error("No data files found")
            return
        
        logging.info(f"Found {len(data_files)} data files to check")
        
        # Check files in parallel
        with ProcessPoolExecutor(max_workers=min(len(data_files), multiprocessing.cpu_count())) as executor:
            results = {}
            for file_path, checks in tqdm(
                zip(data_files, executor.map(check_data_file, data_files)),
                total=len(data_files),
                desc="Checking files"
            ):
                results[str(file_path)] = checks
        
        # Analyze results
        issues_found = False
        for file_path, checks in results.items():
            if not checks['file_exists']:
                logging.error(f"File not found: {file_path}")
                issues_found = True
                continue
            
            if checks['has_duplicates']:
                logging.warning(f"File has duplicate timestamps: {file_path}")
                issues_found = True
            
            if checks['has_nulls']:
                logging.warning(f"File has null values: {file_path}")
                issues_found = True
            
            if checks['has_zeros']:
                logging.warning(f"File has zero values: {file_path}")
                issues_found = True
            
            if checks['has_negative']:
                logging.warning(f"File has negative values: {file_path}")
                issues_found = True
            
            if checks['timestamp_gaps']:
                logging.warning(f"File has timestamp gaps: {file_path}")
                issues_found = True
            
            if checks['price_anomalies']:
                logging.warning(f"File has price anomalies: {file_path}")
                issues_found = True
            
            if checks['volume_anomalies']:
                logging.warning(f"File has volume anomalies: {file_path}")
                issues_found = True
        
        if not issues_found:
            logging.info("No issues found in data files")
        else:
            logging.warning("Issues found in data files. Check the log for details.")
        
    except Exception as e:
        logging.error(f"Error in data integrity check: {str(e)}")

if __name__ == "__main__":
    check_data_integrity() 