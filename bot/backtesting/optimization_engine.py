import logging
import pandas as pd
import numpy as np
from pathlib import Path
import json
import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Tuple
import time
from datetime import datetime, timedelta
import os
import signal
import sys

from bot.backtesting.backtest_engine import BacktestEngine
from bot.strategies.example_strategy import ExampleStrategy

class OptimizationEngine:
    def __init__(self, config: Dict):
        self.config = config
        self.log = logging.getLogger(__name__)
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Stats bestand voor combinaties
        self.stats_file = self.results_dir / 'combination_stats.json'
        self.combination_stats = self._load_stats()
        
        # Parameter ranges voor optimalisatie
        self.param_ranges = {
            'rsi': {
                'period': range(7, 22, 2),
                'overbought': range(65, 81, 5),
                'oversold': range(15, 36, 5)
            },
            'macd': {
                'fast_period': range(8, 17, 2),
                'slow_period': range(21, 36, 3),
                'signal_period': range(7, 14, 2)
            },
            'bollinger': {
                'period': range(15, 31, 5),
                'std_dev': [1.5, 2.0, 2.5]
            },
            'sma': {
                'fast_period': range(5, 21, 5),
                'slow_period': range(20, 61, 10)
            },
            'risk': {
                'position_size': [0.01, 0.02, 0.03, 0.05],
                'stop_loss': [0.01, 0.02, 0.03, 0.05],
                'take_profit': [0.02, 0.03, 0.05, 0.08]
            }
        }
        
        # Timeframes voor optimalisatie
        self.timeframes = ['1h', '4h', '1d']
        
        # Setup signal handlers voor graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
    def _load_stats(self) -> Dict:
        """Laad bestaande statistieken of maak nieuwe aan."""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        return {
            'total_combinations': 0,
            'successful_combinations': 0,
            'combinations': {}
        }
    
    def _save_stats(self):
        """Sla statistieken op."""
        with open(self.stats_file, 'w') as f:
            json.dump(self.combination_stats, f, indent=4)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        self.log.info("Shutdown signal received. Saving stats and exiting...")
        self._save_stats()
        sys.exit(0)
    
    def _get_combination_key(self, params: Dict) -> str:
        """Genereer een unieke key voor een parameter combinatie."""
        return json.dumps(params, sort_keys=True)
    
    def _update_stats(self, params: Dict, success: bool):
        """Update statistieken voor een combinatie."""
        key = self._get_combination_key(params)
        
        if key not in self.combination_stats['combinations']:
            self.combination_stats['combinations'][key] = {
                'total_tests': 0,
                'successful_tests': 0,
                'success_rate': 0.0,
                'last_tested': datetime.now().isoformat()
            }
        
        stats = self.combination_stats['combinations'][key]
        stats['total_tests'] += 1
        if success:
            stats['successful_tests'] += 1
        stats['success_rate'] = stats['successful_tests'] / stats['total_tests']
        stats['last_tested'] = datetime.now().isoformat()
        
        self.combination_stats['total_combinations'] = len(self.combination_stats['combinations'])
        self.combination_stats['successful_combinations'] = sum(
            1 for c in self.combination_stats['combinations'].values()
            if c['success_rate'] > 0
        )
        
        self._save_stats()
    
    def generate_parameter_combinations(self) -> List[Dict]:
        """Genereer alle mogelijke parameter combinaties."""
        param_names = []
        param_values = []
        
        for category, params in self.param_ranges.items():
            for param, values in params.items():
                param_names.append(f"{category}_{param}")
                param_values.append(values)
        
        combinations = []
        for values in itertools.product(*param_values):
            params = dict(zip(param_names, values))
            combinations.append(params)
            
        return combinations
    
    def load_historical_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Laad historische data voor een specifiek timeframe."""
        try:
            data_path = f"data/{symbol}/{symbol.lower()}_data_{timeframe}.csv"
            df = pd.read_csv(data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.log.error(f"Error loading data for {symbol} {timeframe}: {str(e)}")
            raise
    
    def run_backtest(self, params: Dict, data: pd.DataFrame) -> Dict:
        """Run een backtest met gegeven parameters."""
        try:
            strategy = ExampleStrategy(params)
            engine = BacktestEngine(self.config)
            
            results = engine.run_backtest(
                data=data,
                strategy=strategy,
                initial_capital=self.config['initial_capital']
            )
            
            return {
                'params': params,
                'results': results
            }
        except Exception as e:
            self.log.error(f"Error in backtest: {str(e)}")
            return None
    
    def evaluate_results(self, results: Dict) -> bool:
        """Evalueer of de resultaten voldoen aan de minimale eisen."""
        if not results:
            return False
            
        metrics = results['results']['metrics']
        
        success = (
            metrics['sharpe_ratio'] >= self.config['min_sharpe'] and
            metrics['profit_factor'] >= self.config['min_profit_factor'] and
            metrics['win_rate'] >= self.config['min_win_rate']
        )
        
        # Update stats
        self._update_stats(results['params'], success)
        
        return success
    
    def save_results(self, results: Dict, timeframe: str):
        """Sla succesvolle resultaten op."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results_{timeframe}_{timestamp}.json"
        
        with open(self.results_dir / filename, 'w') as f:
            json.dump(results, f, indent=4)
    
    def optimize_timeframe(self, symbol: str, timeframe: str) -> List[Dict]:
        """Optimaliseer voor een specifiek timeframe."""
        self.log.info(f"Starting optimization for {timeframe}")
        
        # Laad data
        data = self.load_historical_data(symbol, timeframe)
        
        # Genereer parameter combinaties
        combinations = self.generate_parameter_combinations()
        
        # Run backtests in parallel
        successful_results = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(self.run_backtest, params, data)
                for params in combinations
            ]
            
            for future in futures:
                results = future.result()
                if results and self.evaluate_results(results):
                    successful_results.append(results)
                    self.save_results(results, timeframe)
        
        return successful_results
    
    def run_continuous_optimization(self, symbol: str):
        """Run continue optimalisatie oneindig."""
        while True:
            try:
                self.log.info("Starting new optimization cycle")
                
                for timeframe in self.timeframes:
                    try:
                        results = self.optimize_timeframe(symbol, timeframe)
                        self.log.info(f"Found {len(results)} successful combinations for {timeframe}")
                        
                        # Log statistieken
                        self.log.info(f"Total combinations tested: {self.combination_stats['total_combinations']}")
                        self.log.info(f"Successful combinations: {self.combination_stats['successful_combinations']}")
                        
                    except Exception as e:
                        self.log.error(f"Error optimizing {timeframe}: {str(e)}")
                        continue
                
                # Wacht 5 minuten voor de volgende cyclus
                time.sleep(300)
                
            except Exception as e:
                self.log.error(f"Critical error in optimization cycle: {str(e)}")
                self.log.info("Restarting in 60 seconds...")
                time.sleep(60)
                continue 