#!/usr/bin/env python3
"""
Script for optimizing trading parameters using Bayesian optimization.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.bot import TradingBot
from src.utils.logger import setup_logger
from src.utils.performance_analyzer import PerformanceAnalyzer

# Configure logging
logger = setup_logger("optimize", "logs/optimize.log")

class BayesianOptimizer:
    """Bayesian optimizer for trading parameters."""

    def __init__(self, param_ranges, n_initial_points=5):
        """Initialize optimizer.

        Args:
            param_ranges (dict): Dictionary of parameter ranges
            n_initial_points (int): Number of initial random points
        """
        self.param_ranges = param_ranges
        self.n_initial_points = n_initial_points
        self.scaler = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42,
        )
        self.X = []
        self.y = []

    def _normalize_params(self, params):
        """Normalize parameters to [0, 1] range."""
        normalized = {}
        for param, value in params.items():
            param_range = self.param_ranges[param]
            normalized[param] = (value - param_range[0]) / (param_range[1] - param_range[0])
        return normalized

    def _denormalize_params(self, normalized_params):
        """Denormalize parameters from [0, 1] range."""
        params = {}
        for param, value in normalized_params.items():
            param_range = self.param_ranges[param]
            params[param] = value * (param_range[1] - param_range[0]) + param_range[0]
        return params

    def _acquisition_function(self, x, xi=0.01):
        """Upper confidence bound acquisition function."""
        mean, std = self.gp.predict(x.reshape(1, -1), return_std=True)
        return mean + xi * std

    def suggest_params(self):
        """Suggest next parameters to try."""
        if len(self.X) < self.n_initial_points:
            # Random sampling for initial points
            params = {}
            for param, (min_val, max_val) in self.param_ranges.items():
                params[param] = np.random.uniform(min_val, max_val)
            return params

        # Convert parameters to array
        X = np.array([list(self._normalize_params(p).values()) for p in self.X])
        y = np.array(self.y)

        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # Fit GP
        self.gp.fit(X_scaled, y)

        # Generate random points
        n_samples = 1000
        X_random = np.random.uniform(0, 1, (n_samples, len(self.param_ranges)))
        X_random_scaled = self.scaler.transform(X_random)

        # Calculate acquisition function values
        acq_values = self._acquisition_function(X_random_scaled)

        # Find best point
        best_idx = np.argmax(acq_values)
        best_point = X_random[best_idx]

        # Convert back to parameter space
        params = {}
        for i, param in enumerate(self.param_ranges.keys()):
            params[param] = best_point[i]

        return self._denormalize_params(params)

    def update(self, params, score):
        """Update optimizer with new results."""
        self.X.append(params)
        self.y.append(score)

def objective_function(params, bot, analyzer):
    """Objective function for optimization."""
    try:
        # Update bot parameters
        bot.update_params(params)

        # Run backtest
        results = bot.run_backtest()

        # Analyze results
        metrics = analyzer.analyze_trades(results)

        # Calculate objective score (e.g., Sharpe ratio)
        score = metrics["sharpe_ratio"]

        return score
    except Exception as e:
        logger.error(f"Error in objective function: {e}")
        return -np.inf

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize trading parameters")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Trading symbols to use",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/optimization",
        help="Output directory for results",
    )
    return parser.parse_args()

def main():
    """Main function to run optimization."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load configuration
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Initialize bot and analyzer
        bot = TradingBot(config_path=args.config, symbols=args.symbols)
        analyzer = PerformanceAnalyzer(config)

        # Define parameter ranges
        param_ranges = {
            "rsi_period": (10, 30),
            "rsi_overbought": (65, 85),
            "rsi_oversold": (15, 35),
            "macd_fast": (8, 16),
            "macd_slow": (20, 40),
            "macd_signal": (5, 15),
            "bb_period": (10, 30),
            "bb_std": (1.5, 3.0),
            "atr_period": (10, 30),
            "atr_multiplier": (1.5, 3.0),
            "volume_ma_period": (10, 30),
            "volume_threshold": (1.0, 3.0),
        }

        # Initialize optimizer
        optimizer = BayesianOptimizer(param_ranges)

        # Run optimization
        logger.info("Starting optimization...")
        results = []

        for i in range(args.n_trials):
            logger.info(f"Trial {i+1}/{args.n_trials}")

            # Get next parameters to try
            params = optimizer.suggest_params()
            logger.info(f"Parameters: {params}")

            # Evaluate parameters
            score = objective_function(params, bot, analyzer)
            logger.info(f"Score: {score}")

            # Update optimizer
            optimizer.update(params, score)

            # Store results
            results.append({
                "trial": i + 1,
                "parameters": params,
                "score": score,
            })

            # Save intermediate results
            pd.DataFrame(results).to_csv(
                output_dir / "optimization_results.csv",
                index=False,
            )

        # Find best parameters
        best_idx = np.argmax([r["score"] for r in results])
        best_params = results[best_idx]["parameters"]
        best_score = results[best_idx]["score"]

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score}")

        # Save best parameters
        with open(output_dir / "best_parameters.yaml", "w") as f:
            yaml.dump(best_params, f)

        # Run final backtest with best parameters
        logger.info("Running final backtest with best parameters...")
        bot.update_params(best_params)
        final_results = bot.run_backtest()
        final_metrics = analyzer.analyze_trades(final_results)

        # Save final results
        with open(output_dir / "final_metrics.yaml", "w") as f:
            yaml.dump(final_metrics, f)

        logger.info("Optimization completed successfully")

    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 