import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from deap import base, creator, tools, algorithms
import random
from bot.backtesting.backtest_engine import BacktestEngine
from bot.analysis.market_regime import MarketRegimeDetector

@dataclass
class OptimizationResult:
    best_parameters: Dict
    best_score: float
    optimization_history: List[Dict]
    parameter_importance: Dict[str, float]

class ParameterOptimizer:
    """Automatic parameter optimization system."""
    
    def __init__(self,
                 parameter_ranges: Dict[str, Tuple[float, float]],
                 optimization_method: str = 'genetic',
                 population_size: int = 50,
                 generations: int = 20,
                 n_initial_points: int = 10,
                 n_iterations: int = 50):
        
        self.parameter_ranges = parameter_ranges
        self.optimization_method = optimization_method
        self.population_size = population_size
        self.generations = generations
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.logger = logging.getLogger(__name__)
        
        self.backtest_engine = BacktestEngine()
        self.market_regime_detector = MarketRegimeDetector()
        
        # Initialize optimization history
        self.optimization_history = []
        
    def optimize(self,
                strategy: Callable,
                market_data: Dict[str, pd.DataFrame],
                start_date: datetime,
                end_date: datetime,
                objective_function: Callable = None) -> OptimizationResult:
        """Optimize strategy parameters."""
        try:
            if self.optimization_method == 'genetic':
                return self._genetic_optimization(
                    strategy, market_data, start_date, end_date, objective_function
                )
            else:
                return self._bayesian_optimization(
                    strategy, market_data, start_date, end_date, objective_function
                )
                
        except Exception as e:
            self.logger.error(f"Error in optimization: {e}")
            return None
            
    def _genetic_optimization(self,
                            strategy: Callable,
                            market_data: Dict[str, pd.DataFrame],
                            start_date: datetime,
                            end_date: datetime,
                            objective_function: Callable) -> OptimizationResult:
        """Optimize parameters using genetic algorithm."""
        try:
            # Create fitness and individual classes
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Initialize toolbox
            toolbox = base.Toolbox()
            
            # Register parameter generation
            for param_name, (min_val, max_val) in self.parameter_ranges.items():
                toolbox.register(
                    f"attr_{param_name}",
                    random.uniform,
                    min_val,
                    max_val
                )
                
            # Create individual and population
            toolbox.register(
                "individual",
                tools.initCycle,
                creator.Individual,
                [getattr(toolbox, f"attr_{param}") for param in self.parameter_ranges.keys()],
                n=1
            )
            toolbox.register(
                "population",
                tools.initRepeat,
                list,
                toolbox.individual
            )
            
            # Register genetic operators
            toolbox.register("evaluate", lambda ind: self._evaluate_parameters(
                ind, strategy, market_data, start_date, end_date, objective_function
            ))
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            
            # Create initial population
            pop = toolbox.population(n=self.population_size)
            
            # Track best solution
            hof = tools.HallOfFame(1)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Run genetic algorithm
            pop, logbook = algorithms.eaSimple(
                pop, toolbox,
                cxpb=0.7,  # Crossover probability
                mutpb=0.3,  # Mutation probability
                ngen=self.generations,
                stats=stats,
                halloffame=hof,
                verbose=True
            )
            
            # Get best parameters
            best_parameters = dict(zip(self.parameter_ranges.keys(), hof[0]))
            best_score = hof[0].fitness.values[0]
            
            # Calculate parameter importance
            parameter_importance = self._calculate_parameter_importance(
                pop, self.parameter_ranges.keys()
            )
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_history=self.optimization_history,
                parameter_importance=parameter_importance
            )
            
        except Exception as e:
            self.logger.error(f"Error in genetic optimization: {e}")
            return None
            
    def _bayesian_optimization(self,
                             strategy: Callable,
                             market_data: Dict[str, pd.DataFrame],
                             start_date: datetime,
                             end_date: datetime,
                             objective_function: Callable) -> OptimizationResult:
        """Optimize parameters using Bayesian optimization."""
        try:
            # Initialize Gaussian Process
            kernel = Matern(nu=2.5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                normalize_y=True
            )
            
            # Generate initial random points
            X = []
            y = []
            
            for _ in range(self.n_initial_points):
                params = self._generate_random_parameters()
                score = self._evaluate_parameters(
                    params, strategy, market_data, start_date, end_date, objective_function
                )
                X.append(list(params.values()))
                y.append(score)
                
            X = np.array(X)
            y = np.array(y)
            
            # Bayesian optimization loop
            for i in range(self.n_iterations):
                # Fit GP
                gp.fit(X, y)
                
                # Generate candidate points
                candidates = self._generate_candidates(X)
                
                # Calculate acquisition function (Expected Improvement)
                ei = self._expected_improvement(candidates, gp, y)
                
                # Select best candidate
                best_idx = np.argmax(ei)
                new_params = dict(zip(self.parameter_ranges.keys(), candidates[best_idx]))
                
                # Evaluate new parameters
                score = self._evaluate_parameters(
                    new_params, strategy, market_data, start_date, end_date, objective_function
                )
                
                # Update data
                X = np.vstack([X, candidates[best_idx]])
                y = np.append(y, score)
                
            # Get best parameters
            best_idx = np.argmax(y)
            best_parameters = dict(zip(self.parameter_ranges.keys(), X[best_idx]))
            best_score = y[best_idx]
            
            # Calculate parameter importance
            parameter_importance = self._calculate_parameter_importance_bayesian(
                X, y, self.parameter_ranges.keys()
            )
            
            return OptimizationResult(
                best_parameters=best_parameters,
                best_score=best_score,
                optimization_history=self.optimization_history,
                parameter_importance=parameter_importance
            )
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian optimization: {e}")
            return None
            
    def _evaluate_parameters(self,
                           parameters: Dict[str, float],
                           strategy: Callable,
                           market_data: Dict[str, pd.DataFrame],
                           start_date: datetime,
                           end_date: datetime,
                           objective_function: Callable) -> float:
        """Evaluate strategy performance with given parameters."""
        try:
            # Run backtest with parameters
            results = self.backtest_engine.run_backtest(
                lambda data, positions: strategy(data, positions, parameters),
                market_data,
                start_date,
                end_date
            )
            
            if results is None:
                return float('-inf')
                
            # Calculate objective score
            if objective_function is None:
                # Default objective: Sharpe ratio with drawdown penalty
                score = results.sharpe_ratio * (1 - results.max_drawdown)
            else:
                score = objective_function(results)
                
            # Record optimization history
            self.optimization_history.append({
                'parameters': parameters,
                'score': score,
                'timestamp': datetime.now()
            })
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {e}")
            return float('-inf')
            
    def _generate_random_parameters(self) -> Dict[str, float]:
        """Generate random parameters within ranges."""
        return {
            param: random.uniform(min_val, max_val)
            for param, (min_val, max_val) in self.parameter_ranges.items()
        }
        
    def _generate_candidates(self, X: np.ndarray) -> np.ndarray:
        """Generate candidate points for Bayesian optimization."""
        candidates = []
        for _ in range(100):  # Generate 100 candidates
            params = []
            for param, (min_val, max_val) in self.parameter_ranges.items():
                params.append(random.uniform(min_val, max_val))
            candidates.append(params)
        return np.array(candidates)
        
    def _expected_improvement(self,
                            candidates: np.ndarray,
                            gp: GaussianProcessRegressor,
                            y: np.ndarray) -> np.ndarray:
        """Calculate Expected Improvement acquisition function."""
        mean, std = gp.predict(candidates, return_std=True)
        best_f = np.max(y)
        
        # Calculate improvement
        improvement = mean - best_f
        
        # Calculate expected improvement
        z = improvement / (std + 1e-9)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        return ei
        
    def _calculate_parameter_importance(self,
                                      population: List,
                                      parameter_names: List[str]) -> Dict[str, float]:
        """Calculate parameter importance using genetic algorithm results."""
        try:
            # Convert population to numpy array
            X = np.array([ind for ind in population])
            y = np.array([ind.fitness.values[0] for ind in population])
            
            # Calculate correlation between parameters and fitness
            correlations = {}
            for i, param in enumerate(parameter_names):
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                correlations[param] = abs(correlation)
                
            # Normalize importance scores
            total = sum(correlations.values())
            if total > 0:
                correlations = {k: v/total for k, v in correlations.items()}
                
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating parameter importance: {e}")
            return {}
            
    def _calculate_parameter_importance_bayesian(self,
                                               X: np.ndarray,
                                               y: np.ndarray,
                                               parameter_names: List[str]) -> Dict[str, float]:
        """Calculate parameter importance using Bayesian optimization results."""
        try:
            # Calculate correlation between parameters and objective
            correlations = {}
            for i, param in enumerate(parameter_names):
                correlation = np.corrcoef(X[:, i], y)[0, 1]
                correlations[param] = abs(correlation)
                
            # Normalize importance scores
            total = sum(correlations.values())
            if total > 0:
                correlations = {k: v/total for k, v in correlations.items()}
                
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating parameter importance: {e}")
            return {}
            
    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            # Extract data
            scores = [entry['score'] for entry in self.optimization_history]
            timestamps = [entry['timestamp'] for entry in self.optimization_history]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, scores, 'b-', label='Score')
            plt.scatter(timestamps, scores, c='b', s=50)
            
            # Add best score line
            best_score = max(scores)
            plt.axhline(y=best_score, color='r', linestyle='--', label=f'Best Score: {best_score:.2f}')
            
            # Customize plot
            plt.title('Optimization History')
            plt.xlabel('Time')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plt.savefig('optimization_history.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting optimization history: {e}")
            
    def plot_parameter_importance(self, importance: Dict[str, float]):
        """Plot parameter importance."""
        try:
            import matplotlib.pyplot as plt
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Sort parameters by importance
            sorted_params = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Create bar plot
            params = [p[0] for p in sorted_params]
            scores = [p[1] for p in sorted_params]
            
            plt.bar(params, scores)
            
            # Customize plot
            plt.title('Parameter Importance')
            plt.xlabel('Parameter')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plt.savefig('parameter_importance.png')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting parameter importance: {e}") 