import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import json
import os
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import time

class AITradingStrategy:
    def __init__(self, config: dict):
        """Initialize the AI trading strategy."""
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model_path = os.path.join('models', 'ai_strategy')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize OpenAI
        openai.api_key = config.get('openai', {}).get('api_key')
        self.gpt_model = "gpt-4"
        
        # Initialize neural network
        self.state_dim = 50  # Number of features (indicators + price data)
        self.action_dim = 3  # Hold, Buy, Sell
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialize experience replay buffer with numpy for better performance
        self.memory = np.zeros((10000, 4), dtype=np.float32)  # state, action, reward, next_state
        self.memory_size = 10000
        self.batch_size = 64
        self.memory_counter = 0
        
        # Initialize trading parameters
        self.indicators = {}
        self.positions = {}
        self.trade_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0
        }
        
        # Self-learning parameters
        self.learning_rate = 0.001
        self.min_trades_for_learning = 10
        self.optimization_interval = 100  # Optimize every 100 trades
        self.last_optimization = 0
        
        # Risk management parameters
        self.max_position_size = config.get('risk', {}).get('max_position_size', 0.1)
        self.stop_loss_pct = config.get('risk', {}).get('stop_loss', 0.02)
        self.take_profit_pct = config.get('risk', {}).get('take_profit', 0.04)
        self.max_drawdown = config.get('risk', {}).get('max_drawdown', 0.1)
        
        # Initialize caches for performance optimization
        self.state_cache = {}
        self.feature_cache = {}
        self.analysis_cache = {}
        self.cache_size = 1000  # Maximum number of cached items
        self.cache_ttl = 300  # Cache time-to-live in seconds
        
        # Load saved model if exists
        self._load_model()
        
        self.logger.info("AI Trading Strategy initialized with performance optimizations")
        
    def _build_model(self) -> nn.Module:
        """Build the neural network model."""
        model = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.action_dim)
        )
        return model
    
    def _prepare_state(self, data: pd.DataFrame) -> torch.Tensor:
        """Prepare the state tensor from market data with caching."""
        try:
            # Generate cache key from data
            cache_key = hash(data.to_string())
            
            # Check cache
            if cache_key in self.state_cache:
                cache_entry = self.state_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['state']
            
            # Extract features efficiently using numpy
            features = np.zeros(self.state_dim, dtype=np.float32)
            
            # Price features (using numpy for better performance)
            price_data = data[['close', 'high', 'low', 'volume']].iloc[-1].values
            features[:4] = price_data
            
            # Technical indicators (using numpy operations)
            if 'rsi' in data.columns:
                features[4] = data['rsi'].iloc[-1]
            if 'macd' in data.columns:
                features[5] = data['macd'].iloc[-1]
            if 'macd_signal' in data.columns:
                features[6] = data['macd_signal'].iloc[-1]
            if 'bb_upper' in data.columns:
                features[7] = data['bb_upper'].iloc[-1]
            if 'bb_lower' in data.columns:
                features[8] = data['bb_lower'].iloc[-1]
            if 'ema_short' in data.columns:
                features[9] = data['ema_short'].iloc[-1]
            if 'ema_medium' in data.columns:
                features[10] = data['ema_medium'].iloc[-1]
            if 'ema_long' in data.columns:
                features[11] = data['ema_long'].iloc[-1]
            
            # Normalize features efficiently
            mean = np.mean(features)
            std = np.std(features)
            if std > 0:
                features = (features - mean) / std
            
            # Convert to tensor
            state = torch.FloatTensor(features)
            
            # Update cache
            self._update_cache(self.state_cache, cache_key, {
                'state': state,
                'timestamp': time.time()
            })
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error preparing state: {str(e)}")
            return torch.zeros(self.state_dim)

    def _update_cache(self, cache: dict, key: Any, value: Any):
        """Update cache with size limit and TTL."""
        # Remove old entries
        current_time = time.time()
        cache = {k: v for k, v in cache.items() 
                if current_time - v['timestamp'] < self.cache_ttl}
        
        # Add new entry
        cache[key] = value
        
        # Remove oldest entries if cache is too large
        if len(cache) > self.cache_size:
            sorted_items = sorted(cache.items(), 
                                key=lambda x: x[1]['timestamp'])
            for k, _ in sorted_items[:len(cache) - self.cache_size]:
                del cache[k]

    def _get_action(self, state: torch.Tensor, epsilon: float = 0.1) -> int:
        """Get action from model with epsilon-greedy exploration."""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()
    
    def _calculate_reward(self, action: int, next_price: float, 
                         current_price: float, position: float) -> float:
        """Calculate reward for the action taken."""
        price_change = (next_price - current_price) / current_price
        
        if action == 0:  # Hold
            reward = 0
        elif action == 1:  # Buy
            reward = price_change if position <= 0 else -0.001  # Penalty for buying when already long
        else:  # Sell
            reward = -price_change if position >= 0 else -0.001  # Penalty for selling when already short
            
        return reward
    
    def update(self, data: pd.DataFrame, action: int, reward: float):
        """Update the model with new experience."""
        state = self._prepare_state(data)
        next_state = self._prepare_state(data.shift(-1))
        
        # Store experience in memory
        self.memory[self.memory_counter] = [state.numpy(), action, reward, next_state.numpy()]
        self.memory_counter = (self.memory_counter + 1) % self.memory_size
            
        # Train on batch of experiences
        if self.memory_counter >= self.batch_size:
            self._train()
    
    def _train(self):
        """Train the model on a batch of experiences with optimized batch processing."""
        if self.memory_counter < self.batch_size:
            return
            
        try:
            # Sample batch efficiently using numpy
            indices = np.random.choice(self.memory_counter, self.batch_size, replace=False)
            batch = self.memory[indices]
            
            # Convert batch to tensors efficiently
            states = torch.FloatTensor(batch[:, 0])
            actions = torch.LongTensor(batch[:, 1])
            rewards = torch.FloatTensor(batch[:, 2])
            next_states = torch.FloatTensor(batch[:, 3])
            
            # Calculate Q-values in a single forward pass
            current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.model(next_states).max(1)[0].detach()
            expected_q_values = rewards + 0.99 * next_q_values
            
            # Calculate loss and update
            loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log training progress
            self.logger.info(f"Training Progress - Loss: {loss.item():.4f}, "
                           f"Memory Size: {self.memory_counter}/{self.memory_size}, "
                           f"Average Reward: {rewards.mean():.4f}, "
                           f"Q-Value Range: [{current_q_values.min():.4f}, {current_q_values.max():.4f}]")
            
            # Save training metrics
            self.performance_metrics['training_loss'] = loss.item()
            self.performance_metrics['average_reward'] = rewards.mean().item()
            self.performance_metrics['q_value_range'] = [current_q_values.min().item(), current_q_values.max().item()]
            
        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
    
    def _calculate_position_size(self, price: float, confidence: float, available_capital: float) -> float:
        """Calculate position size based on risk parameters and confidence."""
        try:
            # Base position size on risk parameters
            risk_amount = available_capital * self.max_position_size
            
            # Adjust for confidence
            position_size = risk_amount * confidence
            
            # Adjust for volatility
            if hasattr(self, 'current_volatility'):
                # Reduce position size in high volatility
                volatility_factor = 1 - (self.current_volatility * 2)
                position_size *= max(0.5, volatility_factor)
            
            # Adjust for market conditions
            if hasattr(self, 'market_regime'):
                if self.market_regime == 'trending':
                    position_size *= 1.1
                elif self.market_regime == 'ranging':
                    position_size *= 0.8
            
            # Ensure minimum and maximum limits
            min_position = available_capital * 0.001  # Minimum 0.1% of capital
            max_position = available_capital * self.max_position_size
            
            position_size = max(min_position, min(position_size, max_position))
            
            # Convert to quantity
            quantity = position_size / price
            
            self.logger.debug(f"Calculated position size: {position_size:.4f} ({quantity:.4f} units)")
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return 0.001  # Return minimum position size on error

    def update_performance_metrics(self, trade_result: dict):
        """Update performance metrics after each trade."""
        self.performance_metrics['total_trades'] += 1
        
        if trade_result['profit'] > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['total_profit'] += trade_result['profit']
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['total_loss'] += abs(trade_result['profit'])
        
        # Update derived metrics
        total_trades = self.performance_metrics['total_trades']
        if total_trades > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / total_trades
            )
            
            if self.performance_metrics['total_loss'] > 0:
                self.performance_metrics['profit_factor'] = (
                    self.performance_metrics['total_profit'] / 
                    self.performance_metrics['total_loss']
                )
        
        # Check if optimization is needed
        if (total_trades >= self.min_trades_for_learning and 
            total_trades % self.optimization_interval == 0):
            self._optimize_strategy()
    
    def _optimize_strategy(self):
        """Optimize strategy parameters based on performance and GPT-4 analysis."""
        try:
            self.logger.info("Starting strategy optimization with GPT-4...")
            
            # Get GPT-4's analysis of current performance
            performance_summary = {
                'win_rate': self.performance_metrics['win_rate'],
                'profit_factor': self.performance_metrics['profit_factor'],
                'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
                'max_drawdown': self.performance_metrics['max_drawdown'],
                'total_trades': self.performance_metrics['total_trades']
            }
            
            # Create prompt for GPT-4 to analyze performance and suggest improvements
            prompt = f"""Analyze the following trading strategy performance and suggest optimizations:

            Current Performance:
            Win Rate: {performance_summary['win_rate']:.2%}
            Profit Factor: {performance_summary['profit_factor']:.2f}
            Sharpe Ratio: {performance_summary['sharpe_ratio']:.2f}
            Max Drawdown: {performance_summary['max_drawdown']:.2%}
            Total Trades: {performance_summary['total_trades']}

            Current Parameters:
            Learning Rate: {self.learning_rate}
            Max Position Size: {self.max_position_size}
            Stop Loss: {self.stop_loss_pct}
            Take Profit: {self.take_profit_pct}

            Please provide:
            1. Analysis of current performance
            2. Suggested parameter adjustments
            3. Neural network architecture modifications
            4. Risk management improvements

            Format the response as a JSON object with these keys:
            analysis, parameter_adjustments, nn_modifications, risk_improvements
            """
            
            # Get GPT-4's optimization suggestions
            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are an expert in AI trading strategy optimization."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse GPT-4's suggestions
            suggestions = json.loads(response.choices[0].message.content)
            
            # Apply parameter adjustments
            if 'parameter_adjustments' in suggestions:
                params = suggestions['parameter_adjustments']
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.max_position_size = params.get('max_position_size', self.max_position_size)
                self.stop_loss_pct = params.get('stop_loss', self.stop_loss_pct)
                self.take_profit_pct = params.get('take_profit', self.take_profit_pct)
            
            # Apply neural network modifications
            if 'nn_modifications' in suggestions:
                mods = suggestions['nn_modifications']
                if mods.get('rebuild_model', False):
                    self._rebuild_model_with_suggestions(mods)
            
            # Apply risk management improvements
            if 'risk_improvements' in suggestions:
                risk = suggestions['risk_improvements']
                self._update_risk_parameters(risk)
            
            # Retrain model with recent experiences
            if self.memory_counter >= self.batch_size:
                self._train()
            
            # Save optimized parameters
            self._save_optimized_parameters()
            
            self.logger.info("Strategy optimization completed with GPT-4 guidance")
            self.logger.info(f"New parameters: max_position_size={self.max_position_size:.3f}, "
                           f"stop_loss={self.stop_loss_pct:.3f}, "
                           f"take_profit={self.take_profit_pct:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error during strategy optimization: {str(e)}")
    
    def _rebuild_model_with_suggestions(self, modifications: Dict):
        """Rebuild the neural network model based on GPT-4's suggestions."""
        try:
            # Extract architecture suggestions
            hidden_layers = modifications.get('hidden_layers', [128, 64])
            dropout_rate = modifications.get('dropout_rate', 0.2)
            activation = modifications.get('activation', 'ReLU')
            
            # Build new model architecture
            layers = []
            input_dim = self.state_dim
            
            for hidden_dim in hidden_layers:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU() if activation == 'ReLU' else nn.LeakyReLU(),
                    nn.Dropout(dropout_rate)
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, self.action_dim))
            
            # Create new model
            self.model = nn.Sequential(*layers)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            self.logger.info(f"Neural network rebuilt with new architecture: {hidden_layers}")
            
        except Exception as e:
            self.logger.error(f"Error rebuilding model: {str(e)}")
    
    def _update_risk_parameters(self, risk_params: Dict):
        """Update risk management parameters based on GPT-4's suggestions."""
        try:
            # Update risk parameters
            if 'max_drawdown' in risk_params:
                self.max_drawdown = risk_params['max_drawdown']
            
            if 'position_sizing' in risk_params:
                sizing = risk_params['position_sizing']
                self.max_position_size = sizing.get('max_size', self.max_position_size)
            
            if 'stop_loss' in risk_params:
                sl = risk_params['stop_loss']
                self.stop_loss_pct = sl.get('percentage', self.stop_loss_pct)
            
            if 'take_profit' in risk_params:
                tp = risk_params['take_profit']
                self.take_profit_pct = tp.get('percentage', self.take_profit_pct)
            
            self.logger.info("Risk parameters updated based on GPT-4 suggestions")
            
        except Exception as e:
            self.logger.error(f"Error updating risk parameters: {str(e)}")
    
    def _save_optimized_parameters(self):
        """Save optimized parameters to file."""
        try:
            params = {
                'max_position_size': self.max_position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'learning_rate': self.learning_rate,
                'performance_metrics': self.performance_metrics
            }
            
            with open(os.path.join(self.model_path, 'optimized_params.json'), 'w') as f:
                json.dump(params, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Error saving optimized parameters: {str(e)}")
    
    def _load_optimized_parameters(self):
        """Load optimized parameters from file."""
        try:
            params_path = os.path.join(self.model_path, 'optimized_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                
                self.max_position_size = params.get('max_position_size', self.max_position_size)
                self.stop_loss_pct = params.get('stop_loss_pct', self.stop_loss_pct)
                self.take_profit_pct = params.get('take_profit_pct', self.take_profit_pct)
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.performance_metrics = params.get('performance_metrics', self.performance_metrics)
                
        except Exception as e:
            self.logger.error(f"Error loading optimized parameters: {str(e)}")
    
    def _get_gpt_analysis(self, data: pd.DataFrame, math_analysis: Dict) -> Dict:
        """Get market analysis from ChatGPT-4 with caching and optimized processing."""
        try:
            # Generate cache key
            cache_key = hash(str(data.iloc[-1].to_dict()) + str(math_analysis))
            
            # Check cache
            if cache_key in self.analysis_cache:
                cache_entry = self.analysis_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['analysis']
            
            # Prepare market data summary efficiently
            market_summary = self._prepare_market_summary(data)
            
            # Get performance metrics
            performance_metrics = self._get_performance_metrics()
            
            # Create optimized prompt
            prompt = self._create_analysis_prompt(market_summary, math_analysis, performance_metrics)
            
            # Get response from GPT-4 with retry mechanism
            response = self._get_gpt_response(prompt)
            
            # Parse and cache the analysis
            analysis = json.loads(response.choices[0].message.content)
            self._update_cache(self.analysis_cache, cache_key, {
                'analysis': analysis,
                'timestamp': time.time()
            })
            
            # Apply optimizations if available
            if 'strategy_optimization' in analysis:
                self._apply_strategy_optimizations(analysis['strategy_optimization'])
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error getting GPT analysis: {str(e)}")
            return None

    def _prepare_market_summary(self, data: pd.DataFrame) -> Dict:
        """Prepare market summary efficiently."""
        try:
            # Use numpy for efficient calculations
            close_prices = data['close'].values
            volumes = data['volume'].values
            
            return {
                'current_price': float(close_prices[-1]),
                'price_change_24h': float((close_prices[-1] - close_prices[-24]) / close_prices[-24] * 100),
                'volume_24h': float(np.sum(volumes[-24:])),
                'volatility': float(np.std(np.diff(close_prices) / close_prices[:-1]) * 100),
                'rsi': float(data['rsi'].iloc[-1]) if 'rsi' in data.columns else None,
                'macd': float(data['macd'].iloc[-1]) if 'macd' in data.columns else None,
                'macd_signal': float(data['macd_signal'].iloc[-1]) if 'macd_signal' in data.columns else None,
                'bb_position': float((close_prices[-1] - data['bb_lower'].iloc[-1]) / 
                                   (data['bb_upper'].iloc[-1] - data['bb_lower'].iloc[-1])) if 'bb_upper' in data.columns else None,
                'ema_short': float(data['ema_short'].iloc[-1]) if 'ema_short' in data.columns else None,
                'ema_medium': float(data['ema_medium'].iloc[-1]) if 'ema_medium' in data.columns else None,
                'ema_long': float(data['ema_long'].iloc[-1]) if 'ema_long' in data.columns else None,
                'market_regime': self.market_regime if hasattr(self, 'market_regime') else None,
                'current_volatility': self.current_volatility if hasattr(self, 'current_volatility') else None
            }
        except Exception as e:
            self.logger.error(f"Error preparing market summary: {str(e)}")
            return {}

    def _get_performance_metrics(self) -> Dict:
        """Get performance metrics efficiently."""
        return {
            'win_rate': self.performance_metrics['win_rate'],
            'profit_factor': self.performance_metrics['profit_factor'],
            'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
            'max_drawdown': self.performance_metrics['max_drawdown'],
            'total_trades': self.performance_metrics['total_trades'],
            'winning_trades': self.performance_metrics['winning_trades'],
            'losing_trades': self.performance_metrics['losing_trades']
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_gpt_response(self, prompt: str) -> Any:
        """Get response from GPT-4 with optimized parameters."""
        return openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=[
                {"role": "system", "content": "You are the central AI coordinator for a sophisticated trading system."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            presence_penalty=0.1,
            frequency_penalty=0.1
        )

    def _apply_strategy_optimizations(self, optimizations: Dict):
        """Apply strategy optimizations suggested by GPT-4."""
        try:
            # Update risk parameters
            if 'risk_parameters' in optimizations:
                risk_params = optimizations['risk_parameters']
                self.max_position_size = risk_params.get('max_position_size', self.max_position_size)
                self.stop_loss_pct = risk_params.get('stop_loss', self.stop_loss_pct)
                self.take_profit_pct = risk_params.get('take_profit', self.take_profit_pct)
                self.max_drawdown = risk_params.get('max_drawdown', self.max_drawdown)
            
            # Update model parameters
            if 'model_parameters' in optimizations:
                model_params = optimizations['model_parameters']
                self.learning_rate = model_params.get('learning_rate', self.learning_rate)
                if 'architecture' in model_params:
                    self._rebuild_model_with_suggestions(model_params['architecture'])
            
            # Update trading parameters
            if 'trading_parameters' in optimizations:
                trading_params = optimizations['trading_parameters']
                self.min_trades_for_learning = trading_params.get('min_trades', self.min_trades_for_learning)
                self.optimization_interval = trading_params.get('optimization_interval', self.optimization_interval)
            
            self.logger.info("Applied strategy optimizations from GPT-4")
            
        except Exception as e:
            self.logger.error(f"Error applying strategy optimizations: {str(e)}")

    def generate_signal(self, data: pd.DataFrame, position: float = 0) -> Dict:
        """Generate trading signal using integrated analysis from all components."""
        try:
            # Get neural network prediction with full mathematical analysis
            state = self._prepare_state(data)
            with torch.no_grad():
                raw_outputs = self.model(state)
                probabilities = torch.softmax(raw_outputs, dim=0)
                action = torch.argmax(probabilities).item()
                action_confidences = probabilities.tolist()
                entropy = -torch.sum(probabilities * torch.log(probabilities)).item()
                variance = torch.var(probabilities).item()
                feature_importance = self._calculate_feature_importance(state)
            
            # Prepare comprehensive analysis for GPT-4
            math_analysis = {
                'action_probabilities': action_confidences,
                'entropy': entropy,
                'variance': variance,
                'feature_importance': feature_importance,
                'raw_outputs': raw_outputs.tolist(),
                'state_features': state.tolist()
            }
            
            # Get integrated analysis from GPT-4
            gpt_analysis = self._get_gpt_analysis(data, math_analysis)
            
            if gpt_analysis:
                # Extract components from GPT-4 analysis
                market_analysis = gpt_analysis['market_analysis']
                nn_analysis = gpt_analysis['nn_analysis']
                risk_assessment = gpt_analysis['risk_assessment']
                trading_decision = gpt_analysis['trading_decision']
                
                # Calculate final confidence using all components
                final_confidence = self._calculate_integrated_confidence(
                    nn_confidence=action_confidences[action],
                    gpt_confidence=float(trading_decision['confidence']),
                    market_confidence=float(market_analysis['confidence']),
                    risk_confidence=float(risk_assessment['confidence']),
                    entropy=entropy,
                    variance=variance
                )
                
                return {
                    'action': trading_decision['recommendation'],
                    'confidence': final_confidence,
                    'market_analysis': market_analysis,
                    'nn_analysis': nn_analysis,
                    'risk_assessment': risk_assessment,
                    'trading_decision': trading_decision,
                    'entropy': entropy,
                    'variance': variance,
                    'feature_importance': feature_importance
                }
            else:
                # Fallback to neural network with mathematical analysis
                return {
                    'action': ['hold', 'buy', 'sell'][action],
                    'confidence': action_confidences[action],
                    'entropy': entropy,
                    'variance': variance,
                    'feature_importance': feature_importance
                }
                
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            return None

    def _calculate_integrated_confidence(self, nn_confidence: float, gpt_confidence: float,
                                      market_confidence: float, risk_confidence: float,
                                      entropy: float, variance: float) -> float:
        """Calculate final confidence by integrating all components."""
        try:
            # Calculate mathematical weights
            entropy_weight = 1 - (entropy / np.log(self.action_dim))
            variance_weight = 1 - variance
            
            # Weighted average of all confidence components
            final_confidence = (
                nn_confidence * 0.3 +          # Neural network confidence
                gpt_confidence * 0.25 +        # GPT-4 analysis confidence
                market_confidence * 0.2 +      # Market analysis confidence
                risk_confidence * 0.15 +       # Risk assessment confidence
                entropy_weight * 0.05 +        # Mathematical certainty
                variance_weight * 0.05         # Prediction stability
            )
            
            return min(max(final_confidence, 0.0), 1.0)  # Ensure between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating integrated confidence: {str(e)}")
            return nn_confidence  # Fallback to neural network confidence

    def _calculate_feature_importance(self, state: torch.Tensor) -> List[float]:
        """Calculate feature importance using gradient-based method."""
        try:
            state.requires_grad = True
            output = self.model(state)
            output.backward()
            
            # Get gradients for input features
            gradients = state.grad.abs()
            
            # Normalize gradients
            feature_importance = (gradients / gradients.sum()).tolist()
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {str(e)}")
            return [0.0] * self.state_dim

    def __call__(self, data, positions, regime):
        """
        Maak de strategie callable voor de backtest engine.
        data: pd.DataFrame met marktdata
        positions: dict met huidige posities
        regime: marktomstandigheden
        Retourneert een dict met signalen per symbool
        """
        try:
            # Voor deze test nemen we aan dat er slechts 1 symbool is
            symbol = list(positions.keys())[0] if positions else 'BTCUSDT'
            
            # Get current price and available capital
            current_price = data['close'].iloc[-1]
            available_capital = 10000  # Example capital, should come from config
            
            # Generate signal
            signal = self.generate_signal(data)
            
            # Calculate position size
            quantity = self._calculate_position_size(
                price=current_price,
                confidence=signal['confidence'],
                available_capital=available_capital
            )
            
            # Log trading decision
            self.logger.info(f"Trading decision for {symbol}:")
            self.logger.info(f"Action: {signal['action']}")
            self.logger.info(f"Confidence: {signal['confidence']:.2f}")
            self.logger.info(f"Quantity: {quantity:.4f}")
            self.logger.info(f"Price: {current_price:.2f}")
            
            # Return signal with calculated quantity
            if signal['action'] == 'buy':
                return {symbol: {'action': 'buy', 'quantity': quantity}}
            elif signal['action'] == 'sell':
                return {symbol: {'action': 'sell', 'quantity': quantity}}
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error in __call__: {str(e)}")
            return {}

    def save_model(self):
        """Save the model state."""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(self.model_path, 'model.pth'))
            
            # Save training history
            with open(os.path.join(self.model_path, 'history.json'), 'w') as f:
                json.dump(self.trade_history, f)
                
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def _load_model(self):
        """Load the saved model state."""
        try:
            model_path = os.path.join(self.model_path, 'model.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load training history
                history_path = os.path.join(self.model_path, 'history.json')
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        self.trade_history = json.load(f)
                        
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
    
    def optimize_indicators(self, data: pd.DataFrame) -> Dict:
        """Optimize indicator parameters based on historical performance."""
        try:
            # Initialize optimization parameters
            indicator_params = {
                'rsi': {'period': range(10, 30)},
                'macd': {
                    'fast_period': range(8, 20),
                    'slow_period': range(20, 40),
                    'signal_period': range(5, 15)
                },
                'bollinger_bands': {
                    'period': range(10, 30),
                    'std_dev': np.arange(1.5, 3.0, 0.1)
                }
            }
            
            best_params = {}
            best_sharpe = -np.inf
            
            # Grid search for optimal parameters
            for rsi_period in indicator_params['rsi']['period']:
                for macd_fast in indicator_params['macd']['fast_period']:
                    for macd_slow in indicator_params['macd']['slow_period']:
                        for macd_signal in indicator_params['macd']['signal_period']:
                            for bb_period in indicator_params['bollinger_bands']['period']:
                                for bb_std in indicator_params['bollinger_bands']['std_dev']:
                                    # Calculate indicators with current parameters
                                    params = {
                                        'rsi': {'period': rsi_period},
                                        'macd': {
                                            'fast_period': macd_fast,
                                            'slow_period': macd_slow,
                                            'signal_period': macd_signal
                                        },
                                        'bollinger_bands': {
                                            'period': bb_period,
                                            'std_dev': bb_std
                                        }
                                    }
                                    
                                    # Calculate performance metrics
                                    performance = self._evaluate_parameters(data, params)
                                    sharpe_ratio = performance['sharpe_ratio']
                                    
                                    if sharpe_ratio > best_sharpe:
                                        best_sharpe = sharpe_ratio
                                        best_params = params
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error optimizing indicators: {str(e)}")
            return {}
    
    def _evaluate_parameters(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Evaluate trading performance with given parameters."""
        try:
            # Calculate returns
            returns = []
            position = 0
            
            for i in range(len(data) - 1):
                # Generate signal
                signal = self.generate_signal(data.iloc[:i+1], position)
                
                # Update position
                if signal['action'] == 'buy' and position <= 0:
                    position = 1
                elif signal['action'] == 'sell' and position >= 0:
                    position = -1
                    
                # Calculate return
                price_change = (data['close'].iloc[i+1] - data['close'].iloc[i]) / data['close'].iloc[i]
                returns.append(position * price_change)
            
            returns = np.array(returns)
            
            # Calculate performance metrics
            total_return = np.sum(returns)
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            max_drawdown = np.min(np.cumsum(returns))
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating parameters: {str(e)}")
            return {
                'total_return': 0,
                'sharpe_ratio': -np.inf,
                'max_drawdown': 0
            } 