import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Any
from .enums import SignalType
from .indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_atr,
    calculate_adx,
    calculate_stochastic_rsi,
    calculate_obv,
    calculate_klinger_oscillator,
    calculate_vortex_indicator,
    calculate_chaikin_money_flow,
    calculate_donchian_channels,
    calculate_dpo,
    calculate_williams_r,
    calculate_schaff_trend_cycle
)
from .rl_agent import RLAgent, TradingEnvironment
import torch

class Strategy:
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any] = None):
        """Initialize strategy with data and configuration."""
        self.data = data
        self.config = config or {}
        
        # Default indicator configurations
        self.indicator_configs = {
            'RSI': {'period': 14},
            'MACD': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'BB': {'period': 20, 'std_dev': 2},
            'KVO': {'short_period': 34, 'long_period': 55},
            'VI': {'period': 14},
            'CMF': {'period': 20},
            'DC': {'period': 20},
            'DPO': {'period': 20},
            'WILLR': {'period': 14},
            'STC': {'fast_period': 23, 'slow_period': 50}
        }
        
        # Update configurations if provided
        if 'indicators' in self.config:
            self.indicator_configs.update(self.config['indicators'])
            
        # Calculate indicators
        self.calculate_indicators()
        
        # Initialize RL agent if enabled
        self.use_rl = self.config.get('use_rl', False)
        if self.use_rl:
            self.rl_agent = RLAgent(
                state_dim=len(self.data.columns) + 2,  # +2 for position and balance
                action_dim=3,  # hold, buy, sell
                learning_rate=self.config.get('learning_rate', 0.001),
                gamma=self.config.get('gamma', 0.99),
                epsilon=self.config.get('epsilon', 0.1)
            )
        
        self.logger = logging.getLogger(__name__)
        self.market_data = None
        self._load_indicator_configs()

    def validate_config(self):
        required_fields = ['symbol', 'timeframe', 'indicators']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field: {field}")

    def load_indicator_configs(self):
        self.indicator_configs = self.config['indicators']

    def update_market_data(self, data: pd.DataFrame):
        """Update market data and calculate indicators."""
        self.data = data
        self.calculate_indicators()

    def calculate_indicators(self):
        """Calculate technical indicators based on configuration."""
        # Calculate RSI
        if 'RSI' in self.indicator_configs:
            rsi_config = self.indicator_configs['RSI']
            self.data['RSI'] = calculate_rsi(self.data, rsi_config['period'])
        
        # Calculate MACD
        if 'MACD' in self.indicator_configs:
            macd_config = self.indicator_configs['MACD']
            self.data['MACD'] = calculate_macd(
                self.data,
                macd_config['fast_period'],
                macd_config['slow_period'],
                macd_config['signal_period']
            )
        
        # Calculate Bollinger Bands
        if 'BB' in self.indicator_configs:
            bb_config = self.indicator_configs['BB']
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
                self.data,
                bb_config['period'],
                bb_config['std_dev']
            )
            self.data['BB_upper'] = bb_upper
            self.data['BB_middle'] = bb_middle
            self.data['BB_lower'] = bb_lower
            
        # Calculate Klinger Oscillator
        if 'KVO' in self.indicator_configs:
            kvo_config = self.indicator_configs['KVO']
            kvo, signal = calculate_klinger_oscillator(
                self.data,
                kvo_config['short_period'],
                kvo_config['long_period']
            )
            self.data['KVO'] = kvo
            self.data['KVO_signal'] = signal
            
        # Calculate Vortex Indicator
        if 'VI' in self.indicator_configs:
            vi_config = self.indicator_configs['VI']
            vi_plus, vi_minus = calculate_vortex_indicator(
                self.data,
                vi_config['period']
            )
            self.data['VI_plus'] = vi_plus
            self.data['VI_minus'] = vi_minus
            
        # Calculate Chaikin Money Flow
        if 'CMF' in self.indicator_configs:
            cmf_config = self.indicator_configs['CMF']
            self.data['CMF'] = calculate_chaikin_money_flow(
                self.data,
                cmf_config['period']
            )
            
        # Calculate Donchian Channels
        if 'DC' in self.indicator_configs:
            dc_config = self.indicator_configs['DC']
            upper, middle, lower = calculate_donchian_channels(
                self.data,
                dc_config['period']
            )
            self.data['DC_upper'] = upper
            self.data['DC_middle'] = middle
            self.data['DC_lower'] = lower
            
        # Calculate DPO
        if 'DPO' in self.indicator_configs:
            dpo_config = self.indicator_configs['DPO']
            self.data['DPO'] = calculate_dpo(
                self.data,
                dpo_config['period']
            )
            
        # Calculate Williams %R
        if 'WILLR' in self.indicator_configs:
            willr_config = self.indicator_configs['WILLR']
            self.data['WILLR'] = calculate_williams_r(
                self.data,
                willr_config['period']
            )
            
        # Calculate Schaff Trend Cycle
        if 'STC' in self.indicator_configs:
            stc_config = self.indicator_configs['STC']
            self.data['STC'] = calculate_schaff_trend_cycle(
                self.data,
                stc_config['fast_period'],
                stc_config['slow_period']
            )

    def generate_signal(self, current_position: int = 0, current_balance: float = 0) -> Dict[str, Any]:
        """Generate trading signal based on current market conditions and indicators."""
        signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Get latest data point
        latest = self.data.iloc[-1]
        
        # RSI Analysis
        if 'RSI' in self.data.columns:
            rsi = latest['RSI']
            signal['indicators']['RSI'] = rsi
            if rsi < 30:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.2
            elif rsi > 70:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.2
                
        # MACD Analysis
        if 'MACD' in self.data.columns:
            macd = latest['MACD']
            signal['indicators']['MACD'] = macd
            if macd > 0:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.2
            elif macd < 0:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.2
                
        # Bollinger Bands Analysis
        if all(x in self.data.columns for x in ['BB_upper', 'BB_lower']):
            price = latest['close']
            bb_upper = latest['BB_upper']
            bb_lower = latest['BB_lower']
            signal['indicators']['BB'] = {
                'upper': bb_upper,
                'lower': bb_lower,
                'price': price
            }
            if price < bb_lower:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.15
            elif price > bb_upper:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.15
                
        # Klinger Oscillator Analysis
        if all(x in self.data.columns for x in ['KVO', 'KVO_signal']):
            kvo = latest['KVO']
            kvo_signal = latest['KVO_signal']
            signal['indicators']['KVO'] = {
                'value': kvo,
                'signal': kvo_signal
            }
            if kvo > kvo_signal:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.1
            elif kvo < kvo_signal:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.1
                
        # Vortex Indicator Analysis
        if all(x in self.data.columns for x in ['VI_plus', 'VI_minus']):
            vi_plus = latest['VI_plus']
            vi_minus = latest['VI_minus']
            signal['indicators']['VI'] = {
                'plus': vi_plus,
                'minus': vi_minus
            }
            if vi_plus > vi_minus:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.1
            elif vi_plus < vi_minus:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.1
                
        # Chaikin Money Flow Analysis
        if 'CMF' in self.data.columns:
            cmf = latest['CMF']
            signal['indicators']['CMF'] = cmf
            if cmf > 0.2:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.1
            elif cmf < -0.2:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.1
                
        # Donchian Channels Analysis
        if all(x in self.data.columns for x in ['DC_upper', 'DC_lower']):
            price = latest['close']
            dc_upper = latest['DC_upper']
            dc_lower = latest['DC_lower']
            signal['indicators']['DC'] = {
                'upper': dc_upper,
                'lower': dc_lower,
                'price': price
            }
            if price < dc_lower:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.1
            elif price > dc_upper:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.1
                
        # DPO Analysis
        if 'DPO' in self.data.columns:
            dpo = latest['DPO']
            signal['indicators']['DPO'] = dpo
            if dpo > 0:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.1
            elif dpo < 0:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.1
                
        # Williams %R Analysis
        if 'WILLR' in self.data.columns:
            willr = latest['WILLR']
            signal['indicators']['WILLR'] = willr
            if willr < -80:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.1
            elif willr > -20:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.1
                
        # Schaff Trend Cycle Analysis
        if 'STC' in self.data.columns:
            stc = latest['STC']
            signal['indicators']['STC'] = stc
            if stc > 75:
                signal['action'] = 'BUY'
                signal['confidence'] += 0.1
            elif stc < 25:
                signal['action'] = 'SELL'
                signal['confidence'] += 0.1
        
        # Normalize confidence to 0-1 range
        signal['confidence'] = min(1.0, signal['confidence'])
        
        # Consider current position
        if current_position > 0 and signal['action'] == 'BUY':
            signal['action'] = 'HOLD'
        elif current_position < 0 and signal['action'] == 'SELL':
            signal['action'] = 'HOLD'
            
        return signal

    def generate_signals(self) -> pd.DataFrame:
        """Generate trading signals for all data points."""
        if self.data.empty:
            return pd.DataFrame()
            
        signals = []
        for i in range(len(self.data)):
            signal = self.generate_signal()
            signals.append({
                'timestamp': self.data.index[i],
                'action': signal['action'],
                'confidence': signal['confidence'],
                'indicators': signal['indicators']
            })
            
        return pd.DataFrame(signals)

    def train(self, episodes: int = 1000, batch_size: int = 32):
        """Train the RL agent on historical data."""
        if not self.use_rl:
            raise ValueError("RL agent is not enabled")
            
        self.logger.info(f"Starting training for {episodes} episodes")
        
        # Create training environment
        env = TradingEnvironment(
            data=self.data,
            indicators=list(self.indicator_configs.keys()),
            initial_balance=self.config.get('initial_balance', 10000),
            transaction_fee=self.config.get('transaction_fee', 0.001)
        )
        
        # Training loop
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Get action from RL agent
                action = self.rl_agent.act(state)
                
                # Take action in environment
                next_state, reward, done, info = env.step(action)
                
                # Store experience in replay buffer
                self.rl_agent.store_experience(state, action, reward, next_state, done)
                
                # Update state and total reward
                state = next_state
                total_reward += reward
                
                # Train on batch of experiences
                if len(self.rl_agent.replay_buffer) >= batch_size:
                    self.rl_agent.train(batch_size)
            
            # Log episode results
            if (episode + 1) % 10 == 0:
                self.logger.info(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")
                
        self.logger.info("Training completed")
        
        # Save trained model if path is provided
        model_path = self.config.get('model_path')
        if model_path:
            self.rl_agent.save(model_path)
            self.logger.info(f"Model saved to {model_path}")

    def _load_indicator_configs(self):
        """Laad indicator configuraties uit de config."""
        self.indicator_configs = self.config['strategy']['indicators']
        self.logger.info("Indicator configuraties geladen:")
        for indicator, config in self.indicator_configs.items():
            self.logger.info(f"{indicator}: {config}")
