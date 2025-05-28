import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

class MovingAverageCrossStrategy:
    """Simple moving average crossover strategy."""
    
    def __init__(self,
                 short_window: int = 20,
                 long_window: int = 50,
                 position_size: float = 0.1):  # 10% of capital per trade
        
        self.short_window = short_window
        self.long_window = long_window
        self.position_size = position_size
        self.logger = logging.getLogger(__name__)
        
    def __call__(self, data: pd.DataFrame, positions: Dict[str, float], regime: str) -> Dict:
        """Generate trading signals."""
        try:
            if len(data) < self.long_window:
                return {}
                
            # Calculate moving averages
            short_ma = data['close'].rolling(window=self.short_window).mean()
            long_ma = data['close'].rolling(window=self.long_window).mean()
            
            # Get current values
            current_price = data['close'].iloc[-1]
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            previous_short_ma = short_ma.iloc[-2]
            previous_long_ma = long_ma.iloc[-2]
            
            # Generate signals
            signals = {}
            
            # Check for crossover
            if previous_short_ma <= previous_long_ma and current_short_ma > current_long_ma:
                # Bullish crossover
                if 'BTCUSDT' not in positions or positions['BTCUSDT'] <= 0:
                    quantity = (self.position_size * 10000) / current_price  # Assuming 10000 capital
                    signals['BTCUSDT'] = {
                        'action': 'buy',
                        'quantity': quantity
                    }
                    
            elif previous_short_ma >= previous_long_ma and current_short_ma < current_long_ma:
                # Bearish crossover
                if 'BTCUSDT' in positions and positions['BTCUSDT'] > 0:
                    signals['BTCUSDT'] = {
                        'action': 'sell',
                        'quantity': positions['BTCUSDT']
                    }
                    
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return {} 