import logging
import os
import time
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.exchange.exchange import Exchange
from src.strategy.trading_strategy import TradingStrategy
from src.risk.risk_manager import RiskManager
from src.utils.data_loader import DataLoader
from src.utils.config_loader import load_config

@dataclass
class BotConfig:
    """Configuration for the trading bot."""
    symbols: List[str]
    timeframes: List[str]
    update_interval: int = 1
    polling_interval: int = 1
    report_interval: int = 3600
    max_open_trades: int = 3
    max_daily_trades: int = 10
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.15
    min_risk_reward: float = 2.0
    base_risk_per_trade: float = 0.01

class BotError(Exception):
    """Base exception for bot errors."""
    pass

class BotConfigError(BotError):
    """Exception for configuration errors."""
    pass

class BotAPIError(BotError):
    """Exception for API errors."""
    pass

class TradingBot:
    """Main trading bot class that coordinates all components."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the trading bot."""
        try:
            self.logger = logging.getLogger(__name__)
            self.config = load_config(config_path)
            self.bot_config = BotConfig(
                symbols=list(self.config['symbols'].keys()),
                timeframes=self.config['trading']['timeframes']
            )
            
            # Initialize components
            self.data_loader = DataLoader(self.config)
            self.exchange = Exchange(self.config)
            self.strategy = TradingStrategy(self.config)
            self.risk_manager = RiskManager(self.config)
            
            # Initialize state
            self.active_trades = {}
            self.last_update = {}
            self.is_running = False
            
            self.logger.info("Trading bot initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing trading bot: {str(e)}")
            raise BotConfigError(f"Failed to initialize bot: {str(e)}")
            
    async def initialize(self):
        """Initialize all components."""
        try:
            await self.data_loader.initialize()
            await self.exchange.initialize()
            self.risk_manager.initialize(self.config['backtest']['initial_capital'])
            self.logger.info("Trading bot started")
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise BotError(f"Failed to initialize components: {str(e)}")

    async def close(self):
        """Close all components."""
        try:
            await self.data_loader.close()
            await self.exchange.close()
            self.logger.info("Trading bot stopped")
        except Exception as e:
            self.logger.error(f"Error closing components: {str(e)}")
            raise BotError(f"Failed to close components: {str(e)}")

    async def _process_trading_cycle(self):
        """Process one trading cycle."""
        try:
            for symbol in self.bot_config.symbols:
                # Get latest data for all timeframes
                data = {}
                for timeframe in self.bot_config.timeframes:
                    df = self.data_loader.get_latest_data(symbol, timeframe)
                    if df is not None and not df.empty:
                        data[timeframe] = df
                
                if not data:
                    continue
                
                # Generate trading signals
                signals = self.strategy.get_trading_signal(symbol)
                if not signals:
                    continue
                
                # Check risk limits
                if not self.risk_manager.check_risk_limits(symbol, signals):
                    continue
                
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(symbol, signals)
                if position_size <= 0:
                    continue
                
                # Execute trade
                if signals['action'] == 'buy':
                    await self._execute_buy(symbol, position_size, signals)
                elif signals['action'] == 'sell':
                    await self._execute_sell(symbol, position_size, signals)
                
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {str(e)}")

    async def _execute_buy(self, symbol: str, size: float, signals: Dict):
        """Execute a buy order."""
        try:
            # Check if we already have a position
            if symbol in self.active_trades:
                return
            
            # Place order
            order = await self.exchange.place_order(
                symbol=symbol,
                side='buy',
                quantity=size,
                order_type='market'
            )
            
            if order and order['status'] == 'filled':
                # Update active trades
                self.active_trades[symbol] = {
                    'entry_price': order['price'],
                    'size': size,
                    'entry_time': datetime.now(),
                    'signals': signals
                }
                
                # Set stop loss and take profit
                stop_loss = self.risk_manager.calculate_stop_loss(symbol, signals)
                take_profit = self.risk_manager.calculate_take_profit(symbol, signals)
                
                await self.exchange.place_order(
                    symbol=symbol,
                    side='sell',
                    quantity=size,
                    order_type='stop',
                    stop_price=stop_loss
                )
                
                await self.exchange.place_order(
                    symbol=symbol,
                    side='sell',
                    quantity=size,
                    order_type='limit',
                    price=take_profit
                )
                
                self.logger.info(f"Executed buy order for {symbol}: {order}")
                
        except Exception as e:
            self.logger.error(f"Error executing buy order: {str(e)}")

    async def _execute_sell(self, symbol: str, size: float, signals: Dict):
        """Execute a sell order."""
        try:
            # Check if we have a position
            if symbol not in self.active_trades:
                return
            
            # Place order
            order = await self.exchange.place_order(
                symbol=symbol,
                side='sell',
                quantity=size,
                order_type='market'
            )
            
            if order and order['status'] == 'filled':
                # Update active trades
                trade = self.active_trades.pop(symbol)
                profit = (order['price'] - trade['entry_price']) * size
                
                self.logger.info(f"Executed sell order for {symbol}: {order}")
                self.logger.info(f"Trade profit: {profit}")
                
        except Exception as e:
            self.logger.error(f"Error executing sell order: {str(e)}")

    async def run(self):
        """Run the trading bot."""
        try:
            await self.initialize()
            self.is_running = True
            
            while self.is_running:
                await self._process_trading_cycle()
                await asyncio.sleep(self.bot_config.polling_interval)
                
        except Exception as e:
            self.logger.error(f"Error running trading bot: {str(e)}")
        finally:
            await self.close()

async def main():
    """Main entry point."""
    try:
        # Load configuration
        config_path = 'config/config.yaml'
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Initialize and run trading bot
        bot = TradingBot(config_path)
        await bot.run()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 