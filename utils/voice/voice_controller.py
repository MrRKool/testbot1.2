import logging
from typing import Dict, Any, Optional
from .voice_interface import VoiceInterface
from utils.trading.rl_agent import RLAgent
from utils.trading.trade_executor import TradeExecutor
import json
import os
from datetime import datetime

class VoiceController:
    """Controller for voice-controlled trading bot."""
    
    def __init__(self, trading_bot, config_path: str = 'config/voice_controller.json'):
        self.logger = logging.getLogger(__name__)
        self.trading_bot = trading_bot
        self.voice_interface = VoiceInterface()
        self._load_config(config_path)
        self._setup_callbacks()
        
    def _load_config(self, config_path: str):
        """Load voice controller configuration."""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {
                'risk_levels': {
                    'low': 0.1,
                    'medium': 0.2,
                    'high': 0.3
                },
                'strategies': {
                    'conservative': {
                        'max_position_size': 0.1,
                        'stop_loss': 0.02,
                        'take_profit': 0.04
                    },
                    'moderate': {
                        'max_position_size': 0.2,
                        'stop_loss': 0.03,
                        'take_profit': 0.06
                    },
                    'aggressive': {
                        'max_position_size': 0.3,
                        'stop_loss': 0.04,
                        'take_profit': 0.08
                    }
                }
            }
            os.makedirs('config', exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
    def _setup_callbacks(self):
        """Setup voice command callbacks."""
        self.voice_interface.register_callback('start_trading', self._handle_start_trading)
        self.voice_interface.register_callback('stop_trading', self._handle_stop_trading)
        self.voice_interface.register_callback('check_status', self._handle_check_status)
        self.voice_interface.register_callback('show_metrics', self._handle_show_metrics)
        self.voice_interface.register_callback('adjust_risk', self._handle_adjust_risk)
        self.voice_interface.register_callback('change_strategy', self._handle_change_strategy)
        
    def _handle_start_trading(self, text: str):
        """Handle start trading command."""
        try:
            self.trading_bot.start()
            self.voice_interface.speak("Trading bot is gestart. Ik begin nu met handelen.")
        except Exception as e:
            self.logger.error(f"Error starting trading: {str(e)}")
            self.voice_interface.speak("Er is een fout opgetreden bij het starten van de trading bot.")
            
    def _handle_stop_trading(self, text: str):
        """Handle stop trading command."""
        try:
            self.trading_bot.stop()
            self.voice_interface.speak("Trading bot is gestopt. Alle posities zijn gesloten.")
        except Exception as e:
            self.logger.error(f"Error stopping trading: {str(e)}")
            self.voice_interface.speak("Er is een fout opgetreden bij het stoppen van de trading bot.")
            
    def _handle_check_status(self, text: str):
        """Handle status check command."""
        try:
            status = self.trading_bot.get_status()
            response = (
                f"De trading bot is {'actief' if status['is_running'] else 'gestopt'}. "
                f"Totale winst: {status['total_profit']:.2f}. "
                f"Open posities: {status['open_positions']}. "
                f"Risiconiveau: {status['risk_level']}."
            )
            self.voice_interface.speak(response)
        except Exception as e:
            self.logger.error(f"Error checking status: {str(e)}")
            self.voice_interface.speak("Ik kan de status niet ophalen op dit moment.")
            
    def _handle_show_metrics(self, text: str):
        """Handle show metrics command."""
        try:
            metrics = self.trading_bot.get_metrics()
            response = (
                f"Hier zijn de prestaties: "
                f"Winst/verlies ratio: {metrics['win_loss_ratio']:.2f}. "
                f"Gemiddelde winst per trade: {metrics['avg_profit']:.2f}. "
                f"Maximale drawdown: {metrics['max_drawdown']:.2f}. "
                f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}."
            )
            self.voice_interface.speak(response)
        except Exception as e:
            self.logger.error(f"Error showing metrics: {str(e)}")
            self.voice_interface.speak("Ik kan de prestaties niet ophalen op dit moment.")
            
    def _handle_adjust_risk(self, text: str):
        """Handle risk adjustment command."""
        try:
            # Extract risk level from text
            risk_level = None
            for level in self.config['risk_levels'].keys():
                if level in text.lower():
                    risk_level = level
                    break
                    
            if risk_level:
                self.trading_bot.set_risk_level(self.config['risk_levels'][risk_level])
                self.voice_interface.speak(f"Risiconiveau is aangepast naar {risk_level}.")
            else:
                self.voice_interface.speak("Ik begrijp niet welk risiconiveau je wilt instellen. "
                                         "Zeg 'laag', 'gemiddeld' of 'hoog'.")
        except Exception as e:
            self.logger.error(f"Error adjusting risk: {str(e)}")
            self.voice_interface.speak("Er is een fout opgetreden bij het aanpassen van het risiconiveau.")
            
    def _handle_change_strategy(self, text: str):
        """Handle strategy change command."""
        try:
            # Extract strategy from text
            strategy = None
            for strat in self.config['strategies'].keys():
                if strat in text.lower():
                    strategy = strat
                    break
                    
            if strategy:
                self.trading_bot.set_strategy(self.config['strategies'][strategy])
                self.voice_interface.speak(f"Strategie is gewijzigd naar {strategy}.")
            else:
                self.voice_interface.speak("Ik begrijp niet welke strategie je wilt gebruiken. "
                                         "Zeg 'conservatief', 'gematigd' of 'agressief'.")
        except Exception as e:
            self.logger.error(f"Error changing strategy: {str(e)}")
            self.voice_interface.speak("Er is een fout opgetreden bij het wijzigen van de strategie.")
            
    def start(self):
        """Start voice controller."""
        self.voice_interface.start()
        
    def stop(self):
        """Stop voice controller."""
        self.voice_interface.stop()
        
    def update_config(self, key: str, value: Any):
        """Update voice controller configuration."""
        if key in self.config:
            self.config[key] = value
            
            # Save updated config
            with open('config/voice_controller.json', 'w') as f:
                json.dump(self.config, f, indent=4) 