import logging
import asyncio
from typing import Dict, List, Optional, Union
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import telegram
import json
import os
from enum import Enum
from dataclasses import dataclass
from utils.shared.enums import AlertType

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Alert:
    type: AlertType
    level: AlertLevel
    message: str
    timestamp: datetime
    data: Optional[Dict] = None

class AlertSystem:
    """Comprehensive alert system for the trading bot."""
    
    def __init__(self,
                 email_config: Optional[Dict] = None,
                 telegram_config: Optional[Dict] = None,
                 alert_thresholds: Optional[Dict] = None):
        
        self.logger = logging.getLogger(__name__)
        self.alerts: List[Alert] = []
        
        # Initialize alert channels
        self.email_config = email_config
        self.telegram_config = telegram_config
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        
        # Initialize alert history
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000
        
        # Initialize alert cooldowns
        self.alert_cooldowns: Dict[str, datetime] = {}
        self.cooldown_periods = {
            AlertType.TRADE: 300,  # 5 minutes
            AlertType.RISK: 60,    # 1 minute
            AlertType.SYSTEM: 300, # 5 minutes
            AlertType.PERFORMANCE: 3600,  # 1 hour
            AlertType.MARKET: 300  # 5 minutes
        }
        
    def _default_thresholds(self) -> Dict:
        """Default alert thresholds."""
        return {
            "max_drawdown": 0.15,  # 15%
            "min_margin_ratio": 0.5,  # 50%
            "max_position_size": 0.2,  # 20%
            "max_daily_loss": 0.05,  # 5%
            "min_win_rate": 0.4,  # 40%
            "max_slippage": 0.002,  # 0.2%
            "min_liquidity": 1000000,  # $1M
            "max_spread": 0.002,  # 0.2%
            "max_volatility": 0.05,  # 5%
            "min_volume": 100000  # $100K
        }
        
    async def send_alert(self,
                        alert_type: AlertType,
                        level: AlertLevel,
                        message: str,
                        data: Optional[Dict] = None) -> None:
        """Send alert through configured channels."""
        try:
            # Create alert
            alert = Alert(
                type=alert_type,
                level=level,
                message=message,
                timestamp=datetime.now(),
                data=data
            )
            
            # Check cooldown
            if not self._check_cooldown(alert):
                return
                
            # Add to history
            self._add_to_history(alert)
            
            # Send through channels
            await self._send_email(alert)
            await self._send_telegram(alert)
            
            # Log alert
            self.logger.info(f"Alert sent: {alert.type.value} - {alert.level.value} - {alert.message}")
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            
    def _check_cooldown(self, alert: Alert) -> bool:
        """Check if alert is in cooldown period."""
        try:
            alert_key = f"{alert.type.value}_{alert.level.value}"
            last_alert = self.alert_cooldowns.get(alert_key)
            
            if last_alert:
                cooldown = self.cooldown_periods[alert.type]
                if (datetime.now() - last_alert).total_seconds() < cooldown:
                    return False
                    
            self.alert_cooldowns[alert_key] = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking alert cooldown: {e}")
            return True
            
    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history."""
        try:
            self.alert_history.append(alert)
            
            # Trim history if too long
            if len(self.alert_history) > self.max_history_size:
                self.alert_history = self.alert_history[-self.max_history_size:]
                
        except Exception as e:
            self.logger.error(f"Error adding alert to history: {e}")
            
    async def _send_email(self, alert: Alert) -> None:
        """Send alert via email."""
        try:
            if not self.email_config:
                return
                
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"Trading Bot Alert: {alert.type.value} - {alert.level.value}"
            
            # Create body
            body = f"""
            Alert Type: {alert.type.value}
            Level: {alert.level.value}
            Time: {alert.timestamp}
            Message: {alert.message}
            """
            
            if alert.data:
                body += f"\nData: {json.dumps(alert.data, indent=2)}"
                
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
                
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
            
    async def _send_telegram(self, alert: Alert) -> None:
        """Send alert via Telegram."""
        try:
            if not self.telegram_config:
                return
                
            # Create message
            message = f"""
🚨 *Trading Bot Alert*
Type: {alert.type.value}
Level: {alert.level.value}
Time: {alert.timestamp}
Message: {alert.message}
            """
            
            if alert.data:
                message += f"\nData: {json.dumps(alert.data, indent=2)}"
                
            # Send message
            bot = telegram.Bot(token=self.telegram_config['token'])
            await bot.send_message(
                chat_id=self.telegram_config['chat_id'],
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {e}")
            
    async def check_risk_limits(self, metrics: Dict) -> None:
        """Check risk metrics against thresholds."""
        try:
            # Check drawdown
            if metrics.get('drawdown', 0) > self.alert_thresholds['max_drawdown']:
                await self.send_alert(
                    AlertType.RISK,
                    AlertLevel.WARNING,
                    f"Drawdown exceeded threshold: {metrics['drawdown']*100:.1f}%",
                    metrics
                )
                
            # Check margin ratio
            if metrics.get('margin_ratio', 1) < self.alert_thresholds['min_margin_ratio']:
                await self.send_alert(
                    AlertType.RISK,
                    AlertLevel.CRITICAL,
                    f"Margin ratio below threshold: {metrics['margin_ratio']*100:.1f}%",
                    metrics
                )
                
            # Check position size
            if metrics.get('position_size', 0) > self.alert_thresholds['max_position_size']:
                await self.send_alert(
                    AlertType.RISK,
                    AlertLevel.WARNING,
                    f"Position size exceeded threshold: {metrics['position_size']*100:.1f}%",
                    metrics
                )
                
            # Check daily loss
            if metrics.get('daily_loss', 0) > self.alert_thresholds['max_daily_loss']:
                await self.send_alert(
                    AlertType.RISK,
                    AlertLevel.CRITICAL,
                    f"Daily loss exceeded threshold: {metrics['daily_loss']*100:.1f}%",
                    metrics
                )
                
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            
    async def check_performance_metrics(self, metrics: Dict) -> None:
        """Check performance metrics against thresholds."""
        try:
            # Check win rate
            if metrics.get('win_rate', 1) < self.alert_thresholds['min_win_rate']:
                await self.send_alert(
                    AlertType.PERFORMANCE,
                    AlertLevel.WARNING,
                    f"Win rate below threshold: {metrics['win_rate']*100:.1f}%",
                    metrics
                )
                
            # Check slippage
            if metrics.get('slippage', 0) > self.alert_thresholds['max_slippage']:
                await self.send_alert(
                    AlertType.PERFORMANCE,
                    AlertLevel.WARNING,
                    f"Slippage exceeded threshold: {metrics['slippage']*100:.3f}%",
                    metrics
                )
                
        except Exception as e:
            self.logger.error(f"Error checking performance metrics: {e}")
            
    async def check_market_conditions(self, market_data: Dict) -> None:
        """Check market conditions against thresholds."""
        try:
            # Check liquidity
            if market_data.get('liquidity', float('inf')) < self.alert_thresholds['min_liquidity']:
                await self.send_alert(
                    AlertType.MARKET,
                    AlertLevel.WARNING,
                    f"Low liquidity: ${market_data['liquidity']:,.2f}",
                    market_data
                )
                
            # Check spread
            if market_data.get('spread', 0) > self.alert_thresholds['max_spread']:
                await self.send_alert(
                    AlertType.MARKET,
                    AlertLevel.WARNING,
                    f"High spread: {market_data['spread']*100:.3f}%",
                    market_data
                )
                
            # Check volatility
            if market_data.get('volatility', 0) > self.alert_thresholds['max_volatility']:
                await self.send_alert(
                    AlertType.MARKET,
                    AlertLevel.WARNING,
                    f"High volatility: {market_data['volatility']*100:.1f}%",
                    market_data
                )
                
            # Check volume
            if market_data.get('volume', 0) < self.alert_thresholds['min_volume']:
                await self.send_alert(
                    AlertType.MARKET,
                    AlertLevel.WARNING,
                    f"Low volume: ${market_data['volume']:,.2f}",
                    market_data
                )
                
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {e}")
            
    async def check_system_health(self, system_metrics: Dict) -> None:
        """Check system health metrics."""
        try:
            # Check CPU usage
            if system_metrics.get('cpu_usage', 0) > 80:
                await self.send_alert(
                    AlertType.SYSTEM,
                    AlertLevel.WARNING,
                    f"High CPU usage: {system_metrics['cpu_usage']}%",
                    system_metrics
                )
                
            # Check memory usage
            if system_metrics.get('memory_usage', 0) > 80:
                await self.send_alert(
                    AlertType.SYSTEM,
                    AlertLevel.WARNING,
                    f"High memory usage: {system_metrics['memory_usage']}%",
                    system_metrics
                )
                
            # Check API latency
            if system_metrics.get('api_latency', 0) > 1000:
                await self.send_alert(
                    AlertType.SYSTEM,
                    AlertLevel.WARNING,
                    f"High API latency: {system_metrics['api_latency']}ms",
                    system_metrics
                )
                
            # Check error rate
            if system_metrics.get('error_rate', 0) > 0.01:
                await self.send_alert(
                    AlertType.SYSTEM,
                    AlertLevel.ERROR,
                    f"High error rate: {system_metrics['error_rate']*100:.1f}%",
                    system_metrics
                )
                
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            
    def get_alert_history(self,
                         alert_type: Optional[AlertType] = None,
                         level: Optional[AlertLevel] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Alert]:
        """Get filtered alert history."""
        try:
            filtered = self.alert_history
            
            if alert_type:
                filtered = [a for a in filtered if a.type == alert_type]
                
            if level:
                filtered = [a for a in filtered if a.level == level]
                
            if start_time:
                filtered = [a for a in filtered if a.timestamp >= start_time]
                
            if end_time:
                filtered = [a for a in filtered if a.timestamp <= end_time]
                
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error getting alert history: {e}")
            return []
            
    def clear_alert_history(self) -> None:
        """Clear alert history."""
        try:
            self.alert_history = []
            self.logger.info("Alert history cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing alert history: {e}")
            
    def update_alert_thresholds(self, new_thresholds: Dict) -> None:
        """Update alert thresholds."""
        try:
            self.alert_thresholds.update(new_thresholds)
            self.logger.info("Alert thresholds updated")
            
        except Exception as e:
            self.logger.error(f"Error updating alert thresholds: {e}")
            
    def update_cooldown_periods(self, new_periods: Dict[AlertType, int]) -> None:
        """Update alert cooldown periods."""
        try:
            self.cooldown_periods.update(new_periods)
            self.logger.info("Alert cooldown periods updated")
            
        except Exception as e:
            self.logger.error(f"Error updating cooldown periods: {e}") 