import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime, timedelta
import os
import structlog
from structlog import get_logger
import pandas as pd

class EventManager:
    """Beheert events voor de trading bot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger()
        
        # Event settings
        self.max_events = config.get('max_events', 1000)
        self.max_event_age_days = config.get('max_event_age_days', 7)
        self.event_export_path = config.get('event_export_path', 'events')
        
        # Initialize state
        self.events = {}
        self.is_running = False
        
    async def start(self):
        """Start de event manager."""
        try:
            self.is_running = True
            
            # Start event cleanup
            asyncio.create_task(self._cleanup_old_events())
            
            self.logger.info("Event manager started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting event manager: {str(e)}")
            return False
            
    async def stop(self):
        """Stop de event manager."""
        try:
            self.is_running = False
            self.logger.info("Event manager stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping event manager: {str(e)}")
            return False
            
    async def _cleanup_old_events(self):
        """Cleanup old events periodically."""
        while self.is_running:
            try:
                cutoff_date = datetime.now() - timedelta(days=self.max_event_age_days)
                
                # Remove old events
                self.events = {
                    k: v for k, v in self.events.items()
                    if datetime.fromisoformat(v['timestamp']) > cutoff_date
                }
                
                # Export events if needed
                if len(self.events) >= self.max_events:
                    self._export_events()
                    
            except Exception as e:
                self.logger.error(f"Error cleaning up events: {str(e)}")
                
            await asyncio.sleep(3600)  # Cleanup every hour
            
    def _store_event(self, event: Dict):
        """Store event in memory."""
        try:
            timestamp = event['timestamp']
            self.events[timestamp] = event
            
        except Exception as e:
            self.logger.error(f"Error storing event: {str(e)}")
            raise
            
    def _export_events(self):
        """Export events to file."""
        try:
            # Create events directory if it doesn't exist
            os.makedirs(self.event_export_path, exist_ok=True)
            
            # Convert events to DataFrame
            df = pd.DataFrame.from_dict(self.events, orient='index')
            
            # Export to CSV
            filename = f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(os.path.join(self.event_export_path, filename))
            
            # Clear events after export
            self.events.clear()
            
        except Exception as e:
            self.logger.error(f"Error exporting events: {str(e)}")
            raise
            
    def log_trade_event(self, trade_data: Dict):
        """Log trade event."""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'trade',
                'trade_id': trade_data.get('trade_id'),
                'symbol': trade_data.get('symbol'),
                'side': trade_data.get('side'),
                'price': trade_data.get('price'),
                'volume': trade_data.get('volume'),
                'profit_loss': trade_data.get('profit_loss')
            }
            
            self._store_event(event)
            
            # Log event
            self.logger.info(
                "trade_event",
                **event
            )
            
        except Exception as e:
            self.logger.error(f"Error logging trade event: {str(e)}")
            raise
            
    def log_system_event(self, event_type: str, event_data: Dict):
        """Log system event."""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'system',
                'event_type': event_type,
                **event_data
            }
            
            self._store_event(event)
            
            # Log event
            self.logger.info(
                "system_event",
                **event
            )
            
        except Exception as e:
            self.logger.error(f"Error logging system event: {str(e)}")
            raise
            
    def log_model_event(self, model_data: Dict):
        """Log model event."""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': 'model',
                'model_type': model_data.get('model_type'),
                'accuracy': model_data.get('accuracy'),
                'loss': model_data.get('loss'),
                'epoch': model_data.get('epoch')
            }
            
            self._store_event(event)
            
            # Log event
            self.logger.info(
                "model_event",
                **event
            )
            
        except Exception as e:
            self.logger.error(f"Error logging model event: {str(e)}")
            raise
            
    def get_events_summary(self) -> Dict:
        """Get summary of events."""
        try:
            # Group events by type
            trade_events = [e for e in self.events.values() if e['type'] == 'trade']
            system_events = [e for e in self.events.values() if e['type'] == 'system']
            model_events = [e for e in self.events.values() if e['type'] == 'model']
            
            return {
                'total_events': len(self.events),
                'trade_events': len(trade_events),
                'system_events': len(system_events),
                'model_events': len(model_events),
                'latest_events': sorted(
                    self.events.values(),
                    key=lambda x: x['timestamp'],
                    reverse=True
                )[:10]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting events summary: {str(e)}")
            return {} 