import logging
from typing import Dict, List, Any, Optional, Callable
import os
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict

class EventManager:
    """Manages events for AI components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Event settings
        self.event_dir = config.get('event_dir', 'events')
        self.max_events = config.get('max_events', 1000)
        self.max_event_age = timedelta(days=config.get('max_event_age_days', 7))
        
        # Initialize event tracking
        self.events = []
        self.handlers = defaultdict(list)
        self._load_events()
        
    def _load_events(self):
        """Load existing events."""
        try:
            # Create event directory if it doesn't exist
            os.makedirs(self.event_dir, exist_ok=True)
            
            # Load event file
            event_file = os.path.join(self.event_dir, 'events.json')
            if os.path.exists(event_file):
                with open(event_file, 'r') as f:
                    self.events = json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Error loading events: {str(e)}")
            
    def _save_events(self):
        """Save events to file."""
        try:
            # Create event directory if it doesn't exist
            os.makedirs(self.event_dir, exist_ok=True)
            
            # Save events
            event_file = os.path.join(self.event_dir, 'events.json')
            with open(event_file, 'w') as f:
                json.dump(self.events, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving events: {str(e)}")
            
    def emit(self, event_type: str, data: Dict = None):
        """Emit an event."""
        try:
            # Create event
            event = {
                'timestamp': datetime.now().isoformat(),
                'type': event_type,
                'data': data or {}
            }
            
            # Add to events
            self.events.append(event)
            
            # Save events
            self._save_events()
            
            # Call handlers
            for handler in self.handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(event))
                    else:
                        handler(event)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {str(e)}")
                    
            # Cleanup if needed
            self.cleanup_old_events()
            
        except Exception as e:
            self.logger.error(f"Error emitting event: {str(e)}")
            
    def on(self, event_type: str, handler: Callable):
        """Register an event handler."""
        try:
            if handler not in self.handlers[event_type]:
                self.handlers[event_type].append(handler)
                
        except Exception as e:
            self.logger.error(f"Error registering event handler: {str(e)}")
            
    def off(self, event_type: str, handler: Callable):
        """Unregister an event handler."""
        try:
            if handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)
                
        except Exception as e:
            self.logger.error(f"Error unregistering event handler: {str(e)}")
            
    def get_events(self, event_type: str = None, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get filtered events."""
        try:
            filtered_events = self.events
            
            # Apply filters
            if event_type:
                filtered_events = [e for e in filtered_events if e['type'] == event_type]
            if start_date:
                filtered_events = [e for e in filtered_events if datetime.fromisoformat(e['timestamp']) >= start_date]
            if end_date:
                filtered_events = [e for e in filtered_events if datetime.fromisoformat(e['timestamp']) <= end_date]
                
            return filtered_events
            
        except Exception as e:
            self.logger.error(f"Error getting events: {str(e)}")
            return []
            
    def get_event_stats(self) -> Dict:
        """Get statistics about events."""
        try:
            stats = {
                'total_events': len(self.events),
                'event_types': {},
                'handlers': {k: len(v) for k, v in self.handlers.items()},
                'oldest_event': min((e['timestamp'] for e in self.events), default=None),
                'newest_event': max((e['timestamp'] for e in self.events), default=None)
            }
            
            # Count event types
            for event in self.events:
                event_type = event.get('type', 'unknown')
                stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting event stats: {str(e)}")
            return {}
            
    def cleanup_old_events(self):
        """Remove old events."""
        try:
            # Get current time
            now = datetime.now()
            
            # Filter out old events
            self.events = [
                e for e in self.events
                if datetime.fromisoformat(e['timestamp']) > now - self.max_event_age
            ]
            
            # Limit total events
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
                
            # Save events
            self._save_events()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old events: {str(e)}")
            
    def export_events(self, filepath: str, event_type: str = None, start_date: datetime = None, end_date: datetime = None) -> bool:
        """Export events to file."""
        try:
            # Get filtered events
            events = self.get_events(event_type, start_date, end_date)
            
            # Export to file
            with open(filepath, 'w') as f:
                json.dump(events, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting events: {str(e)}")
            return False
            
    async def wait_for_event(self, event_type: str, timeout: float = None) -> Dict:
        """Wait for an event."""
        try:
            # Create future
            future = asyncio.Future()
            
            # Create handler
            def handler(event):
                if not future.done():
                    future.set_result(event)
                    
            # Register handler
            self.on(event_type, handler)
            
            try:
                # Wait for event
                return await asyncio.wait_for(future, timeout)
            finally:
                # Unregister handler
                self.off(event_type, handler)
                
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout waiting for event: {event_type}")
            return {}
        except Exception as e:
            self.logger.error(f"Error waiting for event: {str(e)}")
            return {} 