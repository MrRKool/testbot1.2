import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime, timedelta
import os
import structlog
from structlog import get_logger
import pandas as pd
import traceback
import sys

class ErrorManager:
    """Beheert errors voor de trading bot."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = get_logger()
        
        # Error settings
        self.max_errors = config.get('max_errors', 1000)
        self.max_error_age_days = config.get('max_error_age_days', 7)
        self.error_export_path = config.get('error_export_path', 'errors')
        self.error_notification = config.get('error_notification', True)
        
        # Initialize state
        self.errors = {}
        self.is_running = False
        
    async def start(self):
        """Start de error manager."""
        try:
            self.is_running = True
            
            # Start error cleanup
            asyncio.create_task(self._cleanup_old_errors())
            
            # Set up global exception handler
            sys.excepthook = self._handle_uncaught_exception
            
            self.logger.info("Error manager started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting error manager: {str(e)}")
            return False
            
    async def stop(self):
        """Stop de error manager."""
        try:
            self.is_running = False
            self.logger.info("Error manager stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping error manager: {str(e)}")
            return False
            
    async def _cleanup_old_errors(self):
        """Cleanup old errors periodically."""
        while self.is_running:
            try:
                cutoff_date = datetime.now() - timedelta(days=self.max_error_age_days)
                
                # Remove old errors
                self.errors = {
                    k: v for k, v in self.errors.items()
                    if datetime.fromisoformat(v['timestamp']) > cutoff_date
                }
                
                # Export errors if needed
                if len(self.errors) >= self.max_errors:
                    self._export_errors()
                    
            except Exception as e:
                self.logger.error(f"Error cleaning up errors: {str(e)}")
                
            await asyncio.sleep(3600)  # Cleanup every hour
            
    def _store_error(self, error: Dict):
        """Store error in memory."""
        try:
            timestamp = error['timestamp']
            self.errors[timestamp] = error
            
        except Exception as e:
            self.logger.error(f"Error storing error: {str(e)}")
            raise
            
    def _export_errors(self):
        """Export errors to file."""
        try:
            # Create errors directory if it doesn't exist
            os.makedirs(self.error_export_path, exist_ok=True)
            
            # Convert errors to DataFrame
            df = pd.DataFrame.from_dict(self.errors, orient='index')
            
            # Export to CSV
            filename = f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(os.path.join(self.error_export_path, filename))
            
            # Clear errors after export
            self.errors.clear()
            
        except Exception as e:
            self.logger.error(f"Error exporting errors: {str(e)}")
            raise
            
    def _handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        try:
            error = {
                'timestamp': datetime.now().isoformat(),
                'type': 'uncaught_exception',
                'exception_type': exc_type.__name__,
                'exception_message': str(exc_value),
                'stack_trace': ''.join(traceback.format_tb(exc_traceback))
            }
            
            self._store_error(error)
            
            # Log error
            self.logger.error(
                "uncaught_exception",
                **error
            )
            
            # Send notification if enabled
            if self.error_notification:
                self._send_error_notification(error)
                
        except Exception as e:
            self.logger.error(f"Error handling uncaught exception: {str(e)}")
            
    def log_error(self, error_type: str, error_message: str, stack_trace: Optional[str] = None):
        """Log error."""
        try:
            error = {
                'timestamp': datetime.now().isoformat(),
                'type': error_type,
                'message': error_message,
                'stack_trace': stack_trace or traceback.format_exc()
            }
            
            self._store_error(error)
            
            # Log error
            self.logger.error(
                "error_occurred",
                **error
            )
            
            # Send notification if enabled
            if self.error_notification:
                self._send_error_notification(error)
                
        except Exception as e:
            self.logger.error(f"Error logging error: {str(e)}")
            raise
            
    def _send_error_notification(self, error: Dict):
        """Send error notification."""
        try:
            # TODO: Implement notification system (e.g. email, Slack, Telegram)
            pass
            
        except Exception as e:
            self.logger.error(f"Error sending error notification: {str(e)}")
            
    def get_errors_summary(self) -> Dict:
        """Get summary of errors."""
        try:
            # Group errors by type
            error_types = {}
            for error in self.errors.values():
                error_type = error['type']
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error)
                
            return {
                'total_errors': len(self.errors),
                'error_types': {
                    error_type: len(errors)
                    for error_type, errors in error_types.items()
                },
                'latest_errors': sorted(
                    self.errors.values(),
                    key=lambda x: x['timestamp'],
                    reverse=True
                )[:10]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting errors summary: {str(e)}")
            return {} 