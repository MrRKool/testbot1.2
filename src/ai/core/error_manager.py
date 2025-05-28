import logging
from typing import Dict, List, Any, Optional
import os
from datetime import datetime, timedelta
import json
import traceback
import sys

class ErrorManager:
    """Manages errors for AI components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Error settings
        self.error_dir = config.get('error_dir', 'errors')
        self.max_error_age = timedelta(days=config.get('max_error_age_days', 7))
        self.max_errors = config.get('max_errors', 1000)
        
        # Initialize error tracking
        self.errors = []
        self.error_counts = {}
        self._load_errors()
        
    def _load_errors(self):
        """Load existing errors."""
        try:
            # Create error directory if it doesn't exist
            os.makedirs(self.error_dir, exist_ok=True)
            
            # Load error file
            error_file = os.path.join(self.error_dir, 'errors.json')
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    self.errors = json.load(f)
                    
                # Update error counts
                for error in self.errors:
                    error_type = error.get('type', 'unknown')
                    self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                    
        except Exception as e:
            self.logger.error(f"Error loading errors: {str(e)}")
            
    def handle_error(self, error: Exception, component: str, context: Dict = None):
        """Handle an error."""
        try:
            # Get error info
            error_type = type(error).__name__
            error_message = str(error)
            stack_trace = traceback.format_exc()
            
            # Create error entry
            error_entry = {
                'timestamp': datetime.now().isoformat(),
                'component': component,
                'type': error_type,
                'message': error_message,
                'stack_trace': stack_trace,
                'context': context or {}
            }
            
            # Add to errors
            self.errors.append(error_entry)
            
            # Update error count
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Save errors
            self._save_errors()
            
            # Log error
            self.logger.error(f"Error in {component}: {error_message}")
            
            # Cleanup if needed
            self.cleanup_old_errors()
            
        except Exception as e:
            self.logger.error(f"Error handling error: {str(e)}")
            
    def _save_errors(self):
        """Save errors to file."""
        try:
            # Create error directory if it doesn't exist
            os.makedirs(self.error_dir, exist_ok=True)
            
            # Save errors
            error_file = os.path.join(self.error_dir, 'errors.json')
            with open(error_file, 'w') as f:
                json.dump(self.errors, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving errors: {str(e)}")
            
    def get_error_stats(self) -> Dict:
        """Get statistics about errors."""
        try:
            return {
                'total_errors': len(self.errors),
                'error_counts': self.error_counts,
                'components': list(set(e['component'] for e in self.errors)),
                'oldest_error': min((e['timestamp'] for e in self.errors), default=None),
                'newest_error': max((e['timestamp'] for e in self.errors), default=None)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting error stats: {str(e)}")
            return {}
            
    def cleanup_old_errors(self):
        """Remove old errors."""
        try:
            # Get current time
            now = datetime.now()
            
            # Filter out old errors
            self.errors = [
                e for e in self.errors
                if datetime.fromisoformat(e['timestamp']) > now - self.max_error_age
            ]
            
            # Limit total errors
            if len(self.errors) > self.max_errors:
                self.errors = self.errors[-self.max_errors:]
                
            # Update error counts
            self.error_counts = {}
            for error in self.errors:
                error_type = error.get('type', 'unknown')
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
            # Save errors
            self._save_errors()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old errors: {str(e)}")
            
    def get_errors(self, component: str = None, error_type: str = None, start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get filtered errors."""
        try:
            filtered_errors = self.errors
            
            # Apply filters
            if component:
                filtered_errors = [e for e in filtered_errors if e['component'] == component]
            if error_type:
                filtered_errors = [e for e in filtered_errors if e['type'] == error_type]
            if start_date:
                filtered_errors = [e for e in filtered_errors if datetime.fromisoformat(e['timestamp']) >= start_date]
            if end_date:
                filtered_errors = [e for e in filtered_errors if datetime.fromisoformat(e['timestamp']) <= end_date]
                
            return filtered_errors
            
        except Exception as e:
            self.logger.error(f"Error getting errors: {str(e)}")
            return []
            
    def get_error_summary(self, component: str = None) -> Dict:
        """Get summary of errors."""
        try:
            # Get filtered errors
            errors = self.get_errors(component=component)
            
            # Calculate summary
            summary = {
                'total_errors': len(errors),
                'error_types': {},
                'components': {},
                'recent_errors': errors[-10:] if errors else []
            }
            
            # Count error types and components
            for error in errors:
                error_type = error.get('type', 'unknown')
                component = error.get('component', 'unknown')
                
                summary['error_types'][error_type] = summary['error_types'].get(error_type, 0) + 1
                summary['components'][component] = summary['components'].get(component, 0) + 1
                
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting error summary: {str(e)}")
            return {}
            
    def export_errors(self, filepath: str, component: str = None, error_type: str = None, start_date: datetime = None, end_date: datetime = None) -> bool:
        """Export errors to file."""
        try:
            # Get filtered errors
            errors = self.get_errors(component, error_type, start_date, end_date)
            
            # Export to file
            with open(filepath, 'w') as f:
                json.dump(errors, f, indent=2)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting errors: {str(e)}")
            return False 