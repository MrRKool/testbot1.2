import logging
import traceback
from typing import Optional, Callable, Dict, Any
from datetime import datetime
import json
from pathlib import Path
import os
import sys
from functools import wraps
import time
import random

class ErrorHandler:
    """Handelt errors af en zorgt voor recovery mechanismen."""
    
    def __init__(self, log_dir: str = "logs"):
        self.logger = logging.getLogger(__name__)
        self.log_dir = Path(log_dir)
        self.error_count: Dict[str, int] = {}
        self.last_error_time: Dict[str, float] = {}
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        self.recovery_cooldown = 300  # 5 minuten
        
        # Maak log directory als deze niet bestaat
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def handle_error(self, error: Exception, context: str, 
                    recovery_func: Optional[Callable] = None,
                    **recovery_args) -> bool:
        """Handelt een error af en probeert te herstellen."""
        error_type = type(error).__name__
        error_key = f"{context}_{error_type}"
        
        # Log de error
        self._log_error(error, context)
        
        # Update error statistieken
        self.error_count[error_key] = self.error_count.get(error_key, 0) + 1
        self.last_error_time[error_key] = time.time()
        
        # Probeer te herstellen als er een recovery functie is
        if recovery_func:
            return self._attempt_recovery(error_key, recovery_func, **recovery_args)
            
        return False
        
    def _log_error(self, error: Exception, context: str):
        """Log een error met stack trace."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "stack_trace": traceback.format_exc()
        }
        
        # Log naar bestand
        log_file = self.log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, 'a') as f:
            json.dump(error_info, f)
            f.write('\n')
            
        # Log naar console
        self.logger.error(f"Error in {context}: {error}")
        self.logger.debug(traceback.format_exc())
        
    def _attempt_recovery(self, error_key: str, recovery_func: Callable, 
                         **recovery_args) -> bool:
        """Probeert te herstellen van een error."""
        current_time = time.time()
        last_error = self.last_error_time.get(error_key, 0)
        
        # Check cooldown
        if current_time - last_error < self.recovery_cooldown:
            self.logger.warning(f"Recovery cooldown actief voor {error_key}")
            return False
            
        # Check max attempts
        attempts = self.recovery_attempts.get(error_key, 0)
        if attempts >= self.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts bereikt voor {error_key}")
            return False
            
        try:
            # Voeg random delay toe voor rate limiting
            time.sleep(random.uniform(1, 3))
            
            # Probeer te herstellen
            success = recovery_func(**recovery_args)
            
            if success:
                self.logger.info(f"Recovery succesvol voor {error_key}")
                self.recovery_attempts[error_key] = 0
                return True
            else:
                self.logger.warning(f"Recovery gefaald voor {error_key}")
                self.recovery_attempts[error_key] = attempts + 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error tijdens recovery voor {error_key}: {e}")
            self.recovery_attempts[error_key] = attempts + 1
            return False
            
    def get_error_stats(self) -> Dict[str, Any]:
        """Haal error statistieken op."""
        return {
            "error_counts": self.error_count,
            "last_errors": {k: datetime.fromtimestamp(v).isoformat() 
                          for k, v in self.last_error_time.items()},
            "recovery_attempts": self.recovery_attempts
        }
        
    def reset_stats(self):
        """Reset alle error statistieken."""
        self.error_count.clear()
        self.last_error_time.clear()
        self.recovery_attempts.clear()
        
def error_handler_decorator(error_handler: ErrorHandler, context: str):
    """Decorator voor error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, context)
                raise
        return wrapper
    return decorator 