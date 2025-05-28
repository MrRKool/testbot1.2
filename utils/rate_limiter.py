import time
import logging
from typing import Dict, Optional
from threading import Lock

class RateLimiter:
    """Rate limiter voor API calls met verbeterde implementatie."""
    
    def __init__(self, calls_per_second: int = 1, calls_per_minute: int = 30, calls_per_hour: int = 500):
        """Initialiseer de rate limiter met thread-safe implementatie."""
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        
        # Thread-safe opslag van calls
        self._lock = Lock()
        self.calls: Dict[str, list] = {
            "second": [],
            "minute": [],
            "hour": []
        }
        
        # Minimum wachttijd tussen calls
        self.min_wait_time = 1.0 / calls_per_second if calls_per_second > 0 else 0
        
        logging.info(f"RateLimiter geïnitialiseerd met limieten: {calls_per_second}/s, {calls_per_minute}/m, {calls_per_hour}/h")
        
    def _cleanup_old_calls(self):
        """Verwijder oude calls uit de geschiedenis met thread-safety."""
        try:
            with self._lock:
                now = time.time()
                
                # Verwijder calls ouder dan 1 seconde
                self.calls["second"] = [t for t in self.calls["second"] if now - t < 1.0]
                
                # Verwijder calls ouder dan 1 minuut
                self.calls["minute"] = [t for t in self.calls["minute"] if now - t < 60.0]
                
                # Verwijder calls ouder dan 1 uur
                self.calls["hour"] = [t for t in self.calls["hour"] if now - t < 3600.0]
        except Exception as e:
            logging.error(f"Fout bij opschonen oude calls: {e}")
            
    def _get_wait_time(self) -> float:
        """Bereken hoe lang we moeten wachten voordat we een nieuwe call kunnen maken."""
        try:
            with self._lock:
                self._cleanup_old_calls()
                now = time.time()
                
                # Check second limiet
                if len(self.calls["second"]) >= self.calls_per_second:
                    oldest_call = min(self.calls["second"])
                    wait_time = 1.0 - (now - oldest_call)
                    if wait_time > 0:
                        return max(wait_time, self.min_wait_time)
                        
                # Check minute limiet
                if len(self.calls["minute"]) >= self.calls_per_minute:
                    oldest_call = min(self.calls["minute"])
                    wait_time = 60.0 - (now - oldest_call)
                    if wait_time > 0:
                        return max(wait_time, self.min_wait_time)
                        
                # Check hour limiet
                if len(self.calls["hour"]) >= self.calls_per_hour:
                    oldest_call = min(self.calls["hour"])
                    wait_time = 3600.0 - (now - oldest_call)
                    if wait_time > 0:
                        return max(wait_time, self.min_wait_time)
                        
                return self.min_wait_time
        except Exception as e:
            logging.error(f"Fout bij berekenen wachttijd: {e}")
            return self.min_wait_time
            
    def wait_if_needed(self):
        """Wacht indien nodig om rate limits te respecteren."""
        try:
            wait_time = self._get_wait_time()
            if wait_time > 0:
                logging.debug(f"Rate limiter: wachten {wait_time:.2f} seconden")
                time.sleep(wait_time)
        except Exception as e:
            logging.error(f"Fout bij wachten voor rate limit: {e}")
            
    def add_call(self):
        """Voeg een nieuwe call toe aan de geschiedenis met thread-safety."""
        try:
            with self._lock:
                now = time.time()
                self.calls["second"].append(now)
                self.calls["minute"].append(now)
                self.calls["hour"].append(now)
                
                # Log huidige status
                usage = self.get_current_usage()
                logging.debug(f"Rate limiter status na nieuwe call: {usage}")
        except Exception as e:
            logging.error(f"Fout bij toevoegen nieuwe call: {e}")
            
    def get_current_usage(self) -> Dict[str, int]:
        """Krijg huidige gebruik van de rate limiter."""
        try:
            with self._lock:
                self._cleanup_old_calls()
                return {
                    "second": len(self.calls["second"]),
                    "minute": len(self.calls["minute"]),
                    "hour": len(self.calls["hour"])
                }
        except Exception as e:
            logging.error(f"Fout bij ophalen huidige gebruik: {e}")
            return {"second": 0, "minute": 0, "hour": 0}
            
    def reset(self):
        """Reset de rate limiter."""
        try:
            with self._lock:
                self.calls = {
                    "second": [],
                    "minute": [],
                    "hour": []
                }
                logging.info("Rate limiter gereset")
        except Exception as e:
            logging.error(f"Fout bij resetten rate limiter: {e}") 