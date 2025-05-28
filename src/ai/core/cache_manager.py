import logging
from typing import Dict, Any, Optional
import json
import os
from datetime import datetime, timedelta
import threading
from functools import lru_cache

class CacheManager:
    """Manages caching for all AI components."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache_dir = "cache"
        self.cache_ttl = timedelta(hours=1)
        self.lock = threading.Lock()
        
        # Create cache directories
        self._create_cache_dirs()
        
    def _create_cache_dirs(self):
        """Create necessary cache directories."""
        dirs = [
            "sentiment",
            "performance",
            "risk",
            "market",
            "models"
        ]
        for dir_name in dirs:
            os.makedirs(os.path.join(self.cache_dir, dir_name), exist_ok=True)
            
    @lru_cache(maxsize=1000)
    def get_cached_data(self, key: str, category: str) -> Optional[Any]:
        """Get cached data if available and not expired."""
        try:
            cache_file = os.path.join(self.cache_dir, category, f"{key}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - timestamp < self.cache_ttl:
                        return data['value']
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting cached data: {str(e)}")
            return None
            
    def save_to_cache(self, key: str, category: str, value: Any):
        """Save data to cache."""
        try:
            with self.lock:
                cache_file = os.path.join(self.cache_dir, category, f"{key}.json")
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'value': value
                }
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                    
        except Exception as e:
            self.logger.error(f"Error saving to cache: {str(e)}")
            
    def clear_cache(self, category: Optional[str] = None):
        """Clear cache for a specific category or all categories."""
        try:
            if category:
                cache_dir = os.path.join(self.cache_dir, category)
                if os.path.exists(cache_dir):
                    for file in os.listdir(cache_dir):
                        os.remove(os.path.join(cache_dir, file))
            else:
                for dir_name in os.listdir(self.cache_dir):
                    dir_path = os.path.join(self.cache_dir, dir_name)
                    if os.path.isdir(dir_path):
                        for file in os.listdir(dir_path):
                            os.remove(os.path.join(dir_path, file))
                            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            
    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage."""
        try:
            stats = {}
            for dir_name in os.listdir(self.cache_dir):
                dir_path = os.path.join(self.cache_dir, dir_name)
                if os.path.isdir(dir_path):
                    stats[dir_name] = len(os.listdir(dir_path))
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {} 