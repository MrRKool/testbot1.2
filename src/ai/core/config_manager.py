import logging
from typing import Dict, List, Any, Optional
import os
import yaml
import json
from datetime import datetime, timedelta

class ConfigManager:
    """Manages configuration for AI components."""
    
    def __init__(self, config_dir: str = 'config'):
        self.config_dir = config_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize configs
        self.configs = {}
        self.config_timestamps = {}
        self._load_configs()
        
    def _load_configs(self):
        """Load all configuration files."""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Load each config file
            for filename in os.listdir(self.config_dir):
                if filename.endswith(('.yaml', '.yml', '.json')):
                    config_name = os.path.splitext(filename)[0]
                    self.load_config(config_name)
                    
        except Exception as e:
            self.logger.error(f"Error loading configs: {str(e)}")
            
    def load_config(self, config_name: str) -> Dict:
        """Load a specific configuration file."""
        try:
            # Try YAML first
            yaml_path = os.path.join(self.config_dir, f"{config_name}.yaml")
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.configs[config_name] = config
                    self.config_timestamps[config_name] = datetime.fromtimestamp(os.path.getmtime(yaml_path))
                    return config
                    
            # Try YML
            yml_path = os.path.join(self.config_dir, f"{config_name}.yml")
            if os.path.exists(yml_path):
                with open(yml_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.configs[config_name] = config
                    self.config_timestamps[config_name] = datetime.fromtimestamp(os.path.getmtime(yml_path))
                    return config
                    
            # Try JSON
            json_path = os.path.join(self.config_dir, f"{config_name}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    config = json.load(f)
                    self.configs[config_name] = config
                    self.config_timestamps[config_name] = datetime.fromtimestamp(os.path.getmtime(json_path))
                    return config
                    
            self.logger.error(f"Config file not found: {config_name}")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error loading config {config_name}: {str(e)}")
            return {}
            
    def save_config(self, config_name: str, config: Dict, format: str = 'yaml') -> bool:
        """Save a configuration file."""
        try:
            # Create config directory if it doesn't exist
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Save based on format
            if format == 'yaml':
                filepath = os.path.join(self.config_dir, f"{config_name}.yaml")
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            elif format == 'json':
                filepath = os.path.join(self.config_dir, f"{config_name}.json")
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                self.logger.error(f"Unsupported config format: {format}")
                return False
                
            # Update configs
            self.configs[config_name] = config
            self.config_timestamps[config_name] = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving config {config_name}: {str(e)}")
            return False
            
    def get_config(self, config_name: str) -> Dict:
        """Get a configuration."""
        try:
            # Check if config is loaded
            if config_name not in self.configs:
                return self.load_config(config_name)
                
            # Check if config file has been modified
            if config_name in self.config_timestamps:
                filepath = os.path.join(self.config_dir, f"{config_name}.yaml")
                if os.path.exists(filepath):
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if mtime > self.config_timestamps[config_name]:
                        return self.load_config(config_name)
                        
            return self.configs[config_name]
            
        except Exception as e:
            self.logger.error(f"Error getting config {config_name}: {str(e)}")
            return {}
            
    def update_config(self, config_name: str, updates: Dict) -> bool:
        """Update a configuration."""
        try:
            # Get current config
            config = self.get_config(config_name)
            if not config:
                return False
                
            # Update config
            config.update(updates)
            
            # Save updated config
            return self.save_config(config_name, config)
            
        except Exception as e:
            self.logger.error(f"Error updating config {config_name}: {str(e)}")
            return False
            
    def get_config_info(self, config_name: str) -> Dict:
        """Get information about a configuration."""
        try:
            if config_name not in self.configs:
                return {}
                
            return {
                'name': config_name,
                'last_modified': self.config_timestamps.get(config_name),
                'keys': list(self.configs[config_name].keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting config info for {config_name}: {str(e)}")
            return {}
            
    def cleanup_old_configs(self, max_age: timedelta = timedelta(days=30)):
        """Remove old configuration files."""
        try:
            if not os.path.exists(self.config_dir):
                return
                
            # Get current time
            now = datetime.now()
            
            # Check each config file
            for filename in os.listdir(self.config_dir):
                if filename.endswith(('.yaml', '.yml', '.json')):
                    filepath = os.path.join(self.config_dir, filename)
                    file_age = now - datetime.fromtimestamp(os.path.getmtime(filepath))
                    
                    # Remove old files
                    if file_age > max_age:
                        os.remove(filepath)
                        config_name = os.path.splitext(filename)[0]
                        if config_name in self.configs:
                            del self.configs[config_name]
                        if config_name in self.config_timestamps:
                            del self.config_timestamps[config_name]
                            
        except Exception as e:
            self.logger.error(f"Error cleaning up old configs: {str(e)}") 