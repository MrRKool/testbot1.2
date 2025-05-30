def load_config(config_path: str = 'config/config.yaml') -> dict:
    import yaml
    import os
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Fallback voor exchange name
    if 'exchange' in config:
        if 'name' not in config['exchange'] or not config['exchange']['name']:
            config['exchange']['name'] = 'bybit'
    else:
        config['exchange'] = {'name': 'bybit'}
    return config 