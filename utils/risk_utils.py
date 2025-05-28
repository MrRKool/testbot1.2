import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_risk_param(config: Dict[str, Any], symbol: str, param: str) -> Optional[Any]:
    """
    Haal een risk parameter op uit de config.
    Eerst wordt gekeken of er een symbol-specifieke waarde is, anders wordt de globale waarde gebruikt.
    """
    try:
        # Eerst symbol-specifieke waarde ophalen
        if symbol in config.get('symbols', {}):
            symbol_config = config['symbols'][symbol]
            if param in symbol_config:
                return symbol_config[param]
        
        # Anders globale waarde uit risk blok
        risk_config = config.get('risk', {})
        if param in risk_config:
            return risk_config[param]
        
        # Als parameter niet gevonden, log een waarschuwing
        logger.warning(f"Risk parameter '{param}' niet gevonden voor {symbol}")
        return None
        
    except Exception as e:
        logger.error(f"Fout bij ophalen risk parameter '{param}' voor {symbol}: {e}")
        return None 