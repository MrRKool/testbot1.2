from utils.shared.enums import TimeFrame, SignalType
from enum import Enum

class OrderType(str, Enum):
    """Type van een order."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"
    VWAP = "VWAP"

class OrderStatus(str, Enum):
    """Status van een order."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

# Verwijder de lokale definities van TimeFrame en SignalType 
# De rest van het bestand blijft ongewijzigd 