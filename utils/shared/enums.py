from enum import Enum
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

class TimeFrame(str, Enum):
    """Enum voor verschillende tijdsframes."""
    MINUTE = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR = "1h"
    HOUR_4 = "4h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1M"

class SignalType(str, Enum):
    """Enum voor verschillende signaal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"

class ReportType(str, Enum):
    """Enum voor rapport types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"

class ReportFormat(str, Enum):
    """Enum voor rapport formaten."""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"

class ChartType(str, Enum):
    """Enum voor verschillende chart types."""
    CANDLESTICK = "candlestick"
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    EQUITY = "equity"
    PNL = "pnl"
    WIN_LOSS = "win_loss"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"

class IndicatorType(str, Enum):
    """Enum voor verschillende indicator types."""
    RSI = "RSI"
    MACD = "MACD"
    BB = "Bollinger Bands"
    EMA = "EMA"
    VWAP = "VWAP"
    ATR = "ATR"

class AlertType(str, Enum):
    """Enum voor verschillende alert types."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    REPORT = "REPORT"
    TRADE = "TRADE"
    SYSTEM = "SYSTEM"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"
    RISK = "RISK"
    MARKET = "MARKET" 