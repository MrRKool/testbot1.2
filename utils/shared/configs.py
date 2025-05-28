from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from pathlib import Path
from .enums import TimeFrame, ReportFormat, ChartType

@dataclass
class BaseConfig:
    """Basis configuratie class."""
    # Basis instellingen
    base_dir: str = "."
    log_dir: str = "logs"
    report_dir: str = "reports"
    chart_dir: str = "charts"
    backup_dir: str = "backups"
    
    # Grafiek instellingen
    chart_style: str = "seaborn"
    chart_dpi: int = 300
    chart_size: Tuple[int, int] = (10, 6)
    chart_colors: List[str] = field(default_factory=lambda: [
        "#2ecc71",  # groen
        "#e74c3c",  # rood
        "#3498db",  # blauw
        "#f1c40f",  # geel
        "#9b59b6"   # paars
    ])
    
    # Monitoring instellingen
    monitor_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_drawdown": 20.0,
        "min_winrate": 40.0,
        "min_profit_factor": 1.5
    })

@dataclass
class DashboardConfig(BaseConfig):
    """Configuratie voor het dashboard."""
    # Dashboard specifieke instellingen
    title: str = "Trading Dashboard"
    theme: str = "dark"
    refresh_interval: int = 60  # seconden
    default_timeframe: TimeFrame = TimeFrame.HOUR
    
    # Indicator instellingen
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])

@dataclass
class ReportConfig(BaseConfig):
    """Configuratie voor rapporten."""
    # Rapport specifieke instellingen
    trade_log: str = "trade_log.txt"
    equity_file: str = "equity_track.txt"
    pdf_template: str = "templates/report_template.pdf"
    
    # Rapport instellingen
    report_format: ReportFormat = ReportFormat.PDF
    timezone: str = "UTC"
    currency: str = "EUR"
    decimal_places: int = 2
    default_timeframe: TimeFrame = TimeFrame.DAY
    
    # Email instellingen
    email_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    email_subject: str = "Trading Bot Rapport"
    
    # Backup instellingen
    backup_enabled: bool = True
    backup_retention_days: int = 30

@dataclass
class TradingConfig(BaseConfig):
    """Configuratie voor trading."""
    # Trading specifieke instellingen
    default_timeframe: TimeFrame = TimeFrame.MINUTE
    max_open_trades: int = 3
    stake_amount: float = 100.0
    max_stake_amount: float = 1000.0
    
    # Risk management instellingen
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    trailing_stop: bool = True
    trailing_stop_positive: float = 0.5
    trailing_stop_positive_offset: float = 1.0
    
    # Order instellingen
    order_timeout: int = 30  # seconden
    retry_on_timeout: bool = True
    max_retries: int = 3 