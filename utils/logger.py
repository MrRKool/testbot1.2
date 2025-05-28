# --------------------------------------------------------
# Deel 1: Imports en configuratie
# --------------------------------------------------------

import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import json
from dataclasses import dataclass, asdict, field
import traceback
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import requests
from utils.env_loader import check_environment, get_api_keys, get_telegram_config

# Eigen modules
# from utils.price_fetcher import PriceFetcher

@dataclass
class LogConfig:
    """Configuratie voor de logger."""
    # Algemene instellingen
    log_dir: str = "logs"
    log_level: int = logging.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging instellingen
    file_logging: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    rotate_when: str = "midnight"
    rotate_interval: int = 1
    
    # Console logging instellingen
    console_logging: bool = True
    console_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    
    # JSON logging instellingen
    json_logging: bool = False
    json_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance logging instellingen
    performance_logging: bool = False
    performance_threshold: float = 1.0  # seconden

def setup_directories(directories: List[str] = None) -> None:
    """Configureer benodigde directories."""
    try:
        # Standaard directories
        default_dirs = ["logs", "data"]
        
        # Voeg extra directories toe als opgegeven
        if directories:
            default_dirs.extend(directories)
            
        # Maak directories
        for directory in default_dirs:
            os.makedirs(directory, exist_ok=True)
            
        logging.info("Directories geconfigureerd")
        
    except Exception as e:
        logging.error(f"Fout bij configureren directories: {e}")
        raise

def setup_logging(name: str, config: LogConfig) -> 'Logger':
    """Configureer logging voor de applicatie."""
    return Logger(name, config)

# --------------------------------------------------------
# Configuratie: constante instellingen
# --------------------------------------------------------

# Logging de configuraties
logging.info("✅ Start van configuratie en constante instellingen")

MAX_RETRIES = 3                   # Aantal herhalingen bij fouten
logging.info(f"Max retries ingesteld op: {MAX_RETRIES}")

RETRY_DELAY = 5                   # Seconden wachttijd tussen pogingen
logging.info(f"Wachttijd tussen retries ingesteld op: {RETRY_DELAY} seconden")

MAX_RETRY_DELAY = 20              # Maximale wachttijd bij retries
logging.info(f"Maximale wachttijd voor retries ingesteld op: {MAX_RETRY_DELAY} seconden")

MAX_TRADE_DURATION = 60 * 60      # Maximaal 1 uur per trade
logging.info(f"Maximale handelsduur ingesteld op: {MAX_TRADE_DURATION / 60} minuten")

MAX_ITERATIONS = 100              # Aantal loops binnen een trade
logging.info(f"Aantal iteraties binnen een trade ingesteld op: {MAX_ITERATIONS}")

# Zorg dat logs-map bestaat
setup_directories()
logging.info("📂 Logs-map gecontroleerd/creatie succesvol")

# Logging configuratie
logging.basicConfig(
    filename="logs/equity_track.txt",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("📘 Logger gestart")

#-------------------------------------------------------
# Deel 2: log_trade functie — logging naar logs/
# --------------------------------------------------------

import os
from datetime import datetime
import logging

def log_trade(symbol, signal, price, pnl, equity, duration=None):
    """
    Logt een uitgevoerde trade naar zowel trade_log als equity_track bestand.
    
    Parameters:
    - symbol: (str) handelsinstrument, bijv. 'BTCUSDT'
    - signal: (str) "BUY" of "SELL"
    - price: (float) Entry of exit prijs
    - pnl: (float) Profit or Loss van de trade
    - equity: (float) Huidige balans na de trade
    - duration: (float, optioneel) Duur van de trade in minuten
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)

    # Trade logging
    trade_entry = f"{timestamp} | {symbol} | {signal} | Price: {price:.2f} | PNL: {pnl:.2f} | Equity: {equity:.2f}"
    if duration is not None:
        trade_entry += f" | Duration: {duration:.1f}m"

    try:
        with open(os.path.join(log_folder, "trade_log.txt"), "a") as f:
            f.write(trade_entry + "\n")
        logging.info(f"Trade gelogd: {trade_entry}")
    except Exception as e:
        logging.error(f"❌ Fout bij schrijven naar trade_log.txt: {e}")

    # Equity tracking
    try:
        with open(os.path.join(log_folder, "equity_track.txt"), "a") as f:
            f.write(f"{timestamp},{equity:.2f}\n")
        logging.info(f"Equity gelogd voor {symbol}: {equity:.2f}")
    except Exception as e:
        logging.error(f"❌ Fout bij schrijven naar equity_track.txt: {e}")
        
# --------------------------------------------------------
# Deel 3: Indicator Functies
# --------------------------------------------------------

def compute_rsi(df, period=14):
    """
    Bereken de Relative Strength Index (RSI).
    
    Parameters:
    - df: Pandas DataFrame met 'close' kolom
    - period: Periode voor berekening (standaard 14)

    Returns:
    - Pandas Series met RSI-waarden
    """
    logging.info(f"📊 Berekening RSI voor {period} periodes...")
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # voorkomen van deling door 0
    rsi = 100 - (100 / (1 + rs))
    logging.info(f"✅ RSI berekend: {rsi.iloc[-1]:.2f}")
    return rsi


def compute_macd(df, fast=12, slow=26, signal=9):
    """
    Bereken de Moving Average Convergence Divergence (MACD).

    Parameters:
    - df: Pandas DataFrame met 'close' kolom
    - fast: Snelle EMA periode
    - slow: Trage EMA periode
    - signal: Signaallijn periode

    Returns:
    - MACD lijn en signaallijn als Pandas Series
    """
    logging.info(f"📊 Berekening MACD met fast={fast}, slow={slow}, signal={signal}...")
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    logging.info(f"✅ MACD berekend: {macd.iloc[-1]:.2f}, Signal line: {signal_line.iloc[-1]:.2f}")
    return macd, signal_line


def compute_bollinger_bands(df, window=20, num_std=2):
    """
    Bereken Bollinger Bands.

    Parameters:
    - df: Pandas DataFrame met 'close' kolom
    - window: Aantal candles voor SMA
    - num_std: Aantal standaarddeviaties

    Returns:
    - Upper band, lower band als Pandas Series
    """
    logging.info(f"📊 Berekening Bollinger Bands voor window={window}, num_std={num_std}...")
    sma = df['close'].rolling(window=window, min_periods=1).mean()
    std = df['close'].rolling(window=window, min_periods=1).std()

    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    logging.info(f"✅ Bollinger Bands berekend: Upper={upper_band.iloc[-1]:.2f}, Lower={lower_band.iloc[-1]:.2f}")
    return upper_band, lower_band


def compute_ema(df, span):
    """
    Bereken de Exponential Moving Average (EMA).
    
    Parameters:
    - df: Pandas DataFrame met 'close' kolom
    - span: Periode voor EMA

    Returns:
    - Pandas Series met EMA-waarden
    """
    logging.info(f"📊 Berekening EMA voor span={span}...")
    ema = df['close'].ewm(span=span, adjust=False).mean()
    logging.info(f"✅ EMA berekend: {ema.iloc[-1]:.2f}")
    return ema

# --------------------------------------------------------
# Deel 4: Candles ophalen via echte Bybit API
# --------------------------------------------------------

def fetch_candles(symbol, interval="15", limit=200, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """
    Haalt echte candlestick-data op via Bybit API.

    Parameters:
    - symbol: Het handelsinstrument (bijv. 'BTCUSDT')
    - interval: Tijdframe (bijv. '1', '5', '15', '60', '240', 'D')
    - limit: Aantal candles
    - retries: Aantal pogingen bij fouten
    - delay: Wachtduur tussen pogingen

    Returns:
    - Pandas DataFrame met candles of lege DataFrame bij fout
    """
    logging.info(f"📊 Ophalen candles voor {symbol} op interval {interval}...")
    url = "https://api.bybit.com/v5/market/kline"

    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, params={
                "category": "linear",
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }, timeout=10)

            response.raise_for_status()
            raw = response.json().get("result", {}).get("list", [])

            if not raw:
                logging.warning(f"⚠️ Geen candledata voor {symbol} op {interval}m")
                return pd.DataFrame()

            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
            df[["open", "high", "low", "close", "volume", "turnover"]] = df[["open", "high", "low", "close", "volume", "turnover"]].astype(float)
            df.set_index("timestamp", inplace=True)

            logging.info(f"✅ Candles succesvol opgehaald voor {symbol} op {interval}m")
            return df

        except Exception as e:
            logging.error(f"❌ Fout bij ophalen candles ({symbol}): {e}")
            time.sleep(delay * (attempt + 1))  # Exponentiële backoff
            attempt += 1

    logging.error(f"❌ Mislukt om candles op te halen voor {symbol} na {retries} pogingen.")
    return pd.DataFrame()

# --------------------------------------------------------
# Deel 5: Functie om de signalen te genereren
# --------------------------------------------------------

def generate_signal(df_1m, df_15m, df_1h):
    """
    Genereer een handelssignaal op basis van RSI, MACD, Bollinger Bands en EMA-crossovers.
    Retourneert ook trailing start en gap (voor trailing stop-loss).
    """
    if df_1m.empty or df_15m.empty or df_1h.empty:
        logging.error("❌ Eén of meerdere dataframes zijn leeg – geen signaal gegenereerd")
        raise ValueError("❌ Eén of meerdere dataframes zijn leeg – geen signaal gegenereerd")

    signals = []
    logging.info("📊 Start signaalgeneratie...")

    # RSI 1m
    rsi_1m = compute_rsi(df_1m)
    if rsi_1m.iloc[-1] < 30:
        signals.append("BUY")
        logging.info(f"🔍 RSI signaal: BUY ({rsi_1m.iloc[-1]:.2f})")
    elif rsi_1m.iloc[-1] > 70:
        signals.append("SELL")
        logging.info(f"🔍 RSI signaal: SELL ({rsi_1m.iloc[-1]:.2f})")

    # MACD 1h
    macd_1h, signal_line_1h = compute_macd(df_1h)
    if macd_1h.iloc[-1] > signal_line_1h.iloc[-1]:
        signals.append("BUY")
        logging.info(f"🔍 MACD signaal: BUY ({macd_1h.iloc[-1]:.4f} > {signal_line_1h.iloc[-1]:.4f})")
    elif macd_1h.iloc[-1] < signal_line_1h.iloc[-1]:
        signals.append("SELL")
        logging.info(f"🔍 MACD signaal: SELL ({macd_1h.iloc[-1]:.4f} < {signal_line_1h.iloc[-1]:.4f})")

    # Bollinger Bands 15m
    upper_band, lower_band = compute_bollinger_bands(df_15m)
    price_15m = df_15m['close'].iloc[-1]
    if price_15m < lower_band.iloc[-1]:
        signals.append("BUY")
        logging.info(f"🔍 Bollinger signaal: BUY (prijs {price_15m:.2f} < lower {lower_band.iloc[-1]:.2f})")
    elif price_15m > upper_band.iloc[-1]:
        signals.append("SELL")
        logging.info(f"🔍 Bollinger signaal: SELL (prijs {price_15m:.2f} > upper {upper_band.iloc[-1]:.2f})")

    # EMA 50 vs 200 cross op 1h
    ema50 = compute_ema(df_1h, 50)
    ema200 = compute_ema(df_1h, 200)
    if ema50.iloc[-1] > ema200.iloc[-1]:
        signals.append("BUY")
        logging.info(f"🔍 EMA signaal: BUY (EMA50 {ema50.iloc[-1]:.2f} > EMA200 {ema200.iloc[-1]:.2f})")
    elif ema50.iloc[-1] < ema200.iloc[-1]:
        signals.append("SELL")
        logging.info(f"🔍 EMA signaal: SELL (EMA50 {ema50.iloc[-1]:.2f} < EMA200 {ema200.iloc[-1]:.2f})")

    # Beslissing maken
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")

    if buy_count > sell_count:
        logging.info(f"✅ Signaalbeslissing: BUY ({buy_count} vs {sell_count})")
        return "BUY", 1.5, 0.5
    elif sell_count > buy_count:
        logging.info(f"✅ Signaalbeslissing: SELL ({sell_count} vs {buy_count})")
        return "SELL", 1.5, 0.5
    else:
        logging.info(f"⏸️ Geen duidelijk signaal – HOLD ({buy_count} BUY vs {sell_count} SELL)")
        return "HOLD", 0, 0

# --------------------------------------------------------
# Deel 6: Logging configuratie
# --------------------------------------------------------

import os
import logging

# Zorg dat de logmap bestaat
os.makedirs("logs", exist_ok=True)

# Configureer logging
logging.basicConfig(
    filename="logs/trade_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Startmelding
logging.info("✅ Bot gestart...")

# --------------------------------------------------------
# Deel 7 & 8: 3-uurs rapportfunctie + Telegram + statistieken
# --------------------------------------------------------

def calculate_trade_statistics(trades):
    """
    Genereer statistieken over de trades van de laatste 3 uur.
    Returns een dict met: aantal trades, winst/verlies, gemiddeld PNL, winrate, totaal PNL.
    """
    stats = {
        "Trades (laatste 3u)": len(trades),
        "Winsttrades": 0,
        "Verliestrades": 0,
        "Gemiddelde PNL": 0,
        "Totaal PNL": 0,
        "Winrate (%)": 0
    }

    winst_trades = [t for t in trades if "WIN" in t[3]]
    verlies_trades = [t for t in trades if "LOSS" in t[3]]
    stats["Winsttrades"] = len(winst_trades)
    stats["Verliestrades"] = len(verlies_trades)

    try:
        pnls = []
        for t in trades:
            for part in t:
                if "PNL:" in part:
                    value = float(part.replace("PNL:", "").strip())
                    pnls.append(value)
                    break
        if pnls:
            stats["Gemiddelde PNL"] = round(np.mean(pnls), 2)
            stats["Totaal PNL"] = round(sum(pnls), 2)
    except Exception as e:
        stats["Gemiddelde PNL"] = "Fout in PNL data"
        stats["Totaal PNL"] = "Fout in PNL data"
        logging.error(f"❌ Fout bij PNL-berekening: {e}")

    if len(trades) > 0:
        stats["Winrate (%)"] = round(stats["Winsttrades"] / len(trades) * 100, 2)

    return stats


def print_trade_statistics(stats):
    """Print de gegenereerde statistieken overzichtelijk."""
    print("\n📊 3-Uurs Statistieken:")
    for stat, value in stats.items():
        print(f"• {stat}: {value}")


def generate_3hourly_report():
    """Genereer een 3-uurs rapport op basis van de handelslog inclusief statistieken en Telegram-versturing."""
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    TRADE_LOG = os.path.join(BASE_DIR, 'logs', 'trade_log.txt')

    if not os.path.exists(TRADE_LOG):
        print(f"⚠️ Geen trade_log.txt gevonden op {TRADE_LOG}")
        return

    now = datetime.now()
    start_time = now - timedelta(hours=3)

    trades = []
    with open(TRADE_LOG, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = [p.strip() for p in line.strip().split("|")]
            if len(parts) < 6:
                continue
            try:
                ts = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                if ts >= start_time:
                    trades.append(parts)
            except ValueError:
                continue

    if not trades:
        print("⚠️ Geen trades in de laatste 3 uur.")
        return

    # Bereken statistieken
    stats = calculate_trade_statistics(trades)
    print_trade_statistics(stats)

    # Bouw Telegram-rapport
    report_lines = [
        f"📊 3-uurs rapport — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Totaal trades: {stats['Trades (laatste 3u)']}",
        f"🏆 Winsten: {stats['Winsttrades']}",
        f"❌ Verliezen: {stats['Verliestrades']}",
        f"📈 Winrate: {stats['Winrate (%)']}%",
        f"💰 Gemiddelde PNL: {stats['Gemiddelde PNL']}",
        f"💼 Totale PNL: {stats['Totaal PNL']}"
    ]
    report = "\n".join(report_lines)

    try:
        send_telegram_message(report)
    except Exception as e:
        print(f"❌ Fout bij versturen via Telegram: {e}")

# --------------------------------------------------------
# Deel 2: Custom Log Handlers
# --------------------------------------------------------

class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Aangepaste file handler met verbeterde error handling."""
    def __init__(self, filename: str, max_bytes: int = 0, backup_count: int = 0, encoding: str = 'utf-8'):
        try:
            # Zorg ervoor dat de directory bestaat
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            super().__init__(filename, maxBytes=max_bytes, backupCount=backup_count, encoding=encoding)
        except Exception as e:
            print(f"Kritieke fout bij initialiseren file handler: {e}")
            raise

# --------------------------------------------------------
# Deel 3: Logger Class
# --------------------------------------------------------

class Logger:
    """Class voor het beheren van logging."""
    
    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Implementeer singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, name: str, config: LogConfig):
        """Initialiseer de logger."""
        if not self._initialized:
            self.name = name
            self.config = config
            self._setup_directories()
            self._setup_root_logger()
            self._initialized = True
            
    def _setup_directories(self):
        """Configureer benodigde directories."""
        try:
            os.makedirs(self.config.log_dir, exist_ok=True)
            os.makedirs(os.path.join(self.config.log_dir, "json"), exist_ok=True)
            os.makedirs(os.path.join(self.config.log_dir, "performance"), exist_ok=True)
        except Exception as e:
            print(f"Fout bij configureren log directories: {e}")
            sys.exit(1)
            
    def _setup_root_logger(self):
        """Configureer de root logger."""
        try:
            # Verwijder bestaande handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
            # Configureer root logger
            root_logger.setLevel(self.config.log_level)
            
            # File handler
            if self.config.file_logging:
                file_handler = RotatingFileHandler(
                    os.path.join(self.config.log_dir, "trading_bot.log"),
                    max_bytes=self.config.max_file_size,
                    backup_count=self.config.backup_count
                )
                file_handler.setFormatter(logging.Formatter(
                    self.config.log_format,
                    datefmt=self.config.date_format
                ))
                root_logger.addHandler(file_handler)
                
            # Console handler
            if self.config.console_logging:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    self.config.console_format,
                    datefmt=self.config.date_format
                ))
                root_logger.addHandler(console_handler)
                
        except Exception as e:
            print(f"Fout bij configureren root logger: {e}")
            sys.exit(1)
            
    def get_logger(self) -> logging.Logger:
        """Haal een logger op voor de huidige module."""
        return self._loggers[self.name]
        
    def set_level(self, level: int):
        """Verander het logging level voor de huidige logger."""
        self.config.log_level = level
        self.get_logger().setLevel(level)
        logging.getLogger().setLevel(level)
        
    def add_file_handler(self, filename: str):
        """Voeg een file handler toe voor de huidige logger."""
        if filename not in self._loggers:
            file_handler = RotatingFileHandler(
                os.path.join(self.config.log_dir, filename),
                max_bytes=self.config.max_file_size,
                backup_count=self.config.backup_count
            )
            file_handler.setFormatter(logging.Formatter(
                self.config.log_format,
                datefmt=self.config.date_format
            ))
            self._loggers[filename] = logging.getLogger(filename)
            self._loggers[filename].addHandler(file_handler)
            
    def remove_file_handler(self, filename: str):
        """Verwijder een file handler van de huidige logger."""
        if filename in self._loggers:
            for handler in self._loggers[filename].handlers[:]:
                if isinstance(handler, RotatingFileHandler) and handler.baseFilename.endswith(filename):
                    self._loggers[filename].removeHandler(handler)

def main():
    """Test de logger."""
    try:
        # Maak logger instance
        logger = Logger("test", LogConfig())
        
        # Test verschillende log levels
        test_logger = logger.get_logger()
        test_logger.debug("Test debug message")
        test_logger.info("Test info message")
        test_logger.warning("Test warning message")
        test_logger.error("Test error message")
        
        # Test log files
        log_files = logger.get_log_files()
        print(f"Log bestanden: {log_files}")
        
        print("Logger test succesvol uitgevoerd")
        
    except Exception as e:
        print(f"Kritieke fout in logger test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
