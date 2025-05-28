# --------------------------------------------------------
# Deel 1: Imports
# --------------------------------------------------------
import time
from datetime import datetime, date, timedelta
import os
import sys
import pandas as pd
import streamlit as st
import numpy as np
from fpdf import FPDF
import logging
import requests
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
from threading import Thread
from dataclasses import dataclass, field
from enum import Enum
from utils.shared.enums import TimeFrame, SignalType, IndicatorType, AlertType
from utils.shared.configs import DashboardConfig

# --------------------------------------------------------
# Deel 2: Zorg dat de bovenliggende map beschikbaar is voor imports
# --------------------------------------------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.telegram_alerts import send_telegram_message, TelegramAlert
from utils.env_loader import check_environment, get_api_keys, get_telegram_config
from utils.logger import Logger

# --------------------------------------------------------
# Deel 3: Functie om statistieken te berekenen
# --------------------------------------------------------

def calculate_stats(df):
    """Bereken statistieken over de trades."""
    total = len(df)
    wins = len(df[df["Result"] == "WIN"])
    losses = len(df[df["Result"] == "LOSS"])
    winrate = (wins / total) * 100 if total > 0 else 0
    avg_pnl = df["PNL"].mean() if total > 0 else 0
    best_trade = df["PNL"].max() if total > 0 else 0
    worst_trade = df["PNL"].min() if total > 0 else 0
    daily_pnl = df[df["Timestamp"].dt.date == date.today()]["PNL"].sum()

    # Voeg extra logica toe voor een extra niveau van inzicht in prestaties
    logging.info(f"Winrate: {winrate}% | Gemiddelde PNL: {avg_pnl:.2f} | Beste trade: {best_trade:.2f} | Slechtste trade: {worst_trade:.2f} | Dagelijkse PNL: {daily_pnl:.2f}")
    
    return winrate, avg_pnl, best_trade, worst_trade, daily_pnl

# --------------------------------------------------------
# Deel 4: Paddefinities
# --------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TRADE_LOG_PATH = os.path.join(LOG_DIR, "trade_log.txt")
EQUITY_LOG_PATH = os.path.join(LOG_DIR, "equity_track.txt")

# --------------------------------------------------------
# Deel 5: Streamlit Dashboard Setup
# --------------------------------------------------------

def launch_dashboard():
    """Start het trading dashboard en toon de statistieken en grafieken."""
    st.title("Trading Dashboard")
    st.sidebar.title("Navigatie")

    # Laad trade-log en bereken statistieken
    try:
        trade_log_df = pd.read_csv(TRADE_LOG_PATH)
        trade_log_df["Timestamp"] = pd.to_datetime(trade_log_df["Timestamp"])
        winrate, avg_pnl, best_trade, worst_trade, daily_pnl = calculate_stats(trade_log_df)

        # Toon de statistieken
        st.write(f"**Winrate:** {winrate:.2f}%")
        st.write(f"**Gemiddelde PNL:** €{avg_pnl:.2f}")
        st.write(f"**Beste trade:** €{best_trade:.2f}")
        st.write(f"**Slechtste trade:** €{worst_trade:.2f}")
        st.write(f"**Dagelijkse PNL:** €{daily_pnl:.2f}")
        
    except Exception as e:
        st.error(f"Fout bij het laden van trade-log: {e}")
        logging.error(f"Fout bij het laden van trade-log: {e}")
    
    # Voeg meer visualisatie toe, zoals grafieken van win/verlies, PNL-curves, etc.
    # bijvoorbeeld met matplotlib voor een visuele representatie
    st.line_chart(trade_log_df['PNL'])

# --------------------------------------------------------
# --------------------------------------------------------
# Deel 5: Functies voor het laden van gegevens
# --------------------------------------------------------

def load_equity():
    """
    Laadt de equity-waarde uit het bestand.
    Retourneert de equity als een float of een standaardwaarde als het bestand niet gevonden wordt.
    """
    try:
        with open(EQUITY_LOG_PATH, "r") as f:
            equity = float(f.read().strip())
            logging.info(f"✅ Equity succesvol geladen: {equity:.2f}")
            return equity
    except FileNotFoundError:
        st.warning(f"⚠️ Bestand niet gevonden: {EQUITY_LOG_PATH} — standaard equity gebruikt (10.000).")
        logging.warning(f"⚠️ Equity logbestand niet gevonden, gebruik standaardwaarde van 10.000")
        return 10000.0
    except Exception as e:
        st.error(f"❌ Fout bij laden equity: {e}")
        logging.error(f"❌ Fout bij laden equity: {e}")
        return 10000.0

def load_trades():
    """
    Laadt de trades uit het trade logbestand.
    Retourneert een DataFrame met de tradegegevens of een lege DataFrame als er een fout optreedt.
    """
    if not os.path.exists(TRADE_LOG_PATH):
        st.warning(f"⚠️ Trade log niet gevonden: {TRADE_LOG_PATH}")
        logging.warning(f"⚠️ Geen trade logbestand gevonden op {TRADE_LOG_PATH}")
        return pd.DataFrame(columns=["Timestamp", "Symbol", "Direction", "Result", "PNL", "Equity"])
    
    try:
        data = []
        with open(TRADE_LOG_PATH, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("|")]
                if len(parts) < 6:
                    continue
                try:
                    timestamp, symbol, direction, result, pnl_text, equity_text = parts[:6]
                    pnl_val = float(pnl_text.replace("PNL:", "").strip())
                    equity_val = float(equity_text.replace("EQUITY:", "").strip())
                    data.append([timestamp, symbol, direction, result, pnl_val, equity_val])
                except Exception as e:
                    st.error(f"❌ Parse-fout in regel: {line}\n{e}")
                    logging.error(f"❌ Parse-fout in regel: {line}\n{e}")
        
        df = pd.DataFrame(data, columns=["Timestamp", "Symbol", "Direction", "Result", "PNL", "Equity"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df.dropna(subset=["Timestamp"], inplace=True)
        logging.info(f"✅ {len(df)} trades geladen uit {TRADE_LOG_PATH}")
        return df
    except Exception as e:
        st.error(f"❌ Kon het logbestand niet lezen: {e}")
        logging.error(f"❌ Fout bij lezen van trade logbestand: {e}")
        return pd.DataFrame(columns=["Timestamp", "Symbol", "Direction", "Result", "PNL", "Equity"])

# --------------------------------------------------------
# Deel 6: Bereken en toon de verschillende indicatoren
# --------------------------------------------------------

def compute_rsi(df, period=14):
    """
    Bereken de Relative Strength Index (RSI) op basis van de slotkoersen in de dataframe.
    :param df: DataFrame met 'close' kolom
    :param period: Periode voor het berekenen van de RSI (standaard 14)
    :return: RSI-waarde als een pandas Series
    """
    try:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        logging.debug(f"RSI berekend met periode {period} voor {len(df)} candles.")
        return rsi
    except Exception as e:
        logging.error(f"❌ Fout bij berekening van RSI: {e}")
        return pd.Series([np.nan] * len(df), index=df.index)

def compute_macd(df, fast=12, slow=26, signal=9):
    """
    Bereken de Moving Average Convergence Divergence (MACD) en signal line.
    :param df: DataFrame met 'close' kolom
    :param fast: Periode voor de snelle EMA (standaard 12)
    :param slow: Periode voor de langzame EMA (standaard 26)
    :param signal: Periode voor de signal line (standaard 9)
    :return: macd, signal_line als pandas Series
    """
    try:
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        logging.debug(f"MACD en signal line berekend met fast={fast}, slow={slow}, signal={signal}.")
        return macd, signal_line
    except Exception as e:
        logging.error(f"❌ Fout bij berekening van MACD: {e}")
        return pd.Series([np.nan] * len(df), index=df.index), pd.Series([np.nan] * len(df), index=df.index)

def compute_bollinger_bands(df, window=20, num_std=2):
    """
    Bereken de Bollinger Bands op basis van de slotkoersen in de dataframe.
    :param df: DataFrame met 'close' kolom
    :param window: Periode voor de moving average (standaard 20)
    :param num_std: Aantal standaarddeviaties voor de banden (standaard 2)
    :return: upper_band, lower_band als pandas Series
    """
    try:
        sma = df['close'].rolling(window=window, min_periods=1).mean()
        std = df['close'].rolling(window=window, min_periods=1).std()
        
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        
        logging.debug(f"Bollinger Bands berekend met window={window} en num_std={num_std}.")
        return upper_band, lower_band
    except Exception as e:
        logging.error(f"❌ Fout bij berekening van Bollinger Bands: {e}")
        return pd.Series([np.nan] * len(df), index=df.index), pd.Series([np.nan] * len(df), index=df.index)

def compute_ema(df, span):
    """
    Bereken de Exponential Moving Average (EMA) voor de gegeven periode.
    :param df: DataFrame met 'close' kolom
    :param span: Periode voor de EMA
    :return: EMA als pandas Series
    """
    try:
        ema = df['close'].ewm(span=span, adjust=False).mean()
        
        logging.debug(f"EMA berekend met periode {span} voor {len(df)} candles.")
        return ema
    except Exception as e:
        logging.error(f"❌ Fout bij berekening van EMA: {e}")
        return pd.Series([np.nan] * len(df), index=df.index)

# --------------------------------------------------------
# Deel 7: Genereren van een handelssignaal
# --------------------------------------------------------

def generate_signal(df_1m, df_15m, df_1h):
    """
    Genereer een handelssignaal op basis van de indicatoren van candles.
    Gebruikt RSI, MACD, Bollinger Bands en EMA.
    """

    # Berekeningen voor RSI, MACD, Bollinger Bands en EMA
    rsi_1m = compute_rsi(df_1m)
    macd_1h, signal_line_1h = compute_macd(df_1h)
    upper_band_15m, lower_band_15m = compute_bollinger_bands(df_15m)
    ema50_1h = compute_ema(df_1h, 50)
    ema200_1h = compute_ema(df_1h, 200)

    # Initialiseer signalen
    buy_signal = False
    sell_signal = False

    # RSI Signaal (Oververkocht of Overgekocht)
    if rsi_1m.iloc[-1] < 30:
        buy_signal = True
    elif rsi_1m.iloc[-1] > 70:
        sell_signal = True

    # MACD Signaal (Bullish of Bearish Cross)
    if macd_1h.iloc[-1] > signal_line_1h.iloc[-1]:
        buy_signal = True
    elif macd_1h.iloc[-1] < signal_line_1h.iloc[-1]:
        sell_signal = True

    # Bollinger Bands Signaal (Prijs breekt onder of boven de banden)
    if df_15m['close'].iloc[-1] < lower_band_15m.iloc[-1]:
        buy_signal = True
    elif df_15m['close'].iloc[-1] > upper_band_15m.iloc[-1]:
        sell_signal = True

    # EMA Crossover Signaal (50-EMA boven of onder 200-EMA)
    if ema50_1h.iloc[-1] > ema200_1h.iloc[-1]:
        buy_signal = True
    elif ema50_1h.iloc[-1] < ema200_1h.iloc[-1]:
        sell_signal = True

    # Beslissing op basis van de signalen
    if buy_signal and not sell_signal:
        return "BUY", 1.5, 0.5  # Return BUY signal, trailing start, and gap
    elif sell_signal and not buy_signal:
        return "SELL", 1.5, 0.5  # Return SELL signal, trailing start, and gap
    else:
        return "HOLD", 0, 0  # Geen beslissing, houd positie vast

# --------------------------------------------------------
# Deel 8: Weergave van indicatoren in dashboard
# --------------------------------------------------------

def display_indicator_metrics(df_1m, df_15m, df_1h):
    """
    Toont de indicatoren RSI, MACD, Bollinger Bands en EMA op het dashboard.
    Groepeer de indicatoren voor een beter overzicht.
    """

    # Bereken de indicatoren
    try:
        rsi_1m = compute_rsi(df_1m)
        macd_1h, signal_line_1h = compute_macd(df_1h)
        upper_band_15m, lower_band_15m = compute_bollinger_bands(df_15m)
        ema50_1h = compute_ema(df_1h, 50)
        ema200_1h = compute_ema(df_1h, 200)
    except Exception as e:
        st.error(f"❌ Fout bij het berekenen van indicatoren: {e}")
        return

    # Display the metrics using Streamlit columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Weergave van RSI en MACD
        st.subheader("RSI & MACD")
        st.write(f"RSI (1m): {rsi_1m.iloc[-1] if not rsi_1m.isna().any() else 'N/A'}")
        st.write(f"MACD (1h): {macd_1h.iloc[-1] if not macd_1h.isna().any() else 'N/A'}")

    with col2:
        # Weergave van Bollinger Bands en EMA
        st.subheader("Bollinger Bands & EMA")
        st.write(f"Bollinger Band Lower (15m): {lower_band_15m.iloc[-1] if not lower_band_15m.isna().any() else 'N/A'}")
        st.write(f"Bollinger Band Upper (15m): {upper_band_15m.iloc[-1] if not upper_band_15m.isna().any() else 'N/A'}")
        st.write(f"EMA 50 (1h): {ema50_1h.iloc[-1] if not ema50_1h.isna().any() else 'N/A'}")
        st.write(f"EMA 200 (1h): {ema200_1h.iloc[-1] if not ema200_1h.isna().any() else 'N/A'}")

# --------------------------------------------------------
# Deel 9: Constants en Enums
# --------------------------------------------------------

@dataclass
class DashboardConfig:
    """Configuratie voor het dashboard."""
    # Basis instellingen
    title: str = "Trading Dashboard"
    theme: str = "dark"
    refresh_interval: int = 60  # seconden
    default_timeframe: TimeFrame = TimeFrame.HOUR
    
    # Pad instellingen
    base_dir: str = "."
    log_dir: str = "logs"
    trade_log: str = "trade_log.txt"
    equity_log: str = "equity_track.txt"
    
    # Grafiek instellingen
    chart_height: int = 600
    chart_width: int = 800
    chart_colors: List[str] = field(default_factory=lambda: [
        "#2ecc71",  # groen
        "#e74c3c",  # rood
        "#3498db",  # blauw
        "#f1c40f",  # geel
        "#9b59b6"   # paars
    ])
    
    # Indicator instellingen
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: int = 2
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50, 200])
    
    # Monitoring instellingen
    monitor_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_drawdown": 20.0,
        "min_winrate": 40.0,
        "min_profit_factor": 1.5
    })

@dataclass
class Signal:
    """Class voor trading signalen."""
    type: SignalType
    symbol: str
    timeframe: TimeFrame
    price: float
    timestamp: datetime
    confidence: float = 0.0
    indicators: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

# --------------------------------------------------------
# Deel 10: Path Management
# --------------------------------------------------------

class PathManager:
    """Class voor het beheren van bestandspaden."""
    
    def __init__(self, config: DashboardConfig):
        """Initialiseer de path manager."""
        self.config = config
        self.logger = Logger()
        self.log = self.logger.get_logger(__name__)
        
        # Definieer paden
        self.base_dir = Path(config.base_dir)
        self.log_dir = self.base_dir / config.log_dir
        self.trade_log = self.log_dir / config.trade_log
        self.equity_log = self.log_dir / config.equity_log
        
        # Zorg dat directories bestaan
        self.ensure_directories()
        
    def ensure_directories(self) -> None:
        """Zorg dat alle benodigde directories bestaan."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log.info(f"Directories gecontroleerd en aangemaakt indien nodig")
            
        except Exception as e:
            self.log.error(f"Fout bij aanmaken directories: {e}")
            raise

# --------------------------------------------------------
# Deel 11: Data Management
# --------------------------------------------------------

class DataManager:
    """Class voor het beheren van data."""
    
    def __init__(self, path_manager: PathManager):
        """Initialiseer de data manager."""
        self.path_manager = path_manager
        self.logger = Logger()
        self.log = self.logger.get_logger(__name__)
        
    def load_trades(self) -> pd.DataFrame:
        """Laad trades uit het logbestand."""
        try:
            if not self.path_manager.trade_log.exists():
                self.log.warning("Trade log niet gevonden")
                return pd.DataFrame(columns=[
                    "Timestamp", "Symbol", "Direction", "Result",
                    "PNL", "Equity", "Size", "Price"
                ])
            
            # Lees trades
            data = []
            with open(self.path_manager.trade_log, "r") as f:
                for line in f:
                    try:
                        parts = [p.strip() for p in line.strip().split("|")]
                        if len(parts) < 8:
                            continue
                            
                        timestamp, symbol, direction, result, pnl, equity, size, price = parts[:8]
                        data.append([
                            pd.to_datetime(timestamp),
                            symbol,
                            direction,
                            result,
                            float(pnl.replace("PNL:", "").strip()),
                            float(equity.replace("EQUITY:", "").strip()),
                            float(size.replace("SIZE:", "").strip()),
                            float(price.replace("PRICE:", "").strip())
                        ])
                    except Exception as e:
                        self.log.error(f"Fout bij parsen trade: {line}\n{e}")
                        continue
            
            # Maak DataFrame
            df = pd.DataFrame(
                data,
                columns=["Timestamp", "Symbol", "Direction", "Result",
                        "PNL", "Equity", "Size", "Price"]
            )
            
            self.log.info(f"{len(df)} trades geladen")
            return df
            
        except Exception as e:
            self.log.error(f"Fout bij laden trades: {e}")
            return pd.DataFrame()
            
    def load_equity(self) -> float:
        """Laad equity uit het logbestand."""
        try:
            if not self.path_manager.equity_log.exists():
                self.log.warning("Equity log niet gevonden")
                return 10000.0
                
            with open(self.path_manager.equity_log, "r") as f:
                equity = float(f.read().strip())
                self.log.info(f"Equity geladen: {equity:.2f}")
                return equity
                
        except Exception as e:
            self.log.error(f"Fout bij laden equity: {e}")
            return 10000.0

# --------------------------------------------------------
# Deel 12: Technical Analysis
# --------------------------------------------------------

class TechnicalAnalysis:
    """Class voor technische analyse."""
    
    def __init__(self, config: DashboardConfig):
        """Initialiseer de technical analysis."""
        self.config = config
        self.logger = Logger()
        self.log = self.logger.get_logger(__name__)
        
    @staticmethod
    def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Bereken RSI."""
        try:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logging.error(f"Fout bij berekenen RSI: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)
            
    @staticmethod
    def compute_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Bereken MACD."""
        try:
            ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            
            return macd, signal_line
            
        except Exception as e:
            logging.error(f"Fout bij berekenen MACD: {e}")
            return pd.Series([np.nan] * len(df)), pd.Series([np.nan] * len(df))
            
    @staticmethod
    def compute_bollinger_bands(
        df: pd.DataFrame,
        window: int = 20,
        num_std: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Bereken Bollinger Bands."""
        try:
            sma = df["close"].rolling(window=window, min_periods=1).mean()
            std = df["close"].rolling(window=window, min_periods=1).std()
            
            upper_band = sma + (std * num_std)
            lower_band = sma - (std * num_std)
            
            return upper_band, lower_band
            
        except Exception as e:
            logging.error(f"Fout bij berekenen Bollinger Bands: {e}")
            return pd.Series([np.nan] * len(df)), pd.Series([np.nan] * len(df))
            
    @staticmethod
    def compute_ema(df: pd.DataFrame, span: int) -> pd.Series:
        """Bereken EMA."""
        try:
            return df["close"].ewm(span=span, adjust=False).mean()
            
        except Exception as e:
            logging.error(f"Fout bij berekenen EMA: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)
            
    @staticmethod
    def compute_vwap(df: pd.DataFrame) -> pd.Series:
        """Bereken VWAP."""
        try:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            volume = df["volume"]
            
            vwap = (typical_price * volume).cumsum() / volume.cumsum()
            return vwap
            
        except Exception as e:
            logging.error(f"Fout bij berekenen VWAP: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)
            
    @staticmethod
    def compute_atr(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """Bereken Average True Range."""
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=1).mean()
            
            return atr
            
        except Exception as e:
            logging.error(f"Fout bij berekenen ATR: {e}")
            return pd.Series([np.nan] * len(df), index=df.index)

# --------------------------------------------------------
# Deel 13: Dashboard
# --------------------------------------------------------

class Dashboard:
    """Class voor het trading dashboard."""
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        env_loader: Optional[check_environment] = None,
        telegram_alert: Optional[TelegramAlert] = None
    ):
        """Initialiseer het dashboard."""
        self.config = config or DashboardConfig()
        self.env_loader = env_loader or check_environment()
        self.telegram_alert = telegram_alert
        self.logger = Logger()
        self.log = self.logger.get_logger(__name__)
        
        # Initialiseer managers
        self.path_manager = PathManager(self.config)
        self.data_manager = DataManager(self.path_manager)
        self.technical_analysis = TechnicalAnalysis(self.config)
        
        # Setup Streamlit
        st.set_page_config(
            page_title=self.config.title,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def create_plot(self, df: pd.DataFrame) -> None:
        """Maak een plot met Plotly."""
        try:
            # Maak subplots
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="Price"
                ),
                row=1,
                col=1
            )
            
            # Voeg EMA's toe
            for period in self.config.ema_periods:
                ema = self.technical_analysis.compute_ema(df, period)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ema,
                        name=f"EMA {period}",
                        line=dict(width=1)
                    ),
                    row=1,
                    col=1
                )
            
            # Voeg Bollinger Bands toe
            upper, lower = self.technical_analysis.compute_bollinger_bands(df)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper,
                    name="BB Upper",
                    line=dict(width=1, dash="dash")
                ),
                row=1,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=lower,
                    name="BB Lower",
                    line=dict(width=1, dash="dash")
                ),
                row=1,
                col=1
            )
            
            # Voeg RSI toe
            rsi = self.technical_analysis.compute_rsi(df)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=rsi,
                    name="RSI",
                    line=dict(color=self.config.chart_colors[0])
                ),
                row=2,
                col=1
            )
            
            # Voeg MACD toe
            macd, signal = self.technical_analysis.compute_macd(df)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=macd,
                    name="MACD",
                    line=dict(color=self.config.chart_colors[1])
                ),
                row=3,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=signal,
                    name="Signal",
                    line=dict(color=self.config.chart_colors[2])
                ),
                row=3,
                col=1
            )
            
            # Update layout
            fig.update_layout(
                height=self.config.chart_height,
                width=self.config.chart_width,
                xaxis_rangeslider_visible=False,
                template="plotly_dark" if self.config.theme == "dark" else "plotly_white"
            )
            
            # Toon plot
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.log.error(f"Fout bij maken plot: {e}")
            st.error(f"Fout bij maken plot: {e}")
            
    def update_dashboard(self) -> None:
        """Update het dashboard."""
        try:
            # Laad data
            trades_df = self.data_manager.load_trades()
            equity = self.data_manager.load_equity()
            
            # Toon statistieken
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Equity",
                    f"€{equity:.2f}",
                    f"{trades_df['PNL'].iloc[-1]:.2f}" if not trades_df.empty else "0.00"
                )
                
            with col2:
                winrate = (
                    len(trades_df[trades_df["Result"] == "WIN"]) /
                    len(trades_df) * 100
                ) if not trades_df.empty else 0
                st.metric("Winrate", f"{winrate:.1f}%")
                
            with col3:
                avg_pnl = trades_df["PNL"].mean() if not trades_df.empty else 0
                st.metric("Gemiddelde PNL", f"€{avg_pnl:.2f}")
                
            with col4:
                daily_pnl = trades_df[
                    trades_df["Timestamp"].dt.date == date.today()
                ]["PNL"].sum() if not trades_df.empty else 0
                st.metric("Dagelijkse PNL", f"€{daily_pnl:.2f}")
                
            # Toon grafieken
            self.create_plot(trades_df)
            
            # Toon recente trades
            st.subheader("Recente Trades")
            st.dataframe(
                trades_df.tail(10).style.format({
                    "PNL": "€{:.2f}",
                    "Equity": "€{:.2f}",
                    "Size": "{:.4f}",
                    "Price": "€{:.2f}"
                })
            )
            
        except Exception as e:
            self.log.error(f"Fout bij updaten dashboard: {e}")
            st.error(f"Fout bij updaten dashboard: {e}")
            
    def start(self) -> None:
        """Start het dashboard."""
        try:
            st.title(self.config.title)
            
            # Sidebar
            st.sidebar.title("Instellingen")
            
            # Tijdsframe selector
            timeframe = st.sidebar.selectbox(
                "Tijdsframe",
                [tf.value for tf in TimeFrame],
                index=[tf.value for tf in TimeFrame].index(self.config.default_timeframe.value)
            )
            
            # Auto-refresh
            auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
            if auto_refresh:
                time.sleep(self.config.refresh_interval)
                st.experimental_rerun()
                
            # Update dashboard
            self.update_dashboard()
            
        except Exception as e:
            self.log.error(f"Fout bij starten dashboard: {e}")
            st.error(f"Fout bij starten dashboard: {e}")
            
    def stop(self) -> None:
        """Stop het dashboard."""
        try:
            self.log.info("Dashboard gestopt")
            
        except Exception as e:
            self.log.error(f"Fout bij stoppen dashboard: {e}")

# --------------------------------------------------------
# Deel 7: Main
# --------------------------------------------------------

def main() -> None:
    """Start het dashboard."""
    try:
        # Maak configuraties
        config = DashboardConfig()
        env_loader = check_environment()
        telegram_alert = TelegramAlert()
        
        # Start dashboard
        dashboard = Dashboard(
            config=config,
            env_loader=env_loader,
            telegram_alert=telegram_alert
        )
        dashboard.start()
        
    except Exception as e:
        logging.error(f"Fout in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
