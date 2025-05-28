# --------------------------------------------------------
# Deel 1: Imports en basisconfiguratie
# --------------------------------------------------------

from fpdf import FPDF
from datetime import datetime
import os
import requests
from dotenv import load_dotenv
import logging
import sys
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from decimal import Decimal, ROUND_DOWN
import time
from enum import Enum
import traceback
from functools import wraps
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import threading
from telegram.ext import Updater, CommandHandler, MessageHandler, Application

# Eigen modules
from utils.logger import Logger, LogConfig
from utils.env_loader import check_environment, get_api_keys, get_telegram_config
from utils.shared.enums import AlertType

# ✅ Laad omgevingsvariabelen (o.a. TELEGRAM_TOKEN en CHAT_ID)
load_dotenv()

# 📁 Paddefinities
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG = os.path.join(BASE_DIR, '..', 'logs', 'trade_log.txt')
EQUITY_FILE = os.path.join(BASE_DIR, '..', 'logs', 'equity_track.txt')
DEFAULT_EQUITY = 10000.0  # Aanpasbaar beginsaldo

# 📋 Logging configuratie
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------
# Deel 2: PDFReport Klasse met header, footer, samenvatting en trade details
# ---------------------------------------------------------

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.today = datetime.now().strftime("%Y-%m-%d %H:%M")
        logging.info(f"🚀 Nieuwe PDFReport object aangemaakt voor {self.today}")

    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, f" Trading Dagrapport - {self.today}", ln=True, align="C")
        self.ln(5)
        logging.debug(f"📄 Header toegevoegd aan PDF: Trading Dagrapport - {self.today}")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Pagina {self.page_no()} - gegenereerd op {self.today}", 0, 0, "C")
        logging.debug(f"📄 Footer toegevoegd aan PDF: Pagina {self.page_no()}")

    def add_summary(self, total_trades, wins, losses, total_pnl, equity):
        self.set_font("Arial", size=12)
        self.cell(0, 10, f"Totaal Trades: {total_trades}", ln=True)
        self.cell(0, 10, f" Winsten: {wins} | ❌ Verliezen: {losses}", ln=True)
        self.cell(0, 10, f" Totaal PNL: €{round(total_pnl, 2)}", ln=True)
        self.cell(0, 10, f" Huidige Equity: €{round(equity, 2)}", ln=True)
        logging.info(f"📊 Samenvatting toegevoegd aan PDF: Totaal PNL: €{round(total_pnl, 2)} | Equity: €{round(equity, 2)}")

    def add_trades(self, trades):
        self.ln(10)
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, " Trades van vandaag:", ln=True)
        self.ln(5)
        self.set_font("Arial", size=10)
        for trade in trades:
            self.multi_cell(0, 8, trade)
            self.ln(1)
        logging.info(f"📈 Trades toegevoegd aan PDF: {len(trades)} trades van vandaag")

# --------------------------------------------------------
# Deel 3: Functie voor het genereren en versturen van dagelijks rapport
# --------------------------------------------------------

def generate_daily_report():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    trades = []
    total_pnl = 0.0
    win_trades = 0
    loss_trades = 0

    # 📥 Lees trade_log.txt
    try:
        with open(TRADE_LOG, "r", encoding="utf-8") as f:
            logs = f.readlines()
            for line in logs:
                if today in line and "PNL:" in line:
                    trades.append(line.strip())
                    try:
                        pnl_str = line.split("PNL:")[-1].split("|")[0].strip()
                        pnl = float(pnl_str)
                        total_pnl += pnl
                        if pnl > 0:
                            win_trades += 1
                        else:
                            loss_trades += 1
                    except ValueError:
                        continue
        logging.info(f"📄 {len(trades)} trades gevonden voor vandaag in {TRADE_LOG}")
    except FileNotFoundError:
        logging.error(f"❌ Het bestand {TRADE_LOG} bestaat niet. Geen trades om te verwerken.")
        return None

    # 📥 Lees equity
    try:
        with open(EQUITY_FILE, "r", encoding="utf-8") as f:
            equity = float(f.read().strip())
        logging.info(f"📈 Equity gelezen uit {EQUITY_FILE}: €{equity}")
    except FileNotFoundError:
        equity = DEFAULT_EQUITY
        logging.warning(f"❌ Het bestand {EQUITY_FILE} bestaat niet. Gebruikt standaard equity van €{DEFAULT_EQUITY}")

    # 📄 Genereer PDF
    pdf = PDFReport()
    pdf.today = today
    pdf.add_page()
    pdf.add_summary(total_trades=len(trades), wins=win_trades, losses=loss_trades, total_pnl=total_pnl, equity=equity)

    if trades:
        pdf.add_trades(trades)

    # 📂 Opslaan
    os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
    filename = f"logs/report_{today}.pdf"
    if os.path.exists(filename):
        filename = f"logs/report_{today}_{datetime.utcnow().strftime('%H%M%S')}.pdf"

    try:
        pdf.output(filename)
        logging.info(f"✅ PDF-rapport succesvol gegenereerd: {filename}")
    except Exception as e:
        logging.error(f"❌ Fout bij opslaan van PDF: {e}")
        return None

    # 📤 Verstuur naar Telegram
    try:
        send_telegram_pdf(filename)
    except Exception as e:
        logging.error(f"❌ PDF kon niet via Telegram worden verstuurd: {e}")

    return trades, win_trades, loss_trades, total_pnl

# --------------------------------------------------------
# 📥 Deel 5: Lees huidige equity en genereer PDF
# --------------------------------------------------------

    # 📈 Probeer equity op te halen
    try:
        with open(EQUITY_FILE, "r", encoding="utf-8") as f:
            equity = float(f.read().strip())
        logging.info(f"📈 Equity succesvol gelezen uit {EQUITY_FILE}: €{equity}")
    except FileNotFoundError:
        equity = DEFAULT_EQUITY
        logging.warning(f"⚠️ Bestand {EQUITY_FILE} niet gevonden — standaard equity €{DEFAULT_EQUITY} gebruikt.")
    except Exception as e:
        equity = DEFAULT_EQUITY
        logging.error(f"❌ Fout bij openen van {EQUITY_FILE}: {e} — standaard equity €{DEFAULT_EQUITY} gebruikt.")

    # 📄 Genereer PDF-rapport
    pdf = PDFReport()
    pdf.today = today
    pdf.add_page()
    pdf.add_summary(
        total_trades=len(trades),
        wins=win_trades,
        losses=loss_trades,
        total_pnl=total_pnl,
        equity=equity
    )

    if trades:
        pdf.add_trades(trades)

    # 📂 Opslaan in logs/
    os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
    logging.info(f"📂 Logmap gecontroleerd en aangemaakt: {os.path.dirname(TRADE_LOG)}")

    filename = os.path.join("logs", f"report_{today}.pdf")
    if os.path.exists(filename):
        timestamp = datetime.utcnow().strftime('%H%M%S')
        filename = os.path.join("logs", f"report_{today}_{timestamp}.pdf")
        logging.info(f"🕒 Bestandsnaam aangepast door timestamp: {filename}")

    try:
        pdf.output(filename)
        logging.info(f"✅ PDF-rapport succesvol gegenereerd: {filename}")
    except Exception as e:
        logging.error(f"❌ Fout bij opslaan van PDF: {e}")

#--------------------------------------------------------
# Deel 6: Functie om een bericht te versturen via Telegram
# --------------------------------------------------------

load_dotenv()

# --------------------------------------------------------
# Deel 1: Constants en Configuratie
# --------------------------------------------------------

@dataclass
class TelegramConfig:
    """Configuratie voor Telegram alerts."""
    token: str
    chat_id: str
    enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 5
    rate_limit: int = 30  # berichten per minuut

class TelegramAlert:
    """Class voor Telegram alerts."""
    
    def __init__(self, config: Optional[TelegramConfig] = None):
        """Initialiseer de Telegram alert."""
        self.config = config or TelegramConfig(
            token=get_telegram_config()[0],
            chat_id=get_telegram_config()[1]
        )
        self.logger = logging.getLogger(__name__)
    
    def send_message(self, message: str, alert_type: AlertType = AlertType.SYSTEM) -> bool:
        """Stuur een bericht naar Telegram."""
        if not self.config.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.config.token}/sendMessage"
            data = {
                "chat_id": self.config.chat_id,
                "text": f"[{alert_type.value}] {message}",
                "parse_mode": "HTML"
            }
            
            for _ in range(self.config.max_retries):
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    return True
                time.sleep(self.config.retry_delay)
            
            self.logger.error(f"Failed to send Telegram message after {self.config.max_retries} retries")
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    def send_trade_alert(self, message: str) -> bool:
        """Stuur een trade alert."""
        return self.send_message(message, AlertType.TRADE)
    
    def send_error_alert(self, message: str) -> bool:
        """Stuur een error alert."""
        return self.send_message(message, AlertType.ERROR)
    
    def send_system_alert(self, message: str) -> bool:
        """Stuur een system alert."""
        return self.send_message(message, AlertType.SYSTEM)
    
    def send_warning_alert(self, message: str) -> bool:
        """Stuur een warning alert."""
        return self.send_message(message, AlertType.WARNING)
    
    def send_performance_alert(self, message: str) -> bool:
        """Stuur een performance alert."""
        return self.send_message(message, AlertType.PERFORMANCE)
    
    def send_security_alert(self, message: str) -> bool:
        """Stuur een security alert."""
        return self.send_message(message, AlertType.SECURITY)

def send_telegram_message(message: str, alert_type: AlertType = AlertType.SYSTEM) -> bool:
    """Stuur een bericht naar Telegram."""
    alert = TelegramAlert()
    return alert.send_message(message, alert_type)

# --------------------------------------------------------
# Deel 2: Main voor testing
# --------------------------------------------------------

def main():
    """Test de Telegram alert service."""
    try:
        # Maak Telegram alert instance
        telegram = TelegramAlert()
        
        # Test verschillende alerts
        telegram.send_alert(AlertType.INFO, "Test info alert")
        telegram.send_alert(AlertType.SUCCESS, "Test success alert")
        telegram.send_alert(AlertType.WARNING, "Test warning alert")
        telegram.send_alert(AlertType.ERROR, "Test error alert")
        
        print("Telegram alert test succesvol uitgevoerd")
        
    except Exception as e:
        print(f"Kritieke fout in Telegram alert test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
