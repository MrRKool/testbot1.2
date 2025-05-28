import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from fpdf import FPDF
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from enum import Enum
from utils.telegram_alerts import TelegramAlert, AlertType
from utils.env_loader import check_environment, get_api_keys, get_telegram_config
from utils.logger import Logger
from utils.shared.enums import TimeFrame, ReportType, ChartType

TRADE_LOG = "logs/trade_log.txt"
EQUITY_FILE = "logs/equity_track.txt"
DEFAULT_EQUITY = 10000.0  # Dit kan altijd aangepast worden

# 📋 Logging configuratie
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------------
# Deel 1: Enums en Constants
# --------------------------------------------------------

# --------------------------------------------------------
# Deel 2: Configuratie Classes
# --------------------------------------------------------

@dataclass
class ReportConfig:
    """Configuratie voor PDF rapporten."""
    # Basis instellingen
    output_dir: str = "reports"
    title: str = "Trading Bot Rapport"
    author: str = "Trading Bot"
    font_family: str = "Arial"
    font_size: int = 12
    margin: int = 20
    line_height: int = 10
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Grafiek instellingen
    chart_style: str = "seaborn"
    chart_dpi: int = 300
    chart_size: Tuple[int, int] = (10, 6)
    chart_colors: List[str] = field(default_factory=lambda: ["#2ecc71", "#e74c3c", "#3498db"])
    
    # Rapport instellingen
    currency: str = "EUR"
    decimal_places: int = 2
    timezone: str = "UTC"
    default_timeframe: TimeFrame = TimeFrame.DAY
    
    # Email instellingen
    email_enabled: bool = False
    email_recipients: List[str] = field(default_factory=list)
    email_subject: str = "Trading Bot Rapport"
    
    # Backup instellingen
    backup_enabled: bool = True
    backup_retention_days: int = 30
    backup_dir: str = "backups"
    
    # Monitoring instellingen
    monitor_enabled: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_drawdown": 20.0,
        "min_winrate": 40.0,
        "min_profit_factor": 1.5
    })

@dataclass
class ChartConfig:
    """Configuratie voor grafieken."""
    style: str = "seaborn"
    dpi: int = 300
    size: Tuple[int, int] = (10, 6)
    colors: List[str] = field(default_factory=lambda: ["#2ecc71", "#e74c3c", "#3498db"])
    grid: bool = True
    legend: bool = True
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    cmap: str = "RdYlGn"
    alpha: float = 0.7
    line_width: float = 2.0
    marker_size: int = 6

# --------------------------------------------------------
# Deel 3: Custom PDF Class
# --------------------------------------------------------

class TradingPDF(FPDF):
    """Aangepaste PDF class met extra functionaliteit voor trading rapporten."""
    
    def __init__(
        self,
        config: ReportConfig,
        env_loader: Optional[check_environment] = None,
        telegram_alert: Optional[TelegramAlert] = None
    ):
        """Initialiseer de PDF met configuratie."""
        super().__init__()
        self.config = config
        self.env_loader = env_loader or check_environment()
        self.telegram_alert = telegram_alert
        self.logger = Logger(__name__)
        
        self.set_auto_page_break(auto=True, margin=self.config.margin)
        self.add_page()
        self.set_font(self.config.font_family, "", self.config.font_size)
        
    def header(self):
        """Voeg header toe aan elke pagina."""
        try:
            # Logo (indien beschikbaar)
            logo_path = Path("assets/logo.png")
            if logo_path.exists():
                self.image(str(logo_path), 10, 8, 33)
            
            # Titel
            self.set_font(self.config.font_family, "B", 15)
            self.cell(0, 10, self.config.title, 0, 1, "C")
            
            # Datum
            self.set_font(self.config.font_family, "I", 8)
            self.cell(
                0,
                10,
                f"Gegenereerd op: {datetime.now().strftime(self.config.date_format)}",
                0,
                1,
                "C"
            )
            
            # Lijn
            self.line(10, 30, 200, 30)
            self.ln(10)
            
        except Exception as e:
            self.logger.error(f"Fout bij toevoegen header: {e}")
            
    def footer(self):
        """Voeg footer toe aan elke pagina."""
        try:
            self.set_y(-15)
            self.set_font(self.config.font_family, "I", 8)
            self.cell(0, 10, f"Pagina {self.page_no()}", 0, 0, "C")
            
        except Exception as e:
            self.logger.error(f"Fout bij toevoegen footer: {e}")
            
    def add_chart(
        self,
        fig: plt.Figure,
        title: str,
        description: str = "",
        chart_type: Optional[ChartType] = None
    ):
        """Voeg een matplotlib chart toe aan de PDF."""
        try:
            # Sla de chart op als tijdelijke PNG
            img_data = BytesIO()
            fig.savefig(
                img_data,
                format="png",
                dpi=self.config.chart_dpi,
                bbox_inches="tight"
            )
            img_data.seek(0)
            
            # Voeg titel en beschrijving toe
            self.set_font(self.config.font_family, "B", 12)
            self.cell(0, 10, title, 0, 1, "L")
            
            if description:
                self.set_font(self.config.font_family, "", 10)
                self.multi_cell(0, 5, description)
                self.ln(5)
            
            # Voeg de chart toe
            self.image(img_data, x=10, w=190)
            self.ln(10)
            
            # Sluit de figure
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"Fout bij toevoegen chart: {e}")
            
    def add_table(
        self,
        data: List[List[str]],
        headers: List[str],
        col_widths: Optional[List[int]] = None,
        title: str = "",
        description: str = ""
    ):
        """Voeg een tabel toe aan de PDF."""
        try:
            # Voeg titel en beschrijving toe
            if title:
                self.set_font(self.config.font_family, "B", 12)
                self.cell(0, 10, title, 0, 1, "L")
                
            if description:
                self.set_font(self.config.font_family, "", 10)
                self.multi_cell(0, 5, description)
                self.ln(5)
            
            # Bepaal kolombreedtes
            if not col_widths:
                col_widths = [190 // len(headers)] * len(headers)
            
            # Voeg headers toe
            self.set_font(self.config.font_family, "B", 10)
            for i, header in enumerate(headers):
                self.cell(col_widths[i], 7, header, 1, 0, "C")
            self.ln()
            
            # Voeg data toe
            self.set_font(self.config.font_family, "", 10)
            for row in data:
                for i, cell in enumerate(row):
                    self.cell(col_widths[i], 6, str(cell), 1, 0, "C")
                self.ln()
            self.ln(10)
            
        except Exception as e:
            self.logger.error(f"Fout bij toevoegen tabel: {e}")
            
    def add_statistics(
        self,
        stats: Dict[str, Any],
        title: str = "Statistieken",
        description: str = ""
    ):
        """Voeg statistieken toe aan de PDF."""
        try:
            # Voeg titel en beschrijving toe
            self.set_font(self.config.font_family, "B", 12)
            self.cell(0, 10, title, 0, 1, "L")
            
            if description:
                self.set_font(self.config.font_family, "", 10)
                self.multi_cell(0, 5, description)
                self.ln(5)
            
            # Voeg statistieken toe
            self.set_font(self.config.font_family, "", 10)
            for key, value in stats.items():
                if isinstance(value, float):
                    text = f"{key}: {self.config.currency}{value:.2f}"
                else:
                    text = f"{key}: {value}"
                self.cell(0, 6, text, 0, 1, "L")
                
            self.ln(5)
            
        except Exception as e:
            self.logger.error(f"Fout bij toevoegen statistieken: {e}")
            
    def add_section(
        self,
        title: str,
        content: str,
        level: int = 1
    ):
        """Voeg een sectie toe aan de PDF."""
        try:
            # Bepaal font grootte op basis van level
            font_size = 16 - (level * 2)
            self.set_font(self.config.font_family, "B", font_size)
            
            # Voeg titel toe
            self.cell(0, 10, title, 0, 1, "L")
            
            # Voeg content toe
            self.set_font(self.config.font_family, "", 10)
            self.multi_cell(0, 5, content)
            self.ln(5)
            
        except Exception as e:
            self.logger.error(f"Fout bij toevoegen sectie: {e}")
            
    def add_page_break(self):
        """Voeg een pagina-einde toe."""
        try:
            self.add_page()
            
        except Exception as e:
            self.logger.error(f"Fout bij toevoegen pagina-einde: {e}")

# --------------------------------------------------------
# Deel 4: Report Generator
# --------------------------------------------------------

class ReportGenerator:
    """Class voor het genereren van rapporten."""
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        env_loader: Optional[check_environment] = None,
        telegram_alert: Optional[TelegramAlert] = None
    ):
        """Initialiseer de report generator."""
        self.config = config or ReportConfig()
        self.env_loader = env_loader or check_environment()
        self.telegram_alert = telegram_alert
        self.logger = Logger(__name__)
        
        self.setup_style()
        
    def setup_style(self):
        """Configureer de grafiek stijl."""
        try:
            if self.config.chart_style == "seaborn":
                sns.set_style("darkgrid")
                plt.rcParams["figure.figsize"] = self.config.chart_size
                plt.rcParams["figure.dpi"] = self.config.chart_dpi
                plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self.config.chart_colors)
                
            self.logger.info(f"Grafiek stijl geconfigureerd: {self.config.chart_style}")
            
        except Exception as e:
            self.logger.error(f"Fout bij configureren grafiek stijl: {e}")
            
    def generate_performance_chart(
        self,
        trades_df: pd.DataFrame,
        chart_type: ChartType = ChartType.EQUITY
    ) -> plt.Figure:
        """Genereer een performance chart."""
        try:
            if chart_type == ChartType.EQUITY:
                return self._generate_equity_chart(trades_df)
            elif chart_type == ChartType.PNL:
                return self._generate_pnl_chart(trades_df)
            elif chart_type == ChartType.WIN_LOSS:
                return self._generate_win_loss_chart(trades_df)
            elif chart_type == ChartType.DRAWDOWN:
                return self._generate_drawdown_chart(trades_df)
            elif chart_type == ChartType.VOLATILITY:
                return self._generate_volatility_chart(trades_df)
            elif chart_type == ChartType.HEATMAP:
                return self._generate_heatmap_chart(trades_df)
            elif chart_type == ChartType.CORRELATION:
                return self._generate_correlation_chart(trades_df)
            elif chart_type == ChartType.DISTRIBUTION:
                return self._generate_distribution_chart(trades_df)
            else:
                raise ValueError(f"Onbekend chart type: {chart_type}")
                
        except Exception as e:
            self.logger.error(f"Fout bij genereren performance chart: {e}")
            return plt.figure()
            
    def _generate_equity_chart(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Genereer equity curve chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Bereken cumulative PNL
            trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
            
            # Plot equity curve
            ax.plot(
                trades_df.index,
                trades_df["cumulative_pnl"],
                color=self.config.chart_colors[0],
                linewidth=2
            )
            
            ax.set_title("Equity Curve")
            ax.set_xlabel("Datum")
            ax.set_ylabel("Cumulatieve PNL")
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren equity chart: {e}")
            return plt.figure()
            
    def _generate_pnl_chart(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Genereer PNL chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Plot PNL per trade
            ax.bar(
                trades_df.index,
                trades_df["pnl"],
                color=self.config.chart_colors[0],
                alpha=0.7
            )
            
            ax.set_title("PNL per Trade")
            ax.set_xlabel("Datum")
            ax.set_ylabel("PNL")
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren PNL chart: {e}")
            return plt.figure()
            
    def _generate_win_loss_chart(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Genereer Win/Loss chart."""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Bereken win/loss ratio
            wins = len(trades_df[trades_df["pnl"] > 0])
            losses = len(trades_df[trades_df["pnl"] <= 0])
            
            # Maak pie chart
            ax.pie(
                [wins, losses],
                labels=["Winst", "Verlies"],
                colors=[self.config.chart_colors[0], self.config.chart_colors[1]],
                autopct="%1.1f%%",
                startangle=90
            )
            
            ax.set_title("Win/Loss Ratio")
            ax.axis("equal")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren Win/Loss chart: {e}")
            return plt.figure()
            
    def _generate_drawdown_chart(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Genereer Drawdown chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Bereken drawdown
            trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
            trades_df["peak"] = trades_df["cumulative_pnl"].cummax()
            trades_df["drawdown"] = (
                (trades_df["peak"] - trades_df["cumulative_pnl"]) /
                trades_df["peak"] * 100
            )
            
            # Plot drawdown
            ax.plot(
                trades_df.index,
                trades_df["drawdown"],
                color=self.config.chart_colors[1],
                linewidth=2
            )
            
            ax.set_title("Drawdown Over Tijd")
            ax.set_xlabel("Datum")
            ax.set_ylabel("Drawdown %")
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren Drawdown chart: {e}")
            return plt.figure()
            
    def _generate_volatility_chart(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Genereer Volatility chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Bereken rolling volatility
            trades_df["returns"] = trades_df["pnl"].pct_change()
            trades_df["volatility"] = trades_df["returns"].rolling(window=20).std() * np.sqrt(252)
            
            # Plot volatility
            ax.plot(
                trades_df.index,
                trades_df["volatility"] * 100,
                color=self.config.chart_colors[2],
                linewidth=2
            )
            
            ax.set_title("Volatiliteit Over Tijd")
            ax.set_xlabel("Datum")
            ax.set_ylabel("Volatiliteit %")
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren Volatility chart: {e}")
            return plt.figure()
            
    def _generate_heatmap_chart(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Genereer Heatmap chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Bereid data voor
            trades_df["hour"] = trades_df.index.hour
            trades_df["day"] = trades_df.index.strftime("%A")
            
            # Maak pivot table
            pivot = trades_df.pivot_table(
                values="pnl",
                index="day",
                columns="hour",
                aggfunc="mean"
            )
            
            # Plot heatmap
            sns.heatmap(
                pivot,
                cmap=self.config.chart_colors[0],
                center=0,
                ax=ax,
                cbar_kws={"label": "Gemiddelde PNL"}
            )
            
            ax.set_title("PNL Heatmap per Uur en Dag")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren Heatmap chart: {e}")
            return plt.figure()
            
    def _generate_correlation_chart(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Genereer Correlation chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Bereken correlaties
            corr_matrix = trades_df[["pnl", "size", "duration"]].corr()
            
            # Plot correlatie matrix
            sns.heatmap(
                corr_matrix,
                cmap=self.config.chart_colors[0],
                center=0,
                ax=ax,
                annot=True,
                fmt=".2f"
            )
            
            ax.set_title("Correlatie Matrix")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren Correlation chart: {e}")
            return plt.figure()
            
    def _generate_distribution_chart(self, trades_df: pd.DataFrame) -> plt.Figure:
        """Genereer Distribution chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Plot PNL distributie
            sns.histplot(
                trades_df["pnl"],
                kde=True,
                color=self.config.chart_colors[0],
                ax=ax
            )
            
            ax.set_title("PNL Distributie")
            ax.set_xlabel("PNL")
            ax.set_ylabel("Frequentie")
            ax.grid(True)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren Distribution chart: {e}")
            return plt.figure()
            
    def generate_trade_statistics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Genereer trade statistieken."""
        try:
            stats = {}
            
            # Basis statistieken
            stats["total_trades"] = len(trades_df)
            stats["win_count"] = len(trades_df[trades_df["pnl"] > 0])
            stats["loss_count"] = len(trades_df[trades_df["pnl"] <= 0])
            stats["win_rate"] = stats["win_count"] / stats["total_trades"] * 100
            stats["pnl_total"] = trades_df["pnl"].sum()
            stats["avg_win"] = trades_df[trades_df["pnl"] > 0]["pnl"].mean()
            stats["avg_loss"] = trades_df[trades_df["pnl"] <= 0]["pnl"].mean()
            stats["largest_win"] = trades_df["pnl"].max()
            stats["largest_loss"] = trades_df["pnl"].min()
            
            # Performance metrics
            returns = trades_df["pnl"].values
            stats["sharpe_ratio"] = (
                np.mean(returns) / np.std(returns)
                if np.std(returns) != 0 else 0
            )
            
            neg_returns = returns[returns < 0]
            stats["sortino_ratio"] = (
                np.mean(returns) / np.std(neg_returns)
                if len(neg_returns) > 0 and np.std(neg_returns) != 0 else 0
            )
            
            # Drawdown metrics
            cumulative_pnl = trades_df["pnl"].cumsum()
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (peak - cumulative_pnl) / peak * 100
            stats["max_drawdown"] = np.max(drawdown)
            
            # Risico metrics
            if stats["avg_loss"] != 0:
                stats["risk_reward_ratio"] = abs(stats["avg_win"] / stats["avg_loss"])
                stats["kelly_criterion"] = (
                    (stats["win_rate"] / 100) -
                    ((1 - stats["win_rate"] / 100) / stats["risk_reward_ratio"])
                )
                
            # Tijd gebaseerde metrics
            stats["trades_per_day"] = stats["total_trades"] / (
                (trades_df.index[-1] - trades_df.index[0]).days + 1
            )
            
            hours = trades_df.index.hour
            stats["avg_trades_per_hour"] = len(hours) / 24
            
            hour_counts = pd.Series(hours).value_counts()
            if not hour_counts.empty:
                stats["best_trading_hour"] = f"{hour_counts.index[0]:02d}:00"
                stats["worst_trading_hour"] = f"{hour_counts.index[-1]:02d}:00"
                
            days = trades_df.index.strftime("%A")
            day_counts = pd.Series(days).value_counts()
            if not day_counts.empty:
                stats["best_trading_day"] = day_counts.index[0]
                stats["worst_trading_day"] = day_counts.index[-1]
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren trade statistieken: {e}")
            return {}
            
    def generate_report(
        self,
        trades_df: pd.DataFrame,
        report_type: ReportType = ReportType.DAILY,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[str]:
        """Genereer een compleet rapport."""
        try:
            # Filter trades op periode
            if start_date:
                trades_df = trades_df[trades_df.index >= start_date]
            if end_date:
                trades_df = trades_df[trades_df.index <= end_date]
                
            # Maak PDF
            pdf = TradingPDF(
                config=self.config,
                env_loader=self.env_loader,
                telegram_alert=self.telegram_alert
            )
            
            # Voeg titel toe
            pdf.set_font(pdf.config.font_family, "B", 16)
            pdf.cell(0, 10, f"Trading Rapport - {report_type.value.capitalize()}", 0, 1, "C")
            pdf.ln(10)
            
            # Genereer statistieken
            stats = self.generate_trade_statistics(trades_df)
            
            # Voeg statistieken toe
            pdf.add_statistics(
                stats,
                title="Trading Statistieken",
                description="Overzicht van de belangrijkste trading metrics."
            )
            
            # Genereer en voeg grafieken toe
            for chart_type in ChartType:
                fig = self.generate_performance_chart(trades_df, chart_type)
                pdf.add_chart(
                    fig,
                    title=f"{chart_type.value.title()} Chart",
                    description=f"Visualisatie van {chart_type.value} metrics."
                )
                
            # Voeg trades toe
            pdf.add_table(
                data=trades_df.tail(10).values.tolist(),
                headers=trades_df.columns.tolist(),
                title="Recente Trades",
                description="Overzicht van de laatste 10 trades."
            )
            
            # Sla rapport op
            report_path = Path(self.config.output_dir) / f"{report_type.value}_report.pdf"
            pdf.output(str(report_path))
            
            # Stuur rapport
            if self.telegram_alert:
                self._send_telegram_report(report_type, stats, str(report_path))
                
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Fout bij genereren rapport: {e}")
            return None
            
    def _send_telegram_report(
        self,
        report_type: ReportType,
        stats: Dict[str, Any],
        pdf_path: str
    ) -> None:
        """Stuur rapport via Telegram."""
        try:
            if not self.telegram_alert:
                return
                
            # Maak rapport tekst
            report_text = (
                f"📊 {report_type.value.title()} Rapport\n\n"
                f"Totaal Trades: {stats['total_trades']}\n"
                f"Winrate: {stats['win_rate']:.1f}%\n"
                f"Gemiddelde PNL: {self.config.currency}{stats['pnl_total']/stats['total_trades']:.2f}\n"
                f"Profit Factor: {stats['profit_factor']:.2f}\n"
                f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}\n"
                f"Max Drawdown: {stats['max_drawdown']:.1f}%\n\n"
                f"Beste Trading Uur: {stats['best_trading_hour']}\n"
                f"Beste Trading Dag: {stats['best_trading_day']}"
            )
            
            # Stuur rapport
            self.telegram_alert.send_alert(
                AlertType.REPORT,
                report_text
            )
            
            # Stuur PDF als bijlage
            if os.path.exists(pdf_path):
                self.telegram_alert.send_document(
                    pdf_path,
                    f"{report_type.value}_report.pdf"
                )
                
        except Exception as e:
            self.logger.error(f"Fout bij versturen Telegram rapport: {e}")

# --------------------------------------------------------
# Deel 5: Main voor testing
# --------------------------------------------------------

def main():
    """Test de report generator."""
    try:
        # Maak configuraties
        report_config = ReportConfig()
        env_loader = check_environment()
        telegram_alert = TelegramAlert()
        
        # Maak report generator
        generator = ReportGenerator(
            config=report_config,
            env_loader=env_loader,
            telegram_alert=telegram_alert
        )
        
        # Maak test data
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="H")
        trades_df = pd.DataFrame({
            "pnl": np.random.normal(100, 50, len(dates)),
            "size": np.random.uniform(0.1, 1.0, len(dates)),
            "duration": np.random.uniform(5, 60, len(dates))
        }, index=dates)
        
        # Genereer rapport
        pdf_path = generator.generate_report(
            trades_df,
            ReportType.DAILY,
            start_date=datetime.now() - timedelta(days=1),
            end_date=datetime.now()
        )
        
        if pdf_path:
            print(f"Rapport gegenereerd: {pdf_path}")
            
    except Exception as e:
        print(f"Fout in main: {e}")

if __name__ == "__main__":
    main()

