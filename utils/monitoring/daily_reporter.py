import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from fpdf import FPDF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from utils.telegram_alerts import TelegramAlert
from utils.env_loader import check_environment, get_api_keys, get_telegram_config
from utils.logger import Logger
import numpy as np
from utils.shared.enums import TimeFrame, ReportFormat, ReportType, AlertType
from utils.shared.configs import ReportConfig
from utils.monitoring.pdf_reporter import ChartType, ReportConfig as PDFReportConfig

# --------------------------------------------------------
# Deel 1: Enums en Constants
# --------------------------------------------------------

# --------------------------------------------------------
# Deel 2: Configuratie Classes
# --------------------------------------------------------

@dataclass
class TradeStats:
    """Class voor trade statistieken."""
    # Basis statistieken
    total_trades: int = 0
    win_count: int = 0
    loss_count: int = 0
    pnl_total: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    equity: float = 0.0
    drawdown: float = 0.0
    profit_factor: float = 0.0
    
    # Uitgebreide statistieken
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration: float = 0.0
    best_trade: Dict[str, Any] = field(default_factory=dict)
    worst_trade: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risico metrics
    risk_reward_ratio: float = 0.0
    kelly_criterion: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    
    # Tijd gebaseerde metrics
    trades_per_day: float = 0.0
    avg_trades_per_hour: float = 0.0
    best_trading_hour: str = ""
    worst_trading_hour: str = ""
    best_trading_day: str = ""
    worst_trading_day: str = ""

class DailyReporter:
    """Class voor het genereren van dagelijkse rapporten."""
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        env_loader: Optional[check_environment] = None,
        telegram_alert: Optional[TelegramAlert] = None
    ):
        """Initialiseer de reporter."""
        self.config = config or ReportConfig()
        self.env_loader = env_loader or check_environment()
        self.telegram_alert = telegram_alert
        self.logger = Logger()
        self.log = self.logger.get_logger(__name__)
        
        self._setup_directories()
        self._setup_chart_style()
        
    def _setup_directories(self) -> None:
        """Maak benodigde directories aan."""
        try:
            for directory in [
                self.config.log_dir,
                self.config.report_dir,
                self.config.chart_dir,
                self.config.backup_dir
            ]:
                Path(directory).mkdir(parents=True, exist_ok=True)
                
            self.log.info("Directories succesvol aangemaakt")
            
        except Exception as e:
            self.log.error(f"Fout bij aanmaken directories: {e}")
            raise
            
    def _setup_chart_style(self) -> None:
        """Configureer grafiek stijl."""
        try:
            if self.config.chart_style == "seaborn":
                sns.set_style("darkgrid")
                plt.rcParams["figure.figsize"] = self.config.chart_size
                plt.rcParams["figure.dpi"] = self.config.chart_dpi
                plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self.config.chart_colors)
                
            self.log.info(f"Grafiek stijl geconfigureerd: {self.config.chart_style}")
            
        except Exception as e:
            self.log.error(f"Fout bij configureren grafiek stijl: {e}")
            
    def read_trades(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[List[Dict[str, Any]], TradeStats]:
        """Lees trades binnen een bepaalde periode."""
        trades = []
        stats = TradeStats()
        
        try:
            trade_log = Path(self.config.log_dir) / self.config.trade_log
            if not trade_log.exists():
                self.log.warning(f"Trade log niet gevonden: {trade_log}")
                return trades, stats
                
            with open(trade_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        # Parse trade data
                        trade_data = self._parse_trade_line(line)
                        if not trade_data:
                            continue
                            
                        timestamp = datetime.strptime(
                            trade_data["timestamp"],
                            "%Y-%m-%d %H:%M:%S"
                        )
                        
                        if start_date <= timestamp <= end_date:
                            trades.append(trade_data)
                            self._update_stats(stats, trade_data)
                            
                    except Exception as e:
                        self.log.error(f"Fout bij parsen trade regel: {e}")
                        continue
                        
            # Bereken extra statistieken
            self._calculate_advanced_stats(stats, trades)
            
            # Check voor alerts
            self._check_alerts(stats)
            
            return trades, stats
            
        except Exception as e:
            self.log.error(f"Fout bij lezen trades: {e}")
            return trades, stats
            
    def _parse_trade_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse een trade regel naar een dictionary."""
        try:
            parts = line.split("|")
            if len(parts) < 3:
                return None
                
            return {
                "timestamp": parts[0].strip(),
                "symbol": parts[1].strip(),
                "side": parts[2].strip(),
                "entry_price": float(parts[3]),
                "exit_price": float(parts[4]),
                "size": float(parts[5]),
                "pnl": float(parts[6]),
                "duration": float(parts[7]) if len(parts) > 7 else 0.0
            }
            
        except Exception as e:
            self.log.error(f"Fout bij parsen trade regel: {e}")
            return None
            
    def _update_stats(self, stats: TradeStats, trade: Dict[str, Any]) -> None:
        """Update statistieken met nieuwe trade."""
        try:
            # Update basis statistieken
            stats.total_trades += 1
            stats.pnl_total += trade["pnl"]
            
            if trade["pnl"] > 0:
                stats.win_count += 1
                stats.consecutive_wins += 1
                stats.consecutive_losses = 0
                stats.avg_win = (
                    (stats.avg_win * (stats.win_count - 1) + trade["pnl"]) /
                    stats.win_count
                )
                stats.largest_win = max(stats.largest_win, trade["pnl"])
            else:
                stats.loss_count += 1
                stats.consecutive_losses += 1
                stats.consecutive_wins = 0
                stats.avg_loss = (
                    (stats.avg_loss * (stats.loss_count - 1) + trade["pnl"]) /
                    stats.loss_count
                )
                stats.largest_loss = min(stats.largest_loss, trade["pnl"])
                
            # Update max consecutive
            stats.max_consecutive_wins = max(
                stats.max_consecutive_wins,
                stats.consecutive_wins
            )
            stats.max_consecutive_losses = max(
                stats.max_consecutive_losses,
                stats.consecutive_losses
            )
            
            # Update best/worst trade
            if not stats.best_trade or trade["pnl"] > stats.best_trade["pnl"]:
                stats.best_trade = trade
            if not stats.worst_trade or trade["pnl"] < stats.worst_trade["pnl"]:
                stats.worst_trade = trade
                
        except Exception as e:
            self.log.error(f"Fout bij updaten statistieken: {e}")
            
    def _calculate_advanced_stats(
        self,
        stats: TradeStats,
        trades: List[Dict[str, Any]]
    ) -> None:
        """Bereken geavanceerde statistieken."""
        try:
            if not trades:
                return
                
            # Bereken winrate en profit factor
            stats.win_rate = (stats.win_count / stats.total_trades) * 100
            if stats.loss_count > 0:
                stats.profit_factor = abs(
                    stats.avg_win * stats.win_count /
                    (stats.avg_loss * stats.loss_count)
                )
                
            # Bereken performance metrics
            returns = [t["pnl"] for t in trades]
            if returns:
                stats.sharpe_ratio = (
                    np.mean(returns) / np.std(returns)
                    if np.std(returns) != 0 else 0
                )
                
                # Sortino ratio (alleen negatieve returns)
                neg_returns = [r for r in returns if r < 0]
                if neg_returns:
                    stats.sortino_ratio = (
                        np.mean(returns) / np.std(neg_returns)
                        if np.std(neg_returns) != 0 else 0
                    )
                    
            # Bereken drawdown
            cumulative_pnl = np.cumsum([t["pnl"] for t in trades])
            if len(cumulative_pnl) > 0:
                peak = np.maximum.accumulate(cumulative_pnl)
                drawdown = (peak - cumulative_pnl) / peak * 100
                stats.max_drawdown = np.max(drawdown)
                
            # Bereken expectancy
            stats.expectancy = (
                (stats.win_rate / 100 * stats.avg_win) +
                ((1 - stats.win_rate / 100) * stats.avg_loss)
            )
            
            # Bereken risico metrics
            if stats.avg_loss != 0:
                stats.risk_reward_ratio = abs(stats.avg_win / stats.avg_loss)
                stats.kelly_criterion = (
                    (stats.win_rate / 100) -
                    ((1 - stats.win_rate / 100) / stats.risk_reward_ratio)
                )
                
            # Bereken VaR en CVaR
            if returns:
                returns_sorted = np.sort(returns)
                var_index = int(len(returns) * 0.05)
                stats.var_95 = returns_sorted[var_index]
                stats.cvar_95 = np.mean(returns_sorted[:var_index])
                
            # Bereken tijd gebaseerde metrics
            if trades:
                trade_times = [datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S") for t in trades]
                days = (trade_times[-1] - trade_times[0]).days + 1
                stats.trades_per_day = stats.total_trades / days
                
                hours = [t.hour for t in trade_times]
                stats.avg_trades_per_hour = len(hours) / 24
                
                # Best/worst trading uur
                hour_counts = pd.Series(hours).value_counts()
                if not hour_counts.empty:
                    stats.best_trading_hour = f"{hour_counts.index[0]:02d}:00"
                    stats.worst_trading_hour = f"{hour_counts.index[-1]:02d}:00"
                    
                # Best/worst trading dag
                days = [t.strftime("%A") for t in trade_times]
                day_counts = pd.Series(days).value_counts()
                if not day_counts.empty:
                    stats.best_trading_day = day_counts.index[0]
                    stats.worst_trading_day = day_counts.index[-1]
                    
        except Exception as e:
            self.log.error(f"Fout bij berekenen geavanceerde statistieken: {e}")
            
    def _check_alerts(self, stats: TradeStats) -> None:
        """Check statistieken tegen alert thresholds."""
        try:
            if not self.config.monitor_enabled or not self.telegram_alert:
                return
                
            alerts = []
            
            # Check drawdown
            if stats.max_drawdown > self.config.alert_thresholds["max_drawdown"]:
                alerts.append(
                    f"⚠️ Hoge drawdown: {stats.max_drawdown:.1f}% "
                    f"(threshold: {self.config.alert_thresholds['max_drawdown']}%)"
                )
                
            # Check winrate
            if stats.win_rate < self.config.alert_thresholds["min_winrate"]:
                alerts.append(
                    f"⚠️ Lage winrate: {stats.win_rate:.1f}% "
                    f"(threshold: {self.config.alert_thresholds['min_winrate']}%)"
                )
                
            # Check profit factor
            if stats.profit_factor < self.config.alert_thresholds["min_profit_factor"]:
                alerts.append(
                    f"⚠️ Lage profit factor: {stats.profit_factor:.2f} "
                    f"(threshold: {self.config.alert_thresholds['min_profit_factor']})"
                )
                
            # Stuur alerts
            if alerts:
                self.telegram_alert.send_alert(
                    AlertType.WARNING,
                    "\n".join(alerts)
                )
                
        except Exception as e:
            self.log.error(f"Fout bij checken alerts: {e}")
            
    def create_charts(
        self,
        trades: List[Dict[str, Any]],
        stats: TradeStats
    ) -> Dict[str, str]:
        """Maak grafieken voor het rapport."""
        charts = {}
        chart_dir = Path(self.config.chart_dir)
        
        try:
            # Maak PNL chart
            self._create_pnl_chart(trades, chart_dir, charts)
            
            # Maak Win/Loss chart
            self._create_win_loss_chart(stats, chart_dir, charts)
            
            # Maak Equity chart
            self._create_equity_chart(trades, chart_dir, charts)
            
            # Maak Drawdown chart
            self._create_drawdown_chart(trades, chart_dir, charts)
            
            # Maak Heatmap chart
            self._create_heatmap_chart(trades, chart_dir, charts)
            
            return charts
            
        except Exception as e:
            self.log.error(f"Fout bij maken grafieken: {e}")
            return charts
            
    def _create_pnl_chart(
        self,
        trades: List[Dict[str, Any]],
        chart_dir: Path,
        charts: Dict[str, str]
    ) -> None:
        """Maak PNL chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Plot PNL per trade
            timestamps = [t["timestamp"] for t in trades]
            pnl_values = [t["pnl"] for t in trades]
            
            ax.bar(timestamps, pnl_values, color=self.config.chart_colors[0])
            ax.set_title("PNL per Trade")
            ax.set_xlabel("Datum")
            ax.set_ylabel("PNL")
            ax.grid(True)
            
            # Roteer x-as labels voor betere leesbaarheid
            plt.xticks(rotation=45)
            
            # Sla chart op
            chart_path = chart_dir / "pnl_chart.png"
            plt.savefig(chart_path, bbox_inches="tight", dpi=self.config.chart_dpi)
            plt.close()
            
            charts["pnl"] = str(chart_path)
            
        except Exception as e:
            self.log.error(f"Fout bij maken PNL chart: {e}")
            
    def _create_win_loss_chart(
        self,
        stats: TradeStats,
        chart_dir: Path,
        charts: Dict[str, str]
    ) -> None:
        """Maak Win/Loss chart."""
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Maak pie chart
            labels = ["Winst", "Verlies"]
            sizes = [stats.win_count, stats.loss_count]
            colors = [self.config.chart_colors[0], self.config.chart_colors[1]]
            
            ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")
            ax.set_title("Win/Loss Ratio")
            
            # Sla chart op
            chart_path = chart_dir / "win_loss_chart.png"
            plt.savefig(chart_path, bbox_inches="tight", dpi=self.config.chart_dpi)
            plt.close()
            
            charts["win_loss"] = str(chart_path)
            
        except Exception as e:
            self.log.error(f"Fout bij maken Win/Loss chart: {e}")
            
    def _create_equity_chart(
        self,
        trades: List[Dict[str, Any]],
        chart_dir: Path,
        charts: Dict[str, str]
    ) -> None:
        """Maak Equity chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Bereken cumulative PNL
            timestamps = [t["timestamp"] for t in trades]
            cumulative_pnl = np.cumsum([t["pnl"] for t in trades])
            
            ax.plot(timestamps, cumulative_pnl, color=self.config.chart_colors[0])
            ax.set_title("Equity Curve")
            ax.set_xlabel("Datum")
            ax.set_ylabel("Cumulatieve PNL")
            ax.grid(True)
            
            # Roteer x-as labels voor betere leesbaarheid
            plt.xticks(rotation=45)
            
            # Sla chart op
            chart_path = chart_dir / "equity_chart.png"
            plt.savefig(chart_path, bbox_inches="tight", dpi=self.config.chart_dpi)
            plt.close()
            
            charts["equity"] = str(chart_path)
            
        except Exception as e:
            self.log.error(f"Fout bij maken Equity chart: {e}")
            
    def _create_drawdown_chart(
        self,
        trades: List[Dict[str, Any]],
        chart_dir: Path,
        charts: Dict[str, str]
    ) -> None:
        """Maak Drawdown chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Bereken drawdown
            timestamps = [t["timestamp"] for t in trades]
            cumulative_pnl = np.cumsum([t["pnl"] for t in trades])
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (peak - cumulative_pnl) / peak * 100
            
            ax.plot(timestamps, drawdown, color=self.config.chart_colors[1])
            ax.set_title("Drawdown Over Tijd")
            ax.set_xlabel("Datum")
            ax.set_ylabel("Drawdown %")
            ax.grid(True)
            
            # Roteer x-as labels voor betere leesbaarheid
            plt.xticks(rotation=45)
            
            # Sla chart op
            chart_path = chart_dir / "drawdown_chart.png"
            plt.savefig(chart_path, bbox_inches="tight", dpi=self.config.chart_dpi)
            plt.close()
            
            charts["drawdown"] = str(chart_path)
            
        except Exception as e:
            self.log.error(f"Fout bij maken Drawdown chart: {e}")
            
    def _create_heatmap_chart(
        self,
        trades: List[Dict[str, Any]],
        chart_dir: Path,
        charts: Dict[str, str]
    ) -> None:
        """Maak Heatmap chart."""
        try:
            fig, ax = plt.subplots(figsize=self.config.chart_size)
            
            # Bereid data voor
            trade_times = [datetime.strptime(t["timestamp"], "%Y-%m-%d %H:%M:%S") for t in trades]
            hours = [t.hour for t in trade_times]
            days = [t.strftime("%A") for t in trade_times]
            pnl_values = [t["pnl"] for t in trades]
            
            # Maak pivot table
            df = pd.DataFrame({
                "hour": hours,
                "day": days,
                "pnl": pnl_values
            })
            pivot = df.pivot_table(
                values="pnl",
                index="day",
                columns="hour",
                aggfunc="mean"
            )
            
            # Plot heatmap
            sns.heatmap(
                pivot,
                cmap="RdYlGn",
                center=0,
                ax=ax,
                cbar_kws={"label": "Gemiddelde PNL"}
            )
            ax.set_title("PNL Heatmap per Uur en Dag")
            
            # Sla chart op
            chart_path = chart_dir / "heatmap_chart.png"
            plt.savefig(chart_path, bbox_inches="tight", dpi=self.config.chart_dpi)
            plt.close()
            
            charts["heatmap"] = str(chart_path)
            
        except Exception as e:
            self.log.error(f"Fout bij maken Heatmap chart: {e}")
            
    def create_pdf_report(
        self,
        report_type: ReportType,
        trades: List[Dict[str, Any]],
        stats: TradeStats,
        charts: Dict[str, str]
    ) -> Optional[str]:
        """Maak PDF rapport."""
        try:
            # Maak PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Voeg titel toe
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, f"Trading Rapport - {report_type.value.capitalize()}", 0, 1, "C")
            pdf.ln(10)
            
            # Voeg statistieken toe
            self._add_statistics_to_pdf(pdf, stats)
            
            # Voeg grafieken toe
            self._add_charts_to_pdf(pdf, charts)
            
            # Voeg trades toe
            self._add_trades_to_pdf(pdf, trades)
            
            # Sla rapport op
            report_path = Path(self.config.report_dir) / f"{report_type.value}_report.pdf"
            pdf.output(str(report_path))
            
            return str(report_path)
            
        except Exception as e:
            self.log.error(f"Fout bij maken PDF rapport: {e}")
            return None
            
    def _add_statistics_to_pdf(self, pdf: FPDF, stats: TradeStats) -> None:
        """Voeg statistieken toe aan PDF."""
        try:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Statistieken", 0, 1, "L")
            pdf.set_font("Arial", "", 10)
            
            # Basis statistieken
            basic_stats = [
                ("Totaal Trades", stats.total_trades),
                ("Winsttrades", stats.win_count),
                ("Verliestrades", stats.loss_count),
                ("Winrate", f"{stats.win_rate:.1f}%"),
                ("Gemiddelde PNL", f"{self.config.currency}{stats.pnl_total/stats.total_trades:.2f}"),
                ("Profit Factor", f"{stats.profit_factor:.2f}"),
                ("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}"),
                ("Max Drawdown", f"{stats.max_drawdown:.1f}%")
            ]
            
            for label, value in basic_stats:
                pdf.cell(0, 6, f"{label}: {value}", 0, 1, "L")
                
            pdf.ln(5)
            
            # Uitgebreide statistieken
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Uitgebreide Statistieken", 0, 1, "L")
            pdf.set_font("Arial", "", 10)
            
            advanced_stats = [
                ("Max Consecutive Wins", stats.max_consecutive_wins),
                ("Max Consecutive Losses", stats.max_consecutive_losses),
                ("Gemiddelde Trade Duur", f"{stats.avg_trade_duration:.1f} min"),
                ("Expectancy", f"{self.config.currency}{stats.expectancy:.2f}"),
                ("Risk/Reward Ratio", f"{stats.risk_reward_ratio:.2f}"),
                ("Kelly Criterion", f"{stats.kelly_criterion:.2f}"),
                ("VaR (95%)", f"{self.config.currency}{stats.var_95:.2f}"),
                ("CVaR (95%)", f"{self.config.currency}{stats.cvar_95:.2f}")
            ]
            
            for label, value in advanced_stats:
                pdf.cell(0, 6, f"{label}: {value}", 0, 1, "L")
                
            pdf.ln(5)
            
            # Tijd gebaseerde statistieken
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Tijd Gebaseerde Statistieken", 0, 1, "L")
            pdf.set_font("Arial", "", 10)
            
            time_stats = [
                ("Trades per Dag", f"{stats.trades_per_day:.1f}"),
                ("Gemiddelde Trades per Uur", f"{stats.avg_trades_per_hour:.1f}"),
                ("Beste Trading Uur", stats.best_trading_hour),
                ("Slechtste Trading Uur", stats.worst_trading_hour),
                ("Beste Trading Dag", stats.best_trading_day),
                ("Slechtste Trading Dag", stats.worst_trading_day)
            ]
            
            for label, value in time_stats:
                pdf.cell(0, 6, f"{label}: {value}", 0, 1, "L")
                
            pdf.ln(10)
            
        except Exception as e:
            self.log.error(f"Fout bij toevoegen statistieken aan PDF: {e}")
            
    def _add_charts_to_pdf(self, pdf: FPDF, charts: Dict[str, str]) -> None:
        """Voeg grafieken toe aan PDF."""
        try:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Grafieken", 0, 1, "L")
            
            for chart_type, chart_path in charts.items():
                if os.path.exists(chart_path):
                    pdf.image(chart_path, x=10, w=190)
                    pdf.ln(10)
                    
        except Exception as e:
            self.log.error(f"Fout bij toevoegen grafieken aan PDF: {e}")
            
    def _add_trades_to_pdf(self, pdf: FPDF, trades: List[Dict[str, Any]]) -> None:
        """Voeg trades toe aan PDF."""
        try:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Recente Trades", 0, 1, "L")
            
            # Voeg headers toe
            pdf.set_font("Arial", "B", 10)
            headers = ["Timestamp", "Symbol", "Side", "PNL"]
            col_widths = [47, 47, 47, 47]
            
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 7, header, 1, 0, "C")
            pdf.ln()
            
            # Voeg trades toe
            pdf.set_font("Arial", "", 10)
            for trade in trades[-10:]:  # Laatste 10 trades
                pdf.cell(col_widths[0], 6, trade["timestamp"], 1, 0, "C")
                pdf.cell(col_widths[1], 6, trade["symbol"], 1, 0, "C")
                pdf.cell(col_widths[2], 6, trade["side"], 1, 0, "C")
                pdf.cell(col_widths[3], 6, f"{self.config.currency}{trade['pnl']:.2f}", 1, 0, "C")
                pdf.ln()
                
            pdf.ln(10)
            
        except Exception as e:
            self.log.error(f"Fout bij toevoegen trades aan PDF: {e}")
            
    def send_telegram_report(
        self,
        report_type: ReportType,
        stats: TradeStats,
        pdf_path: Optional[str] = None
    ) -> None:
        """Stuur rapport via Telegram."""
        try:
            if not self.telegram_alert:
                return
                
            # Maak rapport tekst
            report_text = (
                f"📊 {report_type.value.title()} Rapport\n\n"
                f"Totaal Trades: {stats.total_trades}\n"
                f"Winrate: {stats.win_rate:.1f}%\n"
                f"Gemiddelde PNL: {self.config.currency}{stats.pnl_total/stats.total_trades:.2f}\n"
                f"Profit Factor: {stats.profit_factor:.2f}\n"
                f"Sharpe Ratio: {stats.sharpe_ratio:.2f}\n"
                f"Max Drawdown: {stats.max_drawdown:.1f}%\n\n"
                f"Beste Trading Uur: {stats.best_trading_hour}\n"
                f"Beste Trading Dag: {stats.best_trading_day}"
            )
            
            # Stuur rapport
            self.telegram_alert.send_alert(
                AlertType.REPORT,
                report_text
            )
            
            # Stuur PDF als bijlage
            if pdf_path and os.path.exists(pdf_path):
                self.telegram_alert.send_document(
                    pdf_path,
                    f"{report_type.value}_report.pdf"
                )
                
        except Exception as e:
            self.log.error(f"Fout bij versturen Telegram rapport: {e}")
            
    def generate_report(
        self,
        report_type: ReportType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[str]:
        """Genereer een compleet rapport."""
        try:
            # Bepaal periode
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                if report_type == ReportType.DAILY:
                    start_date = end_date - timedelta(days=1)
                elif report_type == ReportType.WEEKLY:
                    start_date = end_date - timedelta(weeks=1)
                elif report_type == ReportType.MONTHLY:
                    start_date = end_date - timedelta(days=30)
                else:
                    start_date = end_date - timedelta(hours=1)
                    
            # Lees trades
            trades, stats = self.read_trades(start_date, end_date)
            
            # Maak grafieken
            charts = self.create_charts(trades, stats)
            
            # Maak PDF rapport
            pdf_path = self.create_pdf_report(report_type, trades, stats, charts)
            
            # Stuur rapport
            if pdf_path:
                self.send_telegram_report(report_type, stats, pdf_path)
                
            return pdf_path
            
        except Exception as e:
            self.log.error(f"Fout bij genereren rapport: {e}")
            return None

# --------------------------------------------------------
# Deel 5: Main voor testing
# --------------------------------------------------------

def main():
    """Test de reporter."""
    try:
        # Maak configuraties
        report_config = ReportConfig()
        env_loader = check_environment()
        telegram_alert = TelegramAlert()
        
        # Maak reporter
        reporter = DailyReporter(
            config=report_config,
            env_loader=env_loader,
            telegram_alert=telegram_alert
        )
        
        # Genereer rapport
        pdf_path = reporter.generate_report(
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
