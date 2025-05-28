import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import asyncio
from bot.core.bybit_api import BybitAPI
from bot.analysis.market_regime import MarketRegimeDetector
from bot.monitoring.performance_analytics import PerformanceAnalytics

class TradingDashboard:
    """Real-time monitoring dashboard for the trading bot."""
    
    def __init__(self, api: BybitAPI):
        self.api = api
        self.logger = logging.getLogger(__name__)
        self.market_regime_detector = MarketRegimeDetector()
        self.performance_analytics = PerformanceAnalytics()
        
    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Trading Bot Dashboard",
            page_icon="📈",
            layout="wide"
        )
        
        st.title("Trading Bot Dashboard")
        
        # Sidebar
        self._setup_sidebar()
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_portfolio_overview()
            self._display_risk_metrics()
            
        with col2:
            self._display_trade_statistics()
            self._display_market_regime()
            
        # Performance charts
        self._display_performance_analysis()
        
        # Trade analysis
        self._display_trade_analysis()
        
        # Risk analysis
        self._display_risk_analysis()
        
    def _setup_sidebar(self):
        """Setup dashboard sidebar."""
        st.sidebar.title("Settings")
        
        # Trading mode
        trading_mode = st.sidebar.selectbox(
            "Trading Mode",
            ["Paper", "Live"],
            index=0
        )
        
        # Time range
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["1h", "4h", "1d", "1w", "1m"],
            index=2
        )
        
        # Update interval
        update_interval = st.sidebar.slider(
            "Update Interval (seconds)",
            min_value=5,
            max_value=60,
            value=10
        )
        
        # Auto-refresh
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
        
        return {
            "trading_mode": trading_mode,
            "time_range": time_range,
            "update_interval": update_interval,
            "auto_refresh": auto_refresh
        }
        
    def _display_portfolio_overview(self):
        """Display portfolio overview."""
        st.subheader("Portfolio Overview")
        
        try:
            # Get portfolio data
            portfolio = asyncio.run(self.api.get_wallet_balance())
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Balance",
                    f"${portfolio['total_balance']:,.2f}",
                    f"{portfolio['daily_pnl']:,.2f}"
                )
                
            with col2:
                st.metric(
                    "Available Margin",
                    f"${portfolio['available_margin']:,.2f}",
                    f"{portfolio['margin_ratio']*100:.1f}%"
                )
                
            with col3:
                st.metric(
                    "Unrealized PnL",
                    f"${portfolio['unrealized_pnl']:,.2f}",
                    f"{portfolio['unrealized_pnl_percentage']*100:.1f}%"
                )
                
            # Position breakdown
            st.subheader("Position Breakdown")
            positions = asyncio.run(self.api.get_positions())
            
            if positions:
                df = pd.DataFrame(positions)
                st.dataframe(df)
            else:
                st.info("No open positions")
                
        except Exception as e:
            self.logger.error(f"Error displaying portfolio overview: {e}")
            st.error("Error loading portfolio data")
            
    def _display_risk_metrics(self):
        """Display risk metrics."""
        st.subheader("Risk Metrics")
        
        try:
            # Get risk metrics
            metrics = self.performance_analytics.get_risk_metrics()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Drawdown",
                    f"{metrics['current_drawdown']*100:.1f}%",
                    f"Max: {metrics['max_drawdown']*100:.1f}%"
                )
                
            with col2:
                st.metric(
                    "Sharpe Ratio",
                    f"{metrics['sharpe_ratio']:.2f}",
                    f"Target: 2.0"
                )
                
            with col3:
                st.metric(
                    "Value at Risk (95%)",
                    f"${metrics['var_95']:,.2f}",
                    f"{metrics['var_95_percentage']*100:.1f}%"
                )
                
            # Risk limits
            st.subheader("Risk Limits")
            limits = {
                "Max Position Size": "10%",
                "Max Daily Loss": "5%",
                "Max Drawdown": "20%",
                "Min Margin Ratio": "50%"
            }
            
            for limit, value in limits.items():
                st.text(f"{limit}: {value}")
                
        except Exception as e:
            self.logger.error(f"Error displaying risk metrics: {e}")
            st.error("Error loading risk metrics")
            
    def _display_trade_statistics(self):
        """Display trade statistics."""
        st.subheader("Trade Statistics")
        
        try:
            # Get trade statistics
            stats = self.performance_analytics.get_trade_statistics()
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Trades",
                    stats['total_trades'],
                    f"Win Rate: {stats['win_rate']*100:.1f}%"
                )
                
            with col2:
                st.metric(
                    "Average Win",
                    f"${stats['avg_win']:,.2f}",
                    f"Best: ${stats['max_win']:,.2f}"
                )
                
            with col3:
                st.metric(
                    "Average Loss",
                    f"${stats['avg_loss']:,.2f}",
                    f"Worst: ${stats['max_loss']:,.2f}"
                )
                
            # Trade distribution
            st.subheader("Trade Distribution")
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[stats['winning_trades'], stats['losing_trades']],
                    hole=.3
                )
            ])
            st.plotly_chart(fig)
            
        except Exception as e:
            self.logger.error(f"Error displaying trade statistics: {e}")
            st.error("Error loading trade statistics")
            
    def _display_market_regime(self):
        """Display market regime analysis."""
        st.subheader("Market Regime")
        
        try:
            # Get market data
            market_data = asyncio.run(self.api.get_klines(
                symbol="BTCUSDT",
                interval="1h",
                limit=100
            ))
            
            # Detect regime
            regime = self.market_regime_detector.detect_regime(market_data)
            
            # Display regime probabilities
            st.write("Current Market Regime Probabilities:")
            
            for regime_type, probability in regime.items():
                st.progress(probability)
                st.text(f"{regime_type}: {probability*100:.1f}%")
                
            # Regime history
            st.subheader("Regime History")
            history = self.market_regime_detector.get_regime_history()
            
            if history:
                df = pd.DataFrame(history)
                st.line_chart(df)
                
        except Exception as e:
            self.logger.error(f"Error displaying market regime: {e}")
            st.error("Error loading market regime data")
            
    def _display_performance_analysis(self):
        """Display performance analysis charts."""
        st.subheader("Performance Analysis")
        
        try:
            # Get performance data
            performance = self.performance_analytics.get_performance_metrics()
            
            # Create subplots
            fig = go.Figure()
            
            # Equity curve
            fig.add_trace(go.Scatter(
                x=performance['dates'],
                y=performance['equity'],
                name="Equity"
            ))
            
            # Drawdown
            fig.add_trace(go.Scatter(
                x=performance['dates'],
                y=performance['drawdown'],
                name="Drawdown",
                fill='tozeroy'
            ))
            
            # Monthly returns
            fig.add_trace(go.Bar(
                x=performance['months'],
                y=performance['monthly_returns'],
                name="Monthly Returns"
            ))
            
            # Trade PnL distribution
            fig.add_trace(go.Histogram(
                x=performance['trade_pnl'],
                name="Trade PnL Distribution"
            ))
            
            st.plotly_chart(fig)
            
        except Exception as e:
            self.logger.error(f"Error displaying performance analysis: {e}")
            st.error("Error loading performance data")
            
    def _display_trade_analysis(self):
        """Display detailed trade analysis."""
        st.subheader("Trade Analysis")
        
        try:
            # Get trade analysis
            analysis = self.performance_analytics.get_trade_analysis()
            
            # Trade performance over time
            st.write("Trade Performance Over Time")
            fig = go.Figure(data=[
                go.Scatter(
                    x=analysis['trade_dates'],
                    y=analysis['cumulative_pnl'],
                    name="Cumulative PnL"
                )
            ])
            st.plotly_chart(fig)
            
            # Win/Loss by symbol
            st.write("Win/Loss by Symbol")
            fig = go.Figure(data=[
                go.Bar(
                    x=analysis['symbols'],
                    y=analysis['win_loss_ratio'],
                    name="Win/Loss Ratio"
                )
            ])
            st.plotly_chart(fig)
            
            # Trade duration analysis
            st.write("Trade Duration Analysis")
            fig = go.Figure(data=[
                go.Histogram(
                    x=analysis['trade_durations'],
                    name="Trade Duration"
                )
            ])
            st.plotly_chart(fig)
            
            # Entry/Exit price distribution
            st.write("Entry/Exit Price Distribution")
            fig = go.Figure(data=[
                go.Scatter(
                    x=analysis['entry_prices'],
                    y=analysis['exit_prices'],
                    mode='markers',
                    name="Entry vs Exit"
                )
            ])
            st.plotly_chart(fig)
            
        except Exception as e:
            self.logger.error(f"Error displaying trade analysis: {e}")
            st.error("Error loading trade analysis data")
            
    def _display_risk_analysis(self):
        """Display risk analysis."""
        st.subheader("Risk Analysis")
        
        try:
            # Get risk analysis
            analysis = self.performance_analytics.get_risk_analysis()
            
            # Portfolio exposure
            st.write("Portfolio Exposure")
            fig = go.Figure(data=[
                go.Pie(
                    labels=analysis['exposure_symbols'],
                    values=analysis['exposure_values'],
                    name="Exposure"
                )
            ])
            st.plotly_chart(fig)
            
            # Correlation matrix
            st.write("Correlation Matrix")
            fig = go.Figure(data=[
                go.Heatmap(
                    z=analysis['correlation_matrix'],
                    x=analysis['symbols'],
                    y=analysis['symbols'],
                    name="Correlation"
                )
            ])
            st.plotly_chart(fig)
            
            # Value at Risk analysis
            st.write("Value at Risk Analysis")
            fig = go.Figure(data=[
                go.Scatter(
                    x=analysis['var_levels'],
                    y=analysis['var_values'],
                    name="VaR"
                )
            ])
            st.plotly_chart(fig)
            
            # Regime performance
            st.write("Performance by Market Regime")
            fig = go.Figure(data=[
                go.Bar(
                    x=analysis['regimes'],
                    y=analysis['regime_performance'],
                    name="Regime Performance"
                )
            ])
            st.plotly_chart(fig)
            
        except Exception as e:
            self.logger.error(f"Error displaying risk analysis: {e}")
            st.error("Error loading risk analysis data")
            
    def update_data(self):
        """Update dashboard data."""
        try:
            # Refresh all data
            self.performance_analytics.update_data()
            
            # Clear cache
            st.cache_data.clear()
            
            # Rerun dashboard
            st.experimental_rerun()
            
        except Exception as e:
            self.logger.error(f"Error updating dashboard data: {e}")
            st.error("Error updating dashboard data") 