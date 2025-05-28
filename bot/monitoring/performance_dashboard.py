import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.risk.risk_manager import RiskManager
from bot.analysis.market_regime import MarketRegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize performance dashboard with configuration."""
        self.config = config
        self.risk_manager = RiskManager(config)
        self.market_regime_detector = MarketRegimeDetector()
        self.trades: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        
    def add_trade(self, trade: Dict[str, Any]):
        """Add a completed trade to the dashboard."""
        self.trades.append(trade)
        
    def add_daily_return(self, return_pct: float):
        """Add daily return percentage."""
        self.daily_returns.append(return_pct)
        
    def generate_dashboard(self) -> Dict[str, Any]:
        """Generate dashboard data."""
        metrics = self._calculate_metrics()
        charts = self._generate_charts()
        
        return {
            'metrics': metrics,
            'charts': charts
        }
        
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.trades:
            return {}
            
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if (total_trades - winning_trades) > 0 else 0
        
        # Risk metrics
        returns = pd.Series(self.daily_returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
    def _generate_charts(self) -> Dict[str, Any]:
        """Generate chart data."""
        if not self.trades:
            return {}
            
        # Convert trades to DataFrame
        df = pd.DataFrame(self.trades)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate cumulative PnL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Calculate drawdown
        df['drawdown'] = (df['cumulative_pnl'].expanding().max() - df['cumulative_pnl']) / df['cumulative_pnl'].expanding().max()
        
        # Generate chart data
        charts = {
            'pnl_chart': {
                'x': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'y': df['cumulative_pnl'].tolist()
            },
            'drawdown_chart': {
                'x': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'y': df['drawdown'].tolist()
            }
        }
        
        return charts
        
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        return np.sqrt(252) * returns.mean() / returns.std()
        
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) < 2:
            return 0.0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
        
    def run_dashboard(self, 
                     portfolio_data: Dict,
                     market_data: Dict[str, pd.DataFrame],
                     trade_history: List[Dict]):
        """Run the Streamlit dashboard."""
        try:
            st.set_page_config(
                page_title="Trading Bot Dashboard",
                page_icon="📈",
                layout="wide"
            )
            
            # Header
            st.title("Trading Bot Performance Dashboard")
            
            # Create columns for key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Portfolio Overview
            with col1:
                self._display_portfolio_overview(portfolio_data)
                
            # Risk Metrics
            with col2:
                self._display_risk_metrics(portfolio_data)
                
            # Trade Statistics
            with col3:
                self._display_trade_statistics(trade_history)
                
            # Market Regime
            with col4:
                self._display_market_regime(market_data)
                
            # Create tabs for detailed views
            tab1, tab2, tab3 = st.tabs([
                "Performance Analysis",
                "Trade Analysis",
                "Risk Analysis"
            ])
            
            # Performance Analysis Tab
            with tab1:
                self._display_performance_analysis(portfolio_data, trade_history)
                
            # Trade Analysis Tab
            with tab2:
                self._display_trade_analysis(trade_history)
                
            # Risk Analysis Tab
            with tab3:
                self._display_risk_analysis(portfolio_data, market_data)
                
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            st.error(f"Error running dashboard: {str(e)}")
            
    def _display_portfolio_overview(self, portfolio_data: Dict):
        """Display portfolio overview metrics."""
        try:
            st.subheader("Portfolio Overview")
            
            # Calculate metrics
            total_value = portfolio_data.get('total_value', 0)
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            daily_return = portfolio_data.get('daily_return', 0)
            
            # Display metrics
            st.metric(
                "Portfolio Value",
                f"${total_value:,.2f}",
                f"{daily_pnl:,.2f} ({daily_return:.2%})"
            )
            
            # Display position breakdown
            positions = portfolio_data.get('positions', {})
            if positions:
                st.write("Position Breakdown")
                position_df = pd.DataFrame(positions).T
                st.dataframe(position_df)
                
        except Exception as e:
            logger.error(f"Error displaying portfolio overview: {e}")
            
    def _display_risk_metrics(self, portfolio_data: Dict):
        """Display risk metrics."""
        try:
            st.subheader("Risk Metrics")
            
            # Get risk metrics
            risk_metrics = portfolio_data.get('risk_metrics', {})
            
            # Display metrics
            st.metric(
                "Current Drawdown",
                f"{risk_metrics.get('current_drawdown', 0):.2%}"
            )
            
            st.metric(
                "Sharpe Ratio",
                f"{risk_metrics.get('sharpe_ratio', 0):.2f}"
            )
            
            st.metric(
                "Value at Risk (95%)",
                f"{risk_metrics.get('var_95', 0):.2%}"
            )
            
        except Exception as e:
            logger.error(f"Error displaying risk metrics: {e}")
            
    def _display_trade_statistics(self, trade_history: List[Dict]):
        """Display trade statistics."""
        try:
            st.subheader("Trade Statistics")
            
            if not trade_history:
                st.write("No trades yet")
                return
                
            # Calculate statistics
            total_trades = len(trade_history)
            winning_trades = sum(1 for t in trade_history if t['pnl'] > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Display metrics
            st.metric("Total Trades", total_trades)
            st.metric("Win Rate", f"{win_rate:.2%}")
            
            # Calculate average trade metrics
            avg_win = np.mean([t['pnl'] for t in trade_history if t['pnl'] > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in trade_history if t['pnl'] <= 0]) if (total_trades - winning_trades) > 0 else 0
            
            st.metric("Avg Win", f"${avg_win:,.2f}")
            st.metric("Avg Loss", f"${avg_loss:,.2f}")
            
        except Exception as e:
            logger.error(f"Error displaying trade statistics: {e}")
            
    def _display_market_regime(self, market_data: Dict[str, pd.DataFrame]):
        """Display current market regime."""
        try:
            st.subheader("Market Regime")
            
            # Detect current regime
            regime = self.market_regime_detector.detect_regime(market_data)
            
            # Display regime probabilities
            for regime_name, probability in regime.items():
                st.metric(
                    regime_name.capitalize(),
                    f"{probability:.2%}"
                )
                
        except Exception as e:
            logger.error(f"Error displaying market regime: {e}")
            
    def _display_performance_analysis(self,
                                    portfolio_data: Dict,
                                    trade_history: List[Dict]):
        """Display detailed performance analysis."""
        try:
            st.subheader("Performance Analysis")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Equity Curve",
                    "Drawdown",
                    "Monthly Returns",
                    "Trade PnL Distribution"
                )
            )
            
            # Plot equity curve
            equity_curve = portfolio_data.get('equity_curve', pd.Series())
            fig.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    name="Portfolio Value"
                ),
                row=1, col=1
            )
            
            # Plot drawdown
            drawdown = portfolio_data.get('drawdown_curve', pd.Series())
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    name="Drawdown",
                    fill='tozeroy'
                ),
                row=1, col=2
            )
            
            # Plot monthly returns
            monthly_returns = portfolio_data.get('monthly_returns', pd.Series())
            fig.add_trace(
                go.Bar(
                    x=monthly_returns.index,
                    y=monthly_returns.values,
                    name="Monthly Returns"
                ),
                row=2, col=1
            )
            
            # Plot trade PnL distribution
            if trade_history:
                pnl_values = [t['pnl'] for t in trade_history]
                fig.add_trace(
                    go.Histogram(
                        x=pnl_values,
                        name="Trade PnL"
                    ),
                    row=2, col=2
                )
                
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Performance Analysis"
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error displaying performance analysis: {e}")
            
    def _display_trade_analysis(self, trade_history: List[Dict]):
        """Display detailed trade analysis."""
        try:
            st.subheader("Trade Analysis")
            
            if not trade_history:
                st.write("No trades yet")
                return
                
            # Convert trade history to DataFrame
            trades_df = pd.DataFrame(trade_history)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Trade PnL Over Time",
                    "Win/Loss Ratio by Symbol",
                    "Trade Duration Distribution",
                    "Entry/Exit Price Distribution"
                )
            )
            
            # Plot trade PnL over time
            fig.add_trace(
                go.Scatter(
                    x=trades_df['exit_time'],
                    y=trades_df['pnl'].cumsum(),
                    name="Cumulative PnL"
                ),
                row=1, col=1
            )
            
            # Plot win/loss ratio by symbol
            symbol_stats = trades_df.groupby('symbol').apply(
                lambda x: (x['pnl'] > 0).mean()
            ).reset_index()
            fig.add_trace(
                go.Bar(
                    x=symbol_stats['symbol'],
                    y=symbol_stats[0],
                    name="Win Rate"
                ),
                row=1, col=2
            )
            
            # Plot trade duration distribution
            trades_df['duration'] = (
                pd.to_datetime(trades_df['exit_time']) -
                pd.to_datetime(trades_df['entry_time'])
            ).dt.total_seconds() / 3600  # Convert to hours
            fig.add_trace(
                go.Histogram(
                    x=trades_df['duration'],
                    name="Duration"
                ),
                row=2, col=1
            )
            
            # Plot entry/exit price distribution
            fig.add_trace(
                go.Box(
                    y=trades_df['entry_price'],
                    name="Entry Price"
                ),
                row=2, col=2
            )
            fig.add_trace(
                go.Box(
                    y=trades_df['exit_price'],
                    name="Exit Price"
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Trade Analysis"
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display trade table
            st.write("Recent Trades")
            st.dataframe(trades_df.tail(10))
            
        except Exception as e:
            logger.error(f"Error displaying trade analysis: {e}")
            
    def _display_risk_analysis(self,
                              portfolio_data: Dict,
                              market_data: Dict[str, pd.DataFrame]):
        """Display detailed risk analysis."""
        try:
            st.subheader("Risk Analysis")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Portfolio Exposure",
                    "Correlation Matrix",
                    "Value at Risk",
                    "Regime Performance"
                )
            )
            
            # Plot portfolio exposure
            positions = portfolio_data.get('positions', {})
            if positions:
                exposure_data = pd.DataFrame(positions).T
                fig.add_trace(
                    go.Pie(
                        labels=exposure_data.index,
                        values=exposure_data['size'],
                        name="Exposure"
                    ),
                    row=1, col=1
                )
                
            # Plot correlation matrix
            if market_data:
                returns = pd.DataFrame({
                    symbol: data['close'].pct_change()
                    for symbol, data in market_data.items()
                })
                corr_matrix = returns.corr()
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        name="Correlation"
                    ),
                    row=1, col=2
                )
                
            # Plot Value at Risk
            var_data = portfolio_data.get('var_curve', pd.Series())
            if not var_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=var_data.index,
                        y=var_data.values,
                        name="VaR"
                    ),
                    row=2, col=1
                )
                
            # Plot regime performance
            regime_perf = portfolio_data.get('regime_performance', {})
            if regime_perf:
                fig.add_trace(
                    go.Bar(
                        x=list(regime_perf.keys()),
                        y=list(regime_perf.values()),
                        name="Regime Performance"
                    ),
                    row=2, col=2
                )
                
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Risk Analysis"
            )
            
            # Display plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Display risk metrics table
            st.write("Risk Metrics")
            risk_metrics = portfolio_data.get('risk_metrics', {})
            st.dataframe(pd.DataFrame([risk_metrics]))
            
        except Exception as e:
            logger.error(f"Error displaying risk analysis: {e}")
            
    def update_data(self,
                   portfolio_data: Dict,
                   market_data: Dict[str, pd.DataFrame],
                   trade_history: List[Dict]):
        """Update dashboard data."""
        try:
            # Clear previous data
            st.empty()
            
            # Run dashboard with new data
            self.run_dashboard(portfolio_data, market_data, trade_history)
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}") 