"""Streamlit dashboard for monitoring trading performance."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from datetime import datetime


def create_dashboard(
    pnl_tracker=None,
    trade_log=None,
    portfolio_data: Optional[Dict[str, Any]] = None
):
    """
    Create Streamlit dashboard for trading monitoring.

    Parameters
    ----------
    pnl_tracker : PnLTracker, optional
        P&L tracker instance
    trade_log : TradeLog, optional
        Trade log instance
    portfolio_data : dict, optional
        Additional portfolio data

    Notes
    -----
    This function creates a multi-page Streamlit dashboard with:
    - Portfolio overview
    - Daily P&L
    - Equity curve
    - Open positions
    - Trade log

    To run the dashboard, save this code and run:
    streamlit run dashboard.py
    """
    st.set_page_config(
        page_title="Puffin Trading Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Puffin Algorithmic Trading Dashboard")

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Select Page",
        ["Portfolio Overview", "Daily P&L", "Equity Curve", "Open Positions", "Trade Log"]
    )

    # Portfolio Overview Page
    if page == "Portfolio Overview":
        st.header("Portfolio Overview")

        col1, col2, col3, col4 = st.columns(4)

        if pnl_tracker:
            summary = pnl_tracker.performance_summary()

            with col1:
                st.metric(
                    "Total Equity",
                    f"${summary['current_equity']:,.2f}",
                    f"{summary['total_return']:.2f}%"
                )

            with col2:
                st.metric(
                    "Total P&L",
                    f"${summary['total_pnl']:,.2f}",
                    f"Realized: ${summary['realized_pnl']:,.2f}"
                )

            with col3:
                st.metric(
                    "Cash",
                    f"${summary['cash']:,.2f}",
                    f"{summary['cash'] / summary['current_equity'] * 100:.1f}%"
                )

            with col4:
                st.metric(
                    "Open Positions",
                    summary['num_positions'],
                    f"${summary['positions_value']:,.2f}"
                )

            # Strategy attribution
            st.subheader("Performance by Strategy")
            strategy_attr = pnl_tracker.attribution_by_strategy()
            if not strategy_attr.empty:
                st.dataframe(
                    strategy_attr.style.format({
                        'unrealized_pnl': '${:,.2f}',
                        'market_value': '${:,.2f}'
                    }),
                    use_container_width=True
                )

        else:
            st.info("No P&L tracker data available")

    # Daily P&L Page
    elif page == "Daily P&L":
        st.header("Daily P&L")

        if pnl_tracker and len(pnl_tracker.history) > 0:
            daily = pnl_tracker.daily_pnl()

            # Plot daily P&L
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['green' if x >= 0 else 'red' for x in daily.values]
            ax.bar(range(len(daily)), daily.values, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax.set_xlabel('Date')
            ax.set_ylabel('Daily P&L ($)')
            ax.set_title('Daily P&L')
            ax.grid(True, alpha=0.3)

            # Format x-axis dates
            if len(daily) > 0:
                tick_positions = np.linspace(0, len(daily) - 1, min(10, len(daily)), dtype=int)
                ax.set_xticks(tick_positions)
                ax.set_xticklabels([daily.index[i] for i in tick_positions], rotation=45)

            st.pyplot(fig)

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Daily P&L", f"${daily.mean():,.2f}")
            with col2:
                st.metric("Best Day", f"${daily.max():,.2f}")
            with col3:
                st.metric("Worst Day", f"${daily.min():,.2f}")
            with col4:
                win_rate = (daily > 0).sum() / len(daily) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")

        else:
            st.info("No P&L history available")

    # Equity Curve Page
    elif page == "Equity Curve":
        st.header("Equity Curve")

        if pnl_tracker and len(pnl_tracker.history) > 0:
            df = pd.DataFrame(pnl_tracker.history)

            # Plot equity curve
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['timestamp'], df['equity'], linewidth=2, label='Equity')
            ax.axhline(y=pnl_tracker.initial_cash, color='gray', linestyle='--',
                      alpha=0.5, label='Initial Capital')
            ax.set_xlabel('Date')
            ax.set_ylabel('Equity ($)')
            ax.set_title('Equity Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Drawdown plot
            st.subheader("Drawdown")
            equity_series = pd.Series(df['equity'].values, index=df['timestamp'])
            running_max = equity_series.expanding().max()
            drawdown = (equity_series - running_max) / running_max * 100

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
            ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            ax.set_title('Drawdown')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Drawdown statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Drawdown", f"{drawdown.iloc[-1]:.2f}%")
            with col2:
                st.metric("Max Drawdown", f"{drawdown.min():.2f}%")
            with col3:
                # Calculate recovery time
                max_dd_idx = drawdown.idxmin()
                recovery_idx = drawdown[drawdown.index > max_dd_idx]
                recovery_idx = recovery_idx[recovery_idx >= 0]
                if len(recovery_idx) > 0:
                    recovery_time = (recovery_idx.index[0] - max_dd_idx).days
                    st.metric("Recovery Time (days)", recovery_time)
                else:
                    st.metric("Recovery Time", "Not recovered")

        else:
            st.info("No equity history available")

    # Open Positions Page
    elif page == "Open Positions":
        st.header("Open Positions")

        if pnl_tracker and len(pnl_tracker.positions) > 0:
            asset_attr = pnl_tracker.attribution_by_asset()

            st.dataframe(
                asset_attr.style.format({
                    'quantity': '{:.2f}',
                    'avg_price': '${:.2f}',
                    'current_price': '${:.2f}',
                    'cost_basis': '${:,.2f}',
                    'market_value': '${:,.2f}',
                    'unrealized_pnl': '${:,.2f}',
                    'return_pct': '{:.2f}%'
                }).background_gradient(subset=['unrealized_pnl'], cmap='RdYlGn'),
                use_container_width=True
            )

            # Position allocation pie chart
            st.subheader("Position Allocation")
            fig, ax = plt.subplots(figsize=(8, 8))
            sizes = asset_attr['market_value'].abs()
            labels = asset_attr['ticker']
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        else:
            st.info("No open positions")

    # Trade Log Page
    elif page == "Trade Log":
        st.header("Trade Log")

        if trade_log and len(trade_log.trades) > 0:
            # Filters
            col1, col2, col3 = st.columns(3)

            with col1:
                all_tickers = list(set(t.ticker for t in trade_log.trades))
                ticker_filter = st.selectbox("Filter by Ticker", ["All"] + all_tickers)

            with col2:
                all_strategies = list(set(t.strategy for t in trade_log.trades))
                strategy_filter = st.selectbox("Filter by Strategy", ["All"] + all_strategies)

            with col3:
                side_filter = st.selectbox("Filter by Side", ["All", "buy", "sell"])

            # Apply filters
            filtered_trades = trade_log.trades
            if ticker_filter != "All":
                filtered_trades = [t for t in filtered_trades if t.ticker == ticker_filter]
            if strategy_filter != "All":
                filtered_trades = [t for t in filtered_trades if t.strategy == strategy_filter]
            if side_filter != "All":
                filtered_trades = [t for t in filtered_trades if t.side == side_filter]

            # Display trades
            if filtered_trades:
                trades_data = []
                for trade in filtered_trades:
                    trades_data.append({
                        'Timestamp': trade.timestamp,
                        'Ticker': trade.ticker,
                        'Side': trade.side,
                        'Quantity': trade.qty,
                        'Price': f"${trade.price:.2f}",
                        'Value': f"${trade.qty * trade.price:,.2f}",
                        'Commission': f"${trade.commission:.2f}",
                        'Slippage': f"${trade.slippage:.2f}",
                        'Strategy': trade.strategy
                    })

                df = pd.DataFrame(trades_data)
                st.dataframe(df, use_container_width=True)

                # Summary
                st.subheader("Summary")
                summary = trade_log.summary()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", summary['total_trades'])
                with col2:
                    st.metric("Total Commission", f"${summary['total_commission']:,.2f}")
                with col3:
                    st.metric("Total Slippage", f"${summary['total_slippage']:,.2f}")
                with col4:
                    st.metric("Avg Trade Size", f"${summary['avg_trade_size']:,.2f}")

            else:
                st.info("No trades match the selected filters")

        else:
            st.info("No trade data available")


if __name__ == "__main__":
    # This allows running the dashboard directly
    st.warning("""
    This is a template dashboard. To use it with real data:

    1. Import your PnLTracker and TradeLog instances
    2. Call create_dashboard(pnl_tracker, trade_log)
    3. Run with: streamlit run dashboard.py
    """)

    # Demo mode with sample data
    st.info("Running in demo mode with sample data")

    # Create sample data for demonstration
    demo_portfolio = {
        'equity': 105000,
        'cash': 45000,
        'positions_value': 60000,
        'total_pnl': 5000,
        'num_positions': 5
    }

    create_dashboard(portfolio_data=demo_portfolio)
