## ADDED Requirements

### Requirement: Event-driven backtesting loop
The system SHALL run backtests using an event-driven loop that processes bars sequentially, simulating realistic execution order.

#### Scenario: Sequential bar processing
- **WHEN** a backtest is executed on historical data
- **THEN** the engine processes each bar in chronological order, calling the strategy's signal generation before processing orders

#### Scenario: No lookahead bias
- **WHEN** the strategy generates signals for bar T
- **THEN** only data from bars 0 through T is available to the strategy

### Requirement: Order execution simulation
The system SHALL simulate order execution with configurable slippage and commission models.

#### Scenario: Market order with slippage
- **WHEN** a market order is placed
- **THEN** it executes at the next bar's open price plus/minus the configured slippage

#### Scenario: Limit order execution
- **WHEN** a limit order is placed with a price of $50
- **THEN** it executes only if the next bar's low is at or below $50 (for buys)

#### Scenario: Commission calculation
- **WHEN** an order executes
- **THEN** the configured commission (flat fee or percentage) is deducted from the portfolio

### Requirement: Performance metrics calculation
The system SHALL calculate standard performance metrics after each backtest run.

#### Scenario: Core metrics computed
- **WHEN** a backtest completes
- **THEN** the system reports: total return, annualized return, Sharpe ratio, max drawdown, win rate, profit factor, and number of trades

#### Scenario: Equity curve generation
- **WHEN** a backtest completes
- **THEN** the system produces a time-series equity curve of portfolio value over time

### Requirement: Walk-forward analysis
The system SHALL support walk-forward optimization by splitting data into in-sample and out-of-sample periods.

#### Scenario: Walk-forward split
- **WHEN** the user runs walk-forward analysis with 70/30 split
- **THEN** the system trains on 70% of data and validates on the remaining 30% across rolling windows

### Requirement: Backtest visualization
The system SHALL generate visualizations of backtest results including equity curve, drawdown chart, and trade markers.

#### Scenario: Plot generation
- **WHEN** the user calls `backtest.plot()`
- **THEN** the system displays equity curve, drawdown, and buy/sell markers overlaid on price data

### Requirement: Multi-asset backtesting
The system SHALL support backtesting strategies across multiple assets simultaneously.

#### Scenario: Portfolio backtest
- **WHEN** a strategy generates signals for 5 different tickers
- **THEN** the backtester tracks positions and P&L across all tickers with a unified portfolio equity curve
