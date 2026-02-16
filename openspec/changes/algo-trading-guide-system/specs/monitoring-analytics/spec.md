## ADDED Requirements

### Requirement: Performance dashboard
The system SHALL provide a Streamlit-based dashboard displaying portfolio performance, open positions, and recent trades.

#### Scenario: Dashboard launch
- **WHEN** the user runs the monitoring dashboard
- **THEN** a Streamlit app launches showing portfolio value, daily P&L, equity curve, and open positions

#### Scenario: Real-time updates
- **WHEN** the dashboard is open during a trading session
- **THEN** it refreshes portfolio metrics at a configurable interval (default: 30 seconds)

### Requirement: Trade logging
The system SHALL log every trade execution with full details for audit and analysis.

#### Scenario: Trade log entry
- **WHEN** a trade is executed (backtest or live)
- **THEN** the system records: timestamp, ticker, side, quantity, price, commission, slippage, strategy name, and signal metadata

#### Scenario: Trade log export
- **WHEN** the user exports the trade log
- **THEN** the system produces a CSV file with all trade records

### Requirement: P&L tracking and attribution
The system SHALL track profit and loss by strategy, asset, and time period with attribution analysis.

#### Scenario: Strategy attribution
- **WHEN** the user views P&L attribution
- **THEN** the system breaks down total P&L by contributing strategy and individual positions

#### Scenario: Period-based P&L
- **WHEN** the user selects a time period (daily, weekly, monthly)
- **THEN** the system displays P&L aggregated by that period with comparison to benchmarks

### Requirement: System health monitoring
The system SHALL monitor and report the health of system components: data feeds, broker connections, and strategy execution.

#### Scenario: Component health check
- **WHEN** the user views system status
- **THEN** the system shows connection status for data feeds, broker API, and the last heartbeat timestamp for each

#### Scenario: Error alerting
- **WHEN** a system component fails (data feed drops, broker disconnects, strategy error)
- **THEN** the system logs the error and displays a warning on the dashboard

### Requirement: Benchmark comparison
The system SHALL compare portfolio performance against configurable benchmarks (SPY, QQQ, or custom).

#### Scenario: Benchmark overlay
- **WHEN** the user views the equity curve
- **THEN** the portfolio equity curve is displayed alongside the selected benchmark with relative performance metrics (alpha, beta, information ratio)
