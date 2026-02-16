## ADDED Requirements

### Requirement: Position sizing
The system SHALL calculate position sizes based on configurable methods: fixed fractional, Kelly criterion, and volatility-based (ATR).

#### Scenario: Fixed fractional sizing
- **WHEN** a trade signal is generated with fixed fractional sizing at 2% risk per trade
- **THEN** the system calculates position size such that the maximum loss (to stop-loss) equals 2% of portfolio value

#### Scenario: Kelly criterion sizing
- **WHEN** a strategy has historical win rate and payoff ratio
- **THEN** the system calculates optimal position size using the Kelly formula and applies a configurable fraction (half-Kelly, quarter-Kelly)

### Requirement: Stop-loss management
The system SHALL support multiple stop-loss types: fixed percentage, trailing, ATR-based, and time-based.

#### Scenario: Trailing stop-loss
- **WHEN** a position is open with a 5% trailing stop
- **THEN** the stop-loss price ratchets up with the highest price reached and triggers when price falls 5% from the peak

#### Scenario: ATR-based stop
- **WHEN** a position is opened with ATR-based stop at 2x ATR
- **THEN** the stop-loss is placed 2 ATR units below the entry price

### Requirement: Portfolio-level risk controls
The system SHALL enforce portfolio-wide risk limits including maximum drawdown, maximum exposure, and correlation limits.

#### Scenario: Maximum drawdown circuit breaker
- **WHEN** portfolio drawdown from peak exceeds the configured maximum (e.g., 10%)
- **THEN** the system halts all new trades and alerts the user

#### Scenario: Maximum exposure limit
- **WHEN** total portfolio exposure would exceed the configured limit (e.g., 150% of equity)
- **THEN** the system rejects new position entries

### Requirement: Drawdown monitoring
The system SHALL continuously track and report current drawdown, maximum drawdown, and drawdown duration.

#### Scenario: Real-time drawdown tracking
- **WHEN** the portfolio is active
- **THEN** the system calculates and displays current drawdown percentage, max drawdown, and days since peak

### Requirement: Risk reporting
The system SHALL generate risk reports with VaR (Value at Risk), expected shortfall, and sector/asset concentration metrics.

#### Scenario: Daily risk report
- **WHEN** the user requests a risk report
- **THEN** the system computes 1-day and 5-day VaR (historical and parametric), expected shortfall, and concentration by asset
