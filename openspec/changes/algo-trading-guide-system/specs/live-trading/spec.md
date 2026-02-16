## ADDED Requirements

### Requirement: Paper trading mode
The system SHALL support paper trading through Alpaca's paper trading API, allowing strategy testing with simulated money and real market data.

#### Scenario: Paper trade execution
- **WHEN** a strategy generates a buy signal in paper trading mode
- **THEN** the system submits the order to Alpaca's paper trading endpoint and tracks the simulated position

#### Scenario: Paper trading portfolio tracking
- **WHEN** paper trades are active
- **THEN** the system tracks positions, P&L, and portfolio value using Alpaca's paper account state

### Requirement: Live broker integration via Alpaca
The system SHALL support live trading through Alpaca's brokerage API with real order submission.

#### Scenario: Live order submission
- **WHEN** a strategy generates a signal and live trading is enabled
- **THEN** the system submits the order to Alpaca's live API and confirms execution

#### Scenario: Order status tracking
- **WHEN** an order is submitted
- **THEN** the system tracks the order lifecycle (submitted → filled/partially_filled/cancelled) via WebSocket updates

### Requirement: Broker abstraction layer
The system SHALL abstract broker interactions behind a `Broker` interface so additional brokers (e.g., IBKR) can be added.

#### Scenario: Broker interface contract
- **WHEN** a developer implements a new broker integration
- **THEN** they implement `submit_order()`, `cancel_order()`, `get_positions()`, `get_account()` methods

### Requirement: Order management
The system SHALL manage order lifecycle including submission, modification, cancellation, and fill tracking.

#### Scenario: Order cancellation
- **WHEN** the user cancels a pending order
- **THEN** the system sends a cancel request to the broker and updates order status upon confirmation

#### Scenario: Position reconciliation
- **WHEN** the system starts up
- **THEN** it reconciles local position state with the broker's reported positions

### Requirement: Trading session management
The system SHALL manage trading sessions with configurable market hours and pre/post-market trading support.

#### Scenario: Market hours enforcement
- **WHEN** a strategy generates a signal outside configured trading hours
- **THEN** the system queues the order for the next session open (or submits to extended hours if configured)

### Requirement: Safety controls
The system SHALL enforce safety controls to prevent unintended live trading actions.

#### Scenario: Live trading confirmation
- **WHEN** the user switches from paper to live trading mode
- **THEN** the system requires explicit confirmation and displays account balance and risk warnings

#### Scenario: Maximum order size limit
- **WHEN** an order exceeds the configured maximum order size
- **THEN** the system rejects the order and logs a warning
