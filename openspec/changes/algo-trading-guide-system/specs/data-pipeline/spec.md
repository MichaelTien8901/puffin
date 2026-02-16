## ADDED Requirements

### Requirement: Historical data retrieval
The system SHALL retrieve historical OHLCV (Open, High, Low, Close, Volume) data from yfinance with configurable date ranges and intervals.

#### Scenario: Fetch daily historical data
- **WHEN** the user requests historical data for ticker "AAPL" from 2020-01-01 to 2023-12-31
- **THEN** the system returns a DataFrame with daily OHLCV data for that period

#### Scenario: Multiple ticker support
- **WHEN** the user requests data for a list of tickers
- **THEN** the system fetches data for all tickers and returns them in a structured format

### Requirement: Real-time data streaming
The system SHALL support real-time market data streaming via Alpaca's WebSocket API.

#### Scenario: Subscribe to live quotes
- **WHEN** the user subscribes to real-time data for a set of tickers
- **THEN** the system receives and processes quote/trade updates as they arrive

#### Scenario: Connection resilience
- **WHEN** the WebSocket connection drops
- **THEN** the system automatically reconnects and resumes data streaming within 30 seconds

### Requirement: Local data caching with SQLite
The system SHALL cache retrieved market data in a local SQLite database to avoid redundant API calls.

#### Scenario: Cache hit
- **WHEN** the user requests data that already exists in the local cache
- **THEN** the system returns cached data without making an API call

#### Scenario: Cache update
- **WHEN** the user requests data with `force_refresh=True`
- **THEN** the system fetches fresh data from the API and updates the cache

### Requirement: Data provider abstraction
The system SHALL abstract data sources behind a `DataProvider` interface so providers can be swapped without changing consuming code.

#### Scenario: Provider swap
- **WHEN** the user configures a different data provider (e.g., Polygon.io instead of yfinance)
- **THEN** all data-consuming code works without modification

### Requirement: Data preprocessing and validation
The system SHALL validate and preprocess raw market data, handling missing values, stock splits, and outliers.

#### Scenario: Missing data handling
- **WHEN** market data contains gaps (weekends, holidays, missing bars)
- **THEN** the system fills or flags gaps according to the configured fill strategy (forward-fill, interpolate, or drop)

#### Scenario: Split adjustment
- **WHEN** historical data spans a stock split event
- **THEN** the system provides split-adjusted prices by default
