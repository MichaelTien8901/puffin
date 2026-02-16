## ADDED Requirements

### Requirement: LLM-powered sentiment analysis
The system SHALL analyze financial news and social media text using LLM APIs to produce sentiment scores for trading signals.

#### Scenario: News sentiment scoring
- **WHEN** the system receives a financial news article
- **THEN** it sends the text to the configured LLM API and returns a structured sentiment score (bullish/bearish/neutral) with confidence and reasoning

#### Scenario: Batch sentiment processing
- **WHEN** the user provides a batch of news articles for a ticker
- **THEN** the system processes all articles and returns an aggregated sentiment signal with time-weighted scoring

### Requirement: News-driven trading signals
The system SHALL generate trading signals based on aggregated news sentiment with configurable thresholds.

#### Scenario: Signal generation from sentiment
- **WHEN** aggregated sentiment for a ticker exceeds the bullish threshold
- **THEN** the system generates a buy signal with the sentiment score as signal strength

#### Scenario: News source integration
- **WHEN** the system is configured with news API keys
- **THEN** it fetches recent news for watched tickers from configured sources (RSS feeds, news APIs)

### Requirement: AI agent portfolio management
The system SHALL provide an AI agent that uses LLM reasoning to make portfolio allocation decisions based on market data, sentiment, and strategy signals.

#### Scenario: Portfolio rebalancing recommendation
- **WHEN** the AI agent reviews current positions, market conditions, and available signals
- **THEN** it produces a structured recommendation with target allocations and reasoning for each position change

#### Scenario: Agent decision logging
- **WHEN** the AI agent makes a recommendation
- **THEN** the system logs the full reasoning chain, inputs used, and the resulting decision for audit purposes

### Requirement: LLM provider abstraction
The system SHALL abstract LLM interactions behind a provider interface supporting Claude, OpenAI, and local models.

#### Scenario: Provider switching
- **WHEN** the user changes the LLM provider configuration from Claude to OpenAI
- **THEN** all AI-assisted features continue to work without code changes

#### Scenario: Response caching
- **WHEN** an identical prompt is sent within the configured cache TTL
- **THEN** the system returns the cached response without making an API call

### Requirement: AI-assisted market analysis reports
The system SHALL generate human-readable market analysis reports using LLMs, summarizing technical indicators, sentiment, and strategy signals.

#### Scenario: Daily market summary
- **WHEN** the user requests a market analysis for their watchlist
- **THEN** the system generates a report covering price action, key technical levels, news sentiment, and strategy signals for each ticker
