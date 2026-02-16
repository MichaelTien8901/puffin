## ADDED Requirements

### Requirement: Mean-Variance Optimization
The system SHALL implement Markowitz mean-variance optimization to construct efficient portfolios given expected returns and covariance matrix.

#### Scenario: Compute efficient frontier
- **WHEN** expected returns and covariance estimates are provided for a universe of assets
- **THEN** the system SHALL compute the efficient frontier and identify the maximum Sharpe ratio portfolio

#### Scenario: Apply portfolio constraints
- **WHEN** optimization is performed with constraints (long-only, leverage limits, position limits)
- **THEN** the system SHALL respect all constraints and return a feasible optimal portfolio

### Requirement: Risk Parity Portfolio
The system SHALL construct risk parity portfolios that allocate risk equally across assets or factors.

#### Scenario: Equal risk contribution portfolio
- **WHEN** a covariance matrix is provided
- **THEN** the system SHALL compute portfolio weights such that each asset contributes equally to portfolio variance

### Requirement: Hierarchical Risk Parity
The system SHALL implement hierarchical risk parity (HRP) using graph-based clustering and recursive bisection.

#### Scenario: Build HRP portfolio
- **WHEN** a correlation matrix is provided
- **THEN** the system SHALL perform hierarchical clustering, construct a dendrogram, and allocate weights via recursive bisection

### Requirement: pyfolio Integration
The system SHALL integrate pyfolio to generate comprehensive performance tearsheets including returns analysis, risk metrics, and drawdown visualization.

#### Scenario: Generate performance tearsheet
- **WHEN** a portfolio returns time series is provided
- **THEN** the system SHALL produce a tearsheet with cumulative returns, rolling metrics, factor exposures, and underwater plot

### Requirement: Portfolio Rebalancing
The system MUST support periodic and threshold-based portfolio rebalancing with transaction cost awareness.

#### Scenario: Rebalance on schedule
- **WHEN** the rebalancing period expires
- **THEN** the system SHALL compute target weights, generate trades, and estimate transaction costs

#### Scenario: Rebalance on drift threshold
- **WHEN** portfolio weights drift beyond specified thresholds from targets
- **THEN** the system SHALL trigger rebalancing to restore target allocations
