## ADDED Requirements

### Requirement: PCA for Dimensionality Reduction
The system SHALL implement Principal Component Analysis (PCA) for dimensionality reduction, eigenportfolio construction, and risk factor extraction.

#### Scenario: Perform PCA on returns
- **WHEN** a matrix of asset returns is provided
- **THEN** the system SHALL compute principal components, explained variance ratio, and component loadings

#### Scenario: Construct eigenportfolios
- **WHEN** PCA is performed on asset returns
- **THEN** the system SHALL create portfolios based on principal components and analyze their risk-return characteristics

### Requirement: Clustering Algorithms
The system SHALL support k-means, hierarchical clustering, and DBSCAN for grouping securities or identifying market regimes.

#### Scenario: Cluster securities by characteristics
- **WHEN** security features (returns, volatility, fundamentals) are provided
- **THEN** the system SHALL apply k-means clustering and assign each security to a cluster with silhouette score evaluation

#### Scenario: Detect market regimes
- **WHEN** market indicators over time are provided
- **THEN** the system SHALL use clustering to identify distinct market regimes (bull, bear, high volatility)

#### Scenario: Apply hierarchical clustering
- **WHEN** a correlation or distance matrix is available
- **THEN** the system SHALL perform hierarchical clustering and display dendrogram visualization

### Requirement: Gaussian Mixture Models
The system SHALL implement Gaussian Mixture Models (GMM) for probabilistic clustering and regime detection.

#### Scenario: Fit GMM for regime identification
- **WHEN** multivariate market data is analyzed
- **THEN** the system SHALL fit GMM, determine optimal number of components via BIC/AIC, and output regime probabilities

### Requirement: Data-Driven Risk Factors
The system SHALL extract data-driven risk factors using unsupervised learning techniques including PCA and factor analysis.

#### Scenario: Extract statistical risk factors
- **WHEN** a large panel of returns is provided
- **THEN** the system SHALL apply factor analysis or PCA to identify latent risk factors and compute factor exposures for each asset
