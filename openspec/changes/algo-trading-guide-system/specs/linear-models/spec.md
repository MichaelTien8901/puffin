## ADDED Requirements

### Requirement: OLS Regression with statsmodels
The system SHALL perform ordinary least squares regression using statsmodels with comprehensive diagnostics including R-squared, residual analysis, and statistical tests.

#### Scenario: Fit OLS model and generate summary
- **WHEN** dependent and independent variables are provided
- **THEN** the system SHALL fit an OLS model and return coefficients, p-values, confidence intervals, and diagnostic statistics

#### Scenario: Check regression assumptions
- **WHEN** an OLS model is fitted
- **THEN** the system SHALL test for heteroskedasticity, autocorrelation, and normality of residuals

### Requirement: Ridge and Lasso Regression
The system SHALL support L2 (Ridge) and L1 (Lasso) regularized regression for feature selection and coefficient shrinkage.

#### Scenario: Perform Lasso feature selection
- **WHEN** a high-dimensional feature set is provided
- **THEN** the system SHALL apply Lasso regression and identify non-zero coefficients for feature selection

#### Scenario: Optimize regularization parameter
- **WHEN** fitting a regularized model
- **THEN** the system SHALL use cross-validation to select optimal lambda/alpha parameter

### Requirement: Logistic Regression for Price Direction
The system SHALL implement logistic regression to predict binary price direction (up/down) or multi-class outcomes.

#### Scenario: Predict price direction
- **WHEN** features and directional labels are provided
- **THEN** the system SHALL train a logistic regression classifier and output probability estimates and classification metrics

### Requirement: Fama-French Factor Models
The system SHALL implement CAPM, Fama-French 3-factor, and 5-factor models for return attribution and risk adjustment.

#### Scenario: Run 5-factor model regression
- **WHEN** asset returns and factor returns (Mkt-RF, SMB, HML, RMW, CMA) are provided
- **THEN** the system SHALL estimate factor loadings and compute alpha with statistical significance

#### Scenario: Retrieve factor data
- **WHEN** factor model analysis is requested
- **THEN** the system SHALL download or load Fama-French factor returns from Kenneth French's data library

### Requirement: Fama-MacBeth Regression
The system SHALL perform Fama-MacBeth two-stage cross-sectional regression to estimate risk premia.

#### Scenario: Estimate cross-sectional risk premium
- **WHEN** panel data with returns and characteristics is provided
- **THEN** the system SHALL run cross-sectional regressions for each time period and average coefficients with Newey-West standard errors
