## ADDED Requirements

### Requirement: Random Forest Models
The system SHALL implement random forest classifiers and regressors for generating trading signals from features.

#### Scenario: Train random forest classifier
- **WHEN** features and directional labels are provided
- **THEN** the system SHALL train a random forest classifier with configurable tree depth, number of estimators, and return out-of-bag scores

#### Scenario: Prevent overfitting
- **WHEN** training tree ensemble models
- **THEN** the system SHALL use cross-validation, limit tree depth, and require minimum samples per leaf to prevent overfitting

### Requirement: Gradient Boosting Models
The system SHALL support XGBoost, LightGBM, and CatBoost for training gradient boosted decision trees.

#### Scenario: Train XGBoost model
- **WHEN** a trading signal prediction task is defined
- **THEN** the system SHALL train an XGBoost model with learning rate, max depth, and early stopping parameters

#### Scenario: Optimize hyperparameters
- **WHEN** training a gradient boosting model
- **THEN** the system SHALL perform grid search or Bayesian optimization to find optimal hyperparameters

### Requirement: Feature Importance and SHAP
The system SHALL compute feature importance metrics and SHAP (SHapley Additive exPlanations) values for model interpretation.

#### Scenario: Compute SHAP values
- **WHEN** a tree ensemble model is trained
- **THEN** the system SHALL compute SHAP values for each prediction and display summary plots, dependence plots, and force plots

#### Scenario: Rank feature importance
- **WHEN** model interpretation is requested
- **THEN** the system SHALL compute and rank features by gain, split count, and SHAP importance

### Requirement: Long-Short Strategy from Boosting Signals
The system SHALL construct long-short portfolios based on gradient boosting model predictions.

#### Scenario: Generate long-short positions
- **WHEN** boosting model produces predicted returns or probabilities for a universe
- **THEN** the system SHALL rank securities, go long top quintile, short bottom quintile, and compute strategy returns

#### Scenario: Backtest boosting strategy
- **WHEN** a long-short boosting strategy is defined
- **THEN** the system SHALL perform walk-forward backtesting with proper train/test splits and report risk-adjusted performance
