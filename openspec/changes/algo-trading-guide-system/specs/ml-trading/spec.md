## ADDED Requirements

### Requirement: Feature engineering pipeline
The system SHALL provide a feature engineering module that computes technical indicators and statistical features from market data for ML model input.

#### Scenario: Technical indicator features
- **WHEN** the user runs the feature pipeline on OHLCV data
- **THEN** the system computes configurable features including RSI, MACD, Bollinger Bands, ATR, volume profiles, and lagged returns

#### Scenario: Custom feature registration
- **WHEN** the user defines a custom feature function
- **THEN** they can register it with the pipeline and it is included in the feature matrix

### Requirement: Supervised model training for signal prediction
The system SHALL support training supervised ML models (classification and regression) to predict trading signals or price direction.

#### Scenario: Classification model training
- **WHEN** the user trains a classifier to predict next-day price direction
- **THEN** the system trains using scikit-learn, performs cross-validation, and reports accuracy, precision, recall, and F1 score

#### Scenario: Regression model training
- **WHEN** the user trains a regression model to predict next-day returns
- **THEN** the system trains the model and reports RMSE, MAE, and R-squared metrics

### Requirement: Reinforcement learning trading agent
The system SHALL provide an RL environment compatible with Gymnasium for training trading agents using PyTorch.

#### Scenario: Trading environment setup
- **WHEN** the user creates a trading environment with historical data
- **THEN** the environment exposes standard Gymnasium interface (reset, step, observation_space, action_space) with actions: buy, sell, hold

#### Scenario: Agent training
- **WHEN** the user trains a DQN or PPO agent on the trading environment
- **THEN** the agent learns a policy and the training reports episode rewards, cumulative P&L, and Sharpe ratio over training

### Requirement: Model evaluation with walk-forward validation
The system SHALL evaluate ML models using time-series aware validation to prevent data leakage.

#### Scenario: Time-series cross-validation
- **WHEN** the user evaluates a model
- **THEN** the system uses expanding or sliding window cross-validation (never shuffled k-fold) and reports per-fold metrics

### Requirement: Model persistence and versioning
The system SHALL save trained models with metadata for reproducibility and comparison.

#### Scenario: Model save and load
- **WHEN** the user saves a trained model
- **THEN** the system persists the model weights, hyperparameters, training data hash, and performance metrics

#### Scenario: Model comparison
- **WHEN** the user lists saved models
- **THEN** the system displays a comparison table of model versions with their metrics
