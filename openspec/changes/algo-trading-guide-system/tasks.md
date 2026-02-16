## 1. Project Setup & Infrastructure

- [x] 1.1 Initialize Python project with `pyproject.toml` (project name: puffin, Python 3.11+, dependency groups for core, ml, ai, dev)
- [x] 1.2 Create monorepo directory structure: `puffin/`, `docs/`, `tests/`, `notebooks/`
- [x] 1.3 Create `puffin/` package with subpackages: `data`, `strategies`, `backtest`, `ml`, `ai`, `broker`, `risk`, `monitor`
- [x] 1.4 Set up Jekyll + Just the Docs in `docs/` directory: `_config.yml`, `Gemfile`, `Dockerfile`, `docker-compose.yml` (modeled after rust-guide-tutorial)
- [x] 1.5 Create docs folder structure with `index.md` landing page
- [x] 1.6 Create Dockerfile for Python development environment
- [x] 1.7 Add `.env.example` with placeholders for Alpaca, LLM API keys
- [x] 1.8 Set up pytest configuration and test directory structure mirroring `puffin/` subpackages
- [x] 1.9 Create expanded subpackages: `factors`, `portfolio`, `models`, `ensembles`, `unsupervised`, `nlp`, `deep`, `rl`
- [x] 1.10 Update `pyproject.toml` with new dependencies: statsmodels, pymc, xgboost, lightgbm, catboost, gensim, spacy, shap, transformers, ta-lib, alphalens-reloaded, pyfolio-reloaded, arviz

## 2. Tutorial Site (tutorial-site)

- [x] 2.1 Configure Just the Docs navigation with part sections, callouts (note, tip, warning, important), and Mermaid diagram support
- [x] 2.2 Create `docs/index.md` landing page with project overview, learning path diagram, and prerequisites
- [x] 2.3 Configure Rouge syntax highlighting for Python code blocks
- [x] 2.4 Verify Just the Docs built-in search works across all tutorial content
- [x] 2.5 Verify next/previous chapter navigation via Just the Docs defaults
- [x] 2.6 Create tutorial page template (standard chapter layout: theory → code → exercises) with front matter conventions
- [x] 2.7 Update landing page with expanded 25-part learning path diagram
- [x] 2.8 Create index pages for all 25 tutorial parts

## 3. Market Foundations Content (market-foundations)

- [x] 3.1 Write chapter: Market Structure — exchanges, order books, bid-ask spreads, market participants
- [x] 3.2 Write chapter: Asset Classes — equities, ETFs, crypto comparison (trading hours, settlement, fees, volatility)
- [x] 3.3 Write chapter: Trading Mechanics — order types (market, limit, stop, stop-limit), execution, slippage, commissions
- [x] 3.4 Write chapter: Financial Metrics — returns, volatility, Sharpe ratio, drawdown, alpha, beta with Python code examples
- [x] 3.5 Write chapter: Historical Context — case studies of 3+ significant market events and their algorithmic trading implications
- [x] 3.6 Create companion Jupyter notebook with interactive metric calculations
- [x] 3.7 Write chapter: Market Microstructure — ITCH data format, FIX protocol, tick-to-bar conversion (ref: book Ch 2)

## 4. Data Pipeline (data-pipeline)

- [x] 4.1 Implement `DataProvider` abstract interface with `fetch_historical()`, `stream_realtime()`, `get_supported_assets()` methods
- [x] 4.2 Implement `YFinanceProvider` — historical OHLCV retrieval with configurable date range, interval, and multi-ticker support
- [x] 4.3 Implement `AlpacaProvider` — historical data and WebSocket real-time streaming with auto-reconnect
- [x] 4.4 Implement SQLite data cache with cache-hit logic and `force_refresh` support
- [x] 4.5 Implement data preprocessing: missing value handling (forward-fill, interpolate, drop), split adjustment, outlier detection
- [x] 4.6 Write tests for data providers, cache, and preprocessing
- [x] 4.7 Write tutorial chapter: Data Pipeline — fetching, caching, and cleaning market data
- [x] 4.8 Implement fundamental data retrieval (financial statements, SEC EDGAR) (ref: book Ch 2)
- [x] 4.9 Implement efficient data storage with HDF5/Parquet formats (ref: book Ch 2)

## 5. Alternative Data (alternative-data)

- [x] 5.1 Implement web scraper for financial data sources (earnings transcripts, restaurant data) (ref: book Ch 3)
- [x] 5.2 Implement earnings call transcript parser with NLP preprocessing
- [x] 5.3 Implement alternative data evaluation framework (signal quality, data quality, technical criteria)
- [x] 5.4 Write tests for web scraping and transcript parsing
- [x] 5.5 Write tutorial chapter: Alternative Data — sourcing, evaluating, and processing non-traditional data

## 6. Alpha Factors (alpha-factors)

- [x] 6.1 Implement alpha factor computation: momentum, value, volatility, quality factors (ref: book Ch 4)
- [x] 6.2 Integrate TA-Lib for comprehensive technical indicator library (overlap, momentum, volume, volatility)
- [x] 6.3 Implement Kalman filter for alpha signal denoising
- [x] 6.4 Implement wavelet-based signal preprocessing
- [x] 6.5 Implement Alphalens factor evaluation: IC, factor returns, turnover analysis
- [x] 6.6 Implement WorldQuant-style formulaic alpha expressions
- [x] 6.7 Write tests for factor computation and evaluation
- [x] 6.8 Write tutorial chapter: Alpha Factor Research — engineering and evaluating predictive signals

## 7. Portfolio Optimization (portfolio-optimization)

- [x] 7.1 Implement mean-variance optimization (Markowitz efficient frontier) (ref: book Ch 5)
- [x] 7.2 Implement risk parity portfolio construction
- [x] 7.3 Implement hierarchical risk parity (HRP) using clustering
- [x] 7.4 Integrate pyfolio for performance tearsheets (returns, drawdown, position analysis)
- [x] 7.5 Implement portfolio rebalancing with transaction cost awareness
- [x] 7.6 Write tests for portfolio optimization methods
- [x] 7.7 Write tutorial chapter: Portfolio Optimization — from Markowitz to HRP

## 8. Strategy Library (strategy-library)

- [x] 8.1 Define `Strategy` base class with `generate_signals(data) -> SignalFrame` and `get_parameters() -> dict` interface
- [x] 8.2 Implement `MomentumStrategy` — SMA/EMA crossover with configurable lookback periods and signal thresholds
- [x] 8.3 Implement `MeanReversionStrategy` — Bollinger Bands + z-score based entry/exit signals
- [x] 8.4 Implement `StatArbStrategy` — pair screening with cointegration tests (Engle-Granger), spread-based signal generation
- [x] 8.5 Implement `MarketMakingStrategy` — symmetric bid/ask placement around mid-price with configurable spread
- [x] 8.6 Create strategy registry for dynamic strategy discovery and instantiation
- [x] 8.7 Write tests for all strategies with known-output test cases
- [x] 8.8 Write tutorial chapter: Trading Strategies — theory and implementation for each strategy

## 9. Backtesting Engine (backtesting-engine)

- [x] 9.1 Implement event-driven backtesting loop with sequential bar processing and lookahead bias prevention
- [x] 9.2 Implement order execution simulator with slippage models (fixed, percentage) and commission models (flat, percentage)
- [x] 9.3 Implement limit order and stop order execution logic
- [x] 9.4 Implement performance metrics: total return, annualized return, Sharpe, max drawdown, win rate, profit factor, trade count
- [x] 9.5 Implement equity curve generation and multi-asset portfolio tracking
- [x] 9.6 Implement walk-forward analysis with configurable in-sample/out-of-sample splits
- [x] 9.7 Implement backtest visualization: equity curve, drawdown chart, trade markers on price chart
- [x] 9.8 Write tests for backtester with deterministic test cases
- [x] 9.9 Write tutorial chapter: Backtesting — building and using the backtesting engine

## 10. Linear Models (linear-models)

- [x] 10.1 Implement OLS regression with statsmodels for return prediction (ref: book Ch 7)
- [x] 10.2 Implement ridge and lasso regression with scikit-learn
- [x] 10.3 Implement logistic regression for price direction classification
- [x] 10.4 Implement Fama-French factor models (CAPM, 3-factor, 5-factor) with data retrieval
- [x] 10.5 Implement Fama-MacBeth cross-sectional regression
- [x] 10.6 Write tests for all linear models with known outputs
- [x] 10.7 Write tutorial chapter: Linear Models — from risk factors to return forecasts

## 11. Time Series Models (time-series-models)

- [x] 11.1 Implement time series diagnostics: decomposition, stationarity tests (ADF, KPSS), autocorrelation (ref: book Ch 9)
- [x] 11.2 Implement ARIMA model fitting and forecasting with statsmodels
- [x] 11.3 Implement VAR (Vector Autoregression) for multivariate macro forecasting
- [x] 11.4 Implement GARCH volatility forecasting (GARCH, EGARCH, GJR-GARCH)
- [x] 11.5 Implement cointegration tests: Engle-Granger two-step and Johansen likelihood-ratio
- [x] 11.6 Implement time-series-based pairs trading strategy
- [x] 11.7 Write tests for time series models with synthetic data
- [x] 11.8 Write tutorial chapter: Time Series Models — volatility forecasting and statistical arbitrage

## 12. Bayesian ML (bayesian-ml)

- [x] 12.1 Implement Bayesian model framework using PyMC (ref: book Ch 10)
- [x] 12.2 Implement Bayesian Sharpe ratio for strategy comparison
- [x] 12.3 Implement Bayesian rolling regression for pairs trading
- [x] 12.4 Implement stochastic volatility model
- [x] 12.5 Write tests for Bayesian models (with fixed seeds for reproducibility)
- [x] 12.6 Write tutorial chapter: Bayesian ML — dynamic Sharpe ratios and pairs trading

## 13. Tree Ensembles (tree-ensembles)

- [x] 13.1 Implement random forest classifier/regressor for trading signal generation (ref: book Ch 11-12)
- [x] 13.2 Implement XGBoost model training with hyperparameter tuning
- [x] 13.3 Implement LightGBM model training with GPU support
- [x] 13.4 Implement CatBoost model training with categorical feature handling
- [x] 13.5 Implement SHAP-based model interpretation and feature importance
- [x] 13.6 Implement long-short strategy using boosting ensemble signals
- [x] 13.7 Write tests for all ensemble models
- [x] 13.8 Write tutorial chapter: Tree Ensembles — random forests and gradient boosting for trading

## 14. Unsupervised Learning (unsupervised-learning)

- [x] 14.1 Implement PCA for dimensionality reduction and eigenportfolios (ref: book Ch 13)
- [x] 14.2 Implement k-means clustering for asset grouping
- [x] 14.3 Implement hierarchical clustering with dendrogram visualization
- [x] 14.4 Implement DBSCAN and Gaussian mixture models
- [x] 14.5 Implement data-driven risk factor extraction
- [x] 14.6 Write tests for unsupervised learning methods
- [x] 14.7 Write tutorial chapter: Unsupervised Learning — eigenportfolios and clustering for asset allocation

## 15. NLP for Trading (nlp-trading)

- [x] 15.1 Implement NLP pipeline with spaCy: tokenization, lemmatization, NER for financial text (ref: book Ch 14)
- [x] 15.2 Implement bag-of-words and TF-IDF document-term matrix with scikit-learn
- [x] 15.3 Implement naive Bayes classifier for financial news classification
- [x] 15.4 Implement sentiment analysis pipeline for news and social media text
- [x] 15.5 Write tests for NLP pipeline and classifiers
- [x] 15.6 Write tutorial chapter: NLP for Trading — from text to trading signals

## 16. Topic Modeling (topic-modeling)

- [x] 16.1 Implement LSI (Latent Semantic Indexing) with scikit-learn (ref: book Ch 15)
- [x] 16.2 Implement LDA (Latent Dirichlet Allocation) with Gensim
- [x] 16.3 Implement topic visualization with pyLDAvis
- [x] 16.4 Implement earnings call topic analysis pipeline
- [x] 16.5 Write tests for topic models
- [x] 16.6 Write tutorial chapter: Topic Modeling — summarizing financial news and earnings calls

## 17. Word Embeddings (word-embeddings)

- [x] 17.1 Implement word2vec training (skip-gram, CBOW) with Gensim (ref: book Ch 16)
- [x] 17.2 Implement GloVe pretrained vector loading and usage
- [x] 17.3 Implement doc2vec for document-level embeddings
- [x] 17.4 Implement SEC filing analysis with word embeddings
- [x] 17.5 Implement BERT/transformer-based embeddings using HuggingFace
- [x] 17.6 Write tests for embedding models
- [x] 17.7 Write tutorial chapter: Word Embeddings — semantic analysis of financial text

## 18. Deep Learning Fundamentals (deep-learning)

- [x] 18.1 Implement feedforward neural network for return prediction with PyTorch (ref: book Ch 17)
- [x] 18.2 Implement same architecture with TensorFlow 2 for comparison
- [x] 18.3 Implement training utilities: early stopping, learning rate scheduling, dropout, batch normalization
- [x] 18.4 Implement TensorBoard integration for training visualization
- [x] 18.5 Write tests for neural network training
- [x] 18.6 Write tutorial chapter: Deep Learning Fundamentals — building neural networks for trading

## 19. CNNs for Trading (deep-learning)

- [x] 19.1 Implement 1D CNN for autoregressive time-series prediction (ref: book Ch 18)
- [x] 19.2 Implement CNN-TA: 2D image representation of time series for CNN classification
- [x] 19.3 Implement transfer learning pipeline (pretrained ImageNet models for financial data)
- [x] 19.4 Write tests for CNN architectures
- [x] 19.5 Write tutorial chapter: CNNs for Trading — convolutional approaches to financial time series

## 20. RNNs for Trading (deep-learning)

- [x] 20.1 Implement LSTM for univariate time-series prediction (ref: book Ch 19)
- [x] 20.2 Implement stacked LSTM for multivariate time-series regression
- [x] 20.3 Implement GRU as LSTM alternative
- [x] 20.4 Implement LSTM with pretrained word embeddings for sentiment classification
- [x] 20.5 Write tests for RNN architectures
- [x] 20.6 Write tutorial chapter: RNNs for Trading — sequence models for time series and text

## 21. Autoencoders (autoencoders)

- [x] 21.1 Implement standard feedforward autoencoder for feature extraction (ref: book Ch 20)
- [x] 21.2 Implement denoising autoencoder for corrupted data recovery
- [x] 21.3 Implement variational autoencoder (VAE) for generative modeling
- [x] 21.4 Implement conditional autoencoder for asset pricing and risk factors
- [x] 21.5 Write tests for autoencoder architectures
- [x] 21.6 Write tutorial chapter: Autoencoders — nonlinear feature extraction for trading

## 22. Synthetic Data with GANs (synthetic-data)

- [x] 22.1 Implement basic GAN architecture (generator + discriminator) with TensorFlow 2 (ref: book Ch 21)
- [x] 22.2 Implement TimeGAN for synthetic financial time-series generation
- [x] 22.3 Implement synthetic data quality evaluation (distribution matching, autocorrelation, PCA comparison)
- [x] 22.4 Implement data augmentation pipeline for expanding limited training data
- [x] 22.5 Write tests for GAN training and quality evaluation
- [x] 22.6 Write tutorial chapter: Synthetic Data — GANs for financial time-series generation

## 23. Deep Reinforcement Learning (deep-rl)

- [x] 23.1 Implement tabular Q-learning for simple trading environment (ref: book Ch 22)
- [x] 23.2 Implement DQN and DDQN with experience replay and target networks
- [x] 23.3 Implement custom Gymnasium trading environment with realistic features
- [x] 23.4 Implement PPO agent using stable-baselines3
- [x] 23.5 Implement RL agent evaluation: episode rewards, cumulative P&L, Sharpe tracking
- [x] 23.6 Write tests for RL environments and agents
- [x] 23.7 Write tutorial chapter: Deep RL — building a trading agent from scratch

## 24. ML Trading Utilities (ml-trading)

- [x] 24.1 Implement feature engineering pipeline: technical indicators (RSI, MACD, Bollinger, ATR), lagged returns, volume features
- [x] 24.2 Implement custom feature registration system
- [x] 24.3 Implement supervised model training wrapper for scikit-learn (classification + regression) with time-series cross-validation
- [x] 24.4 Implement model evaluation with expanding/sliding window validation and per-fold metric reporting
- [x] 24.5 Implement Gymnasium-compatible trading environment (observation: features, actions: buy/sell/hold, reward: P&L or Sharpe)
- [x] 24.6 Implement DQN and PPO agent training with PyTorch and episode reward tracking
- [x] 24.7 Implement model persistence: save/load with metadata (hyperparameters, data hash, metrics) and comparison table
- [x] 24.8 Write tests for feature pipeline, model training, and RL environment
- [x] 24.9 Write tutorial chapter: ML Trading — feature engineering, supervised models, and reinforcement learning

## 25. AI-Assisted Trading (ai-assisted-trading)

- [x] 25.1 Implement `LLMProvider` abstract interface with `analyze()`, `generate()` methods and response caching
- [x] 25.2 Implement `ClaudeProvider` and `OpenAIProvider` concrete implementations
- [x] 25.3 Implement sentiment analysis module: single article scoring and batch aggregation with time-weighted signals
- [x] 25.4 Implement news-driven signal generator with configurable bullish/bearish thresholds
- [x] 25.5 Implement news source integration (RSS feeds, configurable news APIs)
- [x] 25.6 Implement AI agent portfolio manager: structured recommendations with reasoning chains and decision logging
- [x] 25.7 Implement market analysis report generator (daily summary of technicals, sentiment, signals)
- [x] 25.8 Write tests for LLM providers (with mocked responses), sentiment analysis, and signal generation
- [x] 25.9 Write tutorial chapter: AI-Assisted Trading — sentiment analysis, news signals, and AI agents

## 26. Live Trading (live-trading)

- [x] 26.1 Implement `Broker` abstract interface: `submit_order()`, `cancel_order()`, `get_positions()`, `get_account()`
- [x] 26.2 Implement `AlpacaBroker` for paper and live trading with order lifecycle tracking via WebSocket
- [x] 26.3 Implement order management: submission, modification, cancellation, fill tracking, position reconciliation on startup
- [x] 26.4 Implement trading session management with market hours enforcement and extended hours support
- [x] 26.5 Implement safety controls: live trading confirmation prompt, maximum order size limits
- [x] 26.6 Write tests for broker interface, order management (with mocked API), and safety controls
- [x] 26.7 Write tutorial chapter: Live Trading — paper trading setup, going live, and safety controls

## 27. Risk Management (risk-management)

- [x] 27.1 Implement position sizing: fixed fractional, Kelly criterion (with fractional Kelly), volatility-based (ATR)
- [x] 27.2 Implement stop-loss types: fixed percentage, trailing, ATR-based, time-based
- [x] 27.3 Implement portfolio-level risk controls: max drawdown circuit breaker, max exposure limit
- [x] 27.4 Implement drawdown monitoring: real-time tracking of current drawdown, max drawdown, drawdown duration
- [x] 27.5 Implement risk reporting: VaR (historical + parametric), expected shortfall, concentration metrics
- [x] 27.6 Write tests for position sizing, stop-loss logic, and risk controls
- [x] 27.7 Write tutorial chapter: Risk Management — position sizing, stop losses, and portfolio risk

## 28. Monitoring & Analytics (monitoring-analytics)

- [x] 28.1 Implement trade logging: structured trade records (timestamp, ticker, side, qty, price, commission, slippage, strategy, metadata) with CSV export
- [x] 28.2 Implement P&L tracking and attribution by strategy, asset, and time period
- [x] 28.3 Implement benchmark comparison: equity curve overlay with alpha, beta, information ratio
- [x] 28.4 Build Streamlit dashboard: portfolio value, daily P&L, equity curve, open positions, with configurable refresh interval
- [x] 28.5 Implement system health monitoring: data feed status, broker connection status, heartbeat tracking, error alerting
- [x] 28.6 Write tests for trade logging, P&L calculations, and benchmark metrics
- [x] 28.7 Write tutorial chapter: Monitoring — dashboards, trade logs, and performance analytics

## 29. Integration & Polish

- [x] 29.1 Create end-to-end example: full pipeline from data fetch → strategy → backtest → analysis using momentum strategy
- [x] 29.2 Create end-to-end example: ML pipeline from features → model training → backtested ML strategy
- [x] 29.3 Create end-to-end example: AI-assisted workflow — sentiment + strategy signals → AI agent recommendation
- [x] 29.4 Create end-to-end example: gradient boosting long-short strategy (ref: book Ch 12)
- [x] 29.5 Create end-to-end example: deep RL trading agent with evaluation
- [x] 29.6 Write tutorial chapter: Putting It All Together — combining all modules into a complete system
- [x] 29.7 Create quickstart guide and installation instructions on the landing page
- [x] 29.8 Review and cross-link all tutorial chapters for navigation coherence
- [x] 29.9 Create alpha factor library appendix (TA-Lib indicators, WorldQuant alphas, SHAP analysis) (ref: book Appendix)
