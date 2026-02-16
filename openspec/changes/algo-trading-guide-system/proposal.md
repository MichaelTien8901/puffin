## Why

There is no single, structured learning path that takes someone from market fundamentals through to building a fully capable algorithmic trading system — one that covers classical strategies, ML-based approaches, and modern AI-assisted trading. Existing resources are fragmented across books, courses, and blog posts, making it hard to build knowledge incrementally. This project creates a guided, hands-on tutorial system hosted on GitHub Pages that teaches the foundations and progressively builds toward a production-capable trading system. By combining educational content with working code, learners retain knowledge through practice while ending up with a real system they can use.

The curriculum is based on Stefan Jansen's *Machine Learning for Algorithmic Trading* (2nd Edition), expanded with modern AI-assisted trading capabilities using LLMs.

## What Changes

- Create a GitHub Pages-hosted tutorial site with structured, progressive learning modules
- Build foundational content covering market structure, asset classes, order types, and how exchanges work
- Implement market microstructure data handling (ITCH data, tick-to-bar conversion, FIX protocol concepts)
- Build alternative data modules (web scraping, earnings call transcripts, social media)
- Develop alpha factor research framework (momentum, value, volatility, quality factors, TA-Lib, Kalman filter, wavelets, Alphalens evaluation)
- Implement portfolio optimization (mean-variance, risk parity, hierarchical risk parity, pyfolio integration)
- Develop strategy modules: momentum, mean reversion, statistical arbitrage, pairs trading, market making
- Create a backtesting engine for strategy validation with historical data
- Build data pipeline modules for market data ingestion (real-time and historical)
- Implement linear models for trading (OLS, ridge, lasso, logistic regression, Fama-French factor models)
- Implement time-series models (ARIMA, VAR, GARCH volatility forecasting, cointegration-based stat arb)
- Implement Bayesian ML for trading (PyMC, Bayesian Sharpe ratio, stochastic volatility)
- Implement tree-based ensemble models (random forests, XGBoost, LightGBM, CatBoost for signal generation)
- Implement unsupervised learning (PCA, clustering, eigenportfolios, data-driven risk factors)
- Build NLP pipeline for financial text (tokenization, bag-of-words, TF-IDF, naive Bayes, sentiment analysis)
- Implement topic modeling (LSI, LDA for earnings calls and financial news)
- Implement word embeddings (word2vec, GloVe, doc2vec for SEC filings and financial text)
- Build deep learning models (feedforward NNs, CNNs for time-series, RNNs/LSTMs for sequence modeling)
- Implement autoencoders (conditional risk factors, denoising, variational autoencoders)
- Implement GANs for synthetic time-series data generation (TimeGAN)
- Implement deep reinforcement learning trading agents (DQN, DDQN, PPO with custom OpenAI Gym environment)
- Build AI-assisted trading modules: LLM-powered sentiment analysis, news-driven signals, AI agent-based portfolio management
- Create a live trading integration layer supporting paper trading and real broker APIs
- Implement risk management and position sizing modules
- Build monitoring, logging, and performance analytics dashboards

## Capabilities

### New Capabilities
- `tutorial-site`: GitHub Pages site infrastructure with Jekyll + Just the Docs theme, navigation, and progressive module structure
- `market-foundations`: Educational content on market structure, asset classes, order books, exchanges, trading mechanics, and market microstructure (ITCH, FIX)
- `alternative-data`: Web scraping, earnings call transcripts, alternative data sources and evaluation framework
- `alpha-factors`: Alpha factor research framework — momentum, value, volatility, quality factors, TA-Lib integration, Kalman filter, wavelet denoising, Alphalens factor evaluation
- `portfolio-optimization`: Mean-variance optimization, risk parity, hierarchical risk parity (HRP), pyfolio performance evaluation
- `strategy-library`: Classical trading strategies (momentum, mean reversion, stat arb, pairs trading, market making) with theory and implementation
- `data-pipeline`: Market data ingestion, storage, and preprocessing for both historical and real-time feeds
- `backtesting-engine`: Strategy backtesting framework with performance metrics, walk-forward analysis, and visualization
- `linear-models`: Linear regression (OLS, ridge, lasso), logistic regression, Fama-French factor models for return prediction
- `time-series-models`: ARIMA, VAR, GARCH volatility forecasting, cointegration testing, pairs trading with time-series methods
- `bayesian-ml`: Bayesian ML with PyMC — Bayesian Sharpe ratio comparison, Bayesian rolling regression, stochastic volatility models
- `tree-ensembles`: Random forests, gradient boosting (XGBoost, LightGBM, CatBoost), feature importance, SHAP interpretation
- `unsupervised-learning`: PCA, clustering (k-means, hierarchical, DBSCAN, GMM), eigenportfolios, data-driven risk factors
- `nlp-trading`: NLP pipeline for financial text — tokenization, bag-of-words, TF-IDF, naive Bayes, sentiment analysis on news/social media
- `topic-modeling`: Latent Semantic Indexing (LSI), Latent Dirichlet Allocation (LDA) for earnings calls and financial news
- `word-embeddings`: word2vec, GloVe, doc2vec for financial text, SEC filing analysis, pretrained transformers (BERT)
- `deep-learning`: Feedforward NNs, CNNs for financial time series, RNNs/LSTMs for sequence prediction, TensorFlow 2 and PyTorch
- `autoencoders`: Standard, denoising, variational autoencoders for nonlinear feature extraction, conditional risk factors
- `synthetic-data`: GANs for synthetic financial time-series generation (TimeGAN), data augmentation
- `deep-rl`: Deep reinforcement learning trading agents — DQN, DDQN, PPO with custom Gymnasium trading environment
- `ml-trading`: Machine learning trading modules — feature engineering, model training, signal generation using supervised and reinforcement learning
- `ai-assisted-trading`: AI/LLM-powered trading capabilities — sentiment analysis, news-driven signals, AI agent portfolio management
- `live-trading`: Paper trading and live broker integration layer with order management
- `risk-management`: Position sizing, stop-loss logic, portfolio-level risk controls, and drawdown management
- `monitoring-analytics`: Performance dashboards, trade logging, P&L tracking, and system health monitoring

### Modified Capabilities
<!-- None — this is a greenfield project -->

## Impact

- **Code**: New Python project with modular architecture — each capability maps to a package/module
- **Dependencies**: Core libs (pandas, numpy, scikit-learn, pytorch, tensorflow, statsmodels, pymc, xgboost, lightgbm, catboost, gensim, spacy, shap), data providers (yfinance, alpaca, polygon.io), web framework for dashboards (streamlit), site generator (Jekyll 4.3 + Just the Docs 0.8)
- **APIs**: Broker APIs (Alpaca, Interactive Brokers), market data APIs, LLM APIs (Claude/OpenAI for AI-assisted features)
- **Infrastructure**: GitHub Pages for tutorial hosting, GitHub Actions for CI/CD and site deployment
- **Systems**: Local development environment with Docker support for reproducibility
- **Reference**: Based on *Machine Learning for Algorithmic Trading* by Stefan Jansen (23 chapters), expanded with AI-assisted trading
