# Puffin - Algorithmic Trading Guide & System

## Project Overview
Puffin is a comprehensive guided tutorial and production-capable algorithmic trading system. It combines educational content (hosted on GitHub Pages via Jekyll + Just the Docs) with working Python code modules.

## Reference Material
- **Primary textbook**: `Machine Learning for Algorithmic Trading (9781839217715).pdf` by Stefan Jansen (2nd Edition, 2020)
- This book's content should be incorporated into the project tutorials and code modules
- Use `pdftotext` to read the PDF file when needed: `pdftotext "Machine Learning for Algorithmic Trading(9781839217715).pdf" - | head -N`

## Book Structure to Incorporate

### Part 1: Data & Alpha Factors (Chapters 1-5)
- Ch 1: ML for Trading overview, strategy design, use cases
- Ch 2: Market & fundamental data (NASDAQ ITCH, FIX protocol, tick-to-bar, yfinance, Quandl)
- Ch 3: Alternative data (web scraping, earnings calls, OpenTable, social media)
- Ch 4: Alpha factor engineering (momentum, value, volatility, quality factors, TA-Lib, Kalman filter, wavelets, Alphalens)
- Ch 5: Portfolio optimization (mean-variance, risk parity, hierarchical risk parity, pyfolio)

### Part 2: ML Fundamentals (Chapters 6-8)
- Ch 6: ML process (workflow, cross-validation, parameter tuning, scikit-learn)
- Ch 7: Linear models (OLS, ridge, lasso, logistic regression, Fama-French factors)
- Ch 8: ML4T workflow (backtesting pitfalls, backtrader, Zipline, Pipeline API)

### Part 3: Time Series & Bayesian (Chapters 9-10)
- Ch 9: Time-series models (ARIMA, VAR, GARCH, cointegration, pairs trading)
- Ch 10: Bayesian ML (PyMC3, Bayesian Sharpe ratio, stochastic volatility)

### Part 4: Tree-Based Models (Chapters 11-13)
- Ch 11: Random forests (decision trees, feature importance, Japanese equities strategy)
- Ch 12: Gradient boosting (AdaBoost, XGBoost, LightGBM, CatBoost, intraday strategies)
- Ch 13: Unsupervised learning (PCA, clustering, eigenportfolios, HRP)

### Part 5: NLP & Text Data (Chapters 14-16)
- Ch 14: Sentiment analysis (NLP pipeline, bag-of-words, naive Bayes, Twitter/Yelp)
- Ch 15: Topic modeling (LSI, LDA, earnings calls, financial news)
- Ch 16: Word embeddings (word2vec, GloVe, doc2vec, SEC filings, transformers/BERT)

### Part 6: Deep Learning (Chapters 17-22)
- Ch 17: Deep learning fundamentals (feedforward NNs, TensorFlow 2, PyTorch)
- Ch 18: CNNs (financial time series, satellite images, transfer learning)
- Ch 19: RNNs/LSTMs (multivariate time series, sentiment analysis)
- Ch 20: Autoencoders (conditional risk factors, VAE, asset pricing)
- Ch 21: GANs (synthetic time-series data, TimeGAN)
- Ch 22: Deep RL (Q-learning, DQN, DDQN, OpenAI Gym trading agent)

### Appendix
- Alpha factor library (TA-Lib indicators, WorldQuant formulaic alphas, SHAP values)

## Tech Stack
- **Tutorial site**: Jekyll + Just the Docs theme (consistent with rust-guide-tutorial)
- **Language**: Python 3.11+
- **Data**: yfinance (free), Alpaca (paper/live), SQLite cache
- **ML**: scikit-learn, PyTorch, stable-baselines3, gymnasium
- **AI**: Anthropic Claude API, OpenAI API
- **Dashboards**: Streamlit
- **Brokers**: Alpaca (primary), Interactive Brokers (advanced)

## Project Structure
```
puffin/
├── docs/                    # Jekyll tutorial site
│   ├── 01-market-foundations/
│   ├── 02-data-pipeline/
│   ├── ...
│   └── 09-monitoring-analytics/
├── puffin/                  # Python package
│   ├── data/               # Data providers, cache, preprocessing
│   ├── strategies/         # Trading strategies
│   ├── backtest/           # Backtesting engine
│   ├── ml/                 # ML models, features, RL
│   ├── ai/                 # LLM providers, sentiment, agent
│   ├── broker/             # Live trading integration
│   ├── risk/               # Risk management
│   └── monitor/            # Monitoring & analytics
├── tests/
├── notebooks/
└── openspec/               # OpenSpec change management
```

## Development Environment
- All test and build tools run inside Docker (no local Python/pytest)
- Build image: `docker build -t puffin-test .`
- Run tests: `docker run --rm puffin-test python -m pytest tests/`
- Run specific tests: `docker run --rm puffin-test python -m pytest tests/broker/test_ibkr.py -v`

## Conventions
- Use `pdftotext` for reading the reference PDF, not the Read tool
- Tutorial docs use Jekyll + Just the Docs with folder naming: `NN-topic-name/`
- Python code follows the Strategy pattern for swappable components (DataProvider, Strategy, Broker, LLMProvider)
- All strategies implement the `Strategy` base class with `generate_signals()` and `get_parameters()`


## GitHub Pages / docs/ Setup

* Reference: ~/projects/rust-guide-tutorial-openspec/docs/ for the proven pattern.

* Theme: just-the-docs with color_scheme: dark
* Mermaid: Built-in support via mermaid: config key (version 10.6.0)
* Gemfile: jekyll ~> 4.3, just-the-docs ~> 0.8
* Front matter: Every page needs layout: default, title:, nav_order:, optional parent: for hierarchy
* Custom SCSS: docs/_sass/custom/custom.scss for overrides
* Mermaid Styling Convention
    * Mermaid diagrams use dark fill colors (RGB channels ≤ 0x90) with a paper-like background:

    * Paper background: #d5d0c8 on .mermaid, pre.mermaid, .language-mermaid
    * Node fills: Use RGB values where each channel ≤ 0x90 (e.g., #2d5016, #1a3a5c, #6b2d5b, #8b4513)
    * Text: #2c3e50 (dark gray) on .nodeLabel, .label
    * Edges/arrows: #4a5568 with stroke-width: 2px
    * Actor lines: #7a8a9a
    * In Mermaid source, use style or classDef with dark fills:

    * classDef dark fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    * classDef accent fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4