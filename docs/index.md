---
layout: default
title: Home
nav_order: 1
permalink: /
---

# Puffin: Algorithmic Trading Guide

A comprehensive, hands-on guide to building algorithmic trading systems — from market fundamentals through machine learning, deep learning, and AI-assisted trading. Based on Stefan Jansen's *Machine Learning for Algorithmic Trading* (2nd Edition) and extended with modern LLM-powered trading.

{: .note }
This guide is designed to be followed sequentially. Each part builds on concepts and code from previous parts.

[Start Learning — Part 1: Market Foundations](01-market-foundations/){: .btn .btn-green .fs-5 .mb-4 .mb-md-0 .mr-2 }

## Learning Path

```mermaid
graph TD
    A[Part 1: Market Foundations] --> B[Part 2: Data Pipeline]
    B --> C[Part 3: Alternative Data]
    B --> D[Part 4: Alpha Factors]
    D --> E[Part 5: Portfolio Optimization]
    D --> F[Part 6: Trading Strategies]
    F --> G[Part 7: Backtesting]
    G --> H[Part 8: Linear Models]
    H --> I[Part 9: Time Series Models]
    I --> J[Part 10: Bayesian ML]
    G --> K[Part 11: Tree Ensembles]
    K --> L[Part 12: Unsupervised Learning]
    L --> M[Part 13: NLP for Trading]
    M --> N[Part 14: Topic Modeling]
    N --> O[Part 15: Word Embeddings]
    O --> P[Part 16: Deep Learning]
    P --> Q[Part 17: CNNs]
    P --> R[Part 18: RNNs]
    P --> S[Part 19: Autoencoders]
    S --> T[Part 20: GANs & Synthetic Data]
    P --> U[Part 21: Deep RL]
    U --> V[Part 22: AI-Assisted Trading]
    G --> W[Part 23: Live Trading]
    W --> X[Part 24: Risk Management]
    X --> Y[Part 25: Monitoring & Analytics]

    style A fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    style B fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    style C fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    style D fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4
    style E fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4
    style F fill:#8b4513,stroke:#5c2d0a,color:#e8e0d4
    style G fill:#8b4513,stroke:#5c2d0a,color:#e8e0d4
    style H fill:#4a2060,stroke:#2d1040,color:#e8e0d4
    style I fill:#4a2060,stroke:#2d1040,color:#e8e0d4
    style J fill:#4a2060,stroke:#2d1040,color:#e8e0d4
    style K fill:#4a2060,stroke:#2d1040,color:#e8e0d4
    style L fill:#4a2060,stroke:#2d1040,color:#e8e0d4
    style M fill:#0d4a5c,stroke:#083040,color:#e8e0d4
    style N fill:#0d4a5c,stroke:#083040,color:#e8e0d4
    style O fill:#0d4a5c,stroke:#083040,color:#e8e0d4
    style P fill:#6b2d5b,stroke:#401a38,color:#e8e0d4
    style Q fill:#6b2d5b,stroke:#401a38,color:#e8e0d4
    style R fill:#6b2d5b,stroke:#401a38,color:#e8e0d4
    style S fill:#6b2d5b,stroke:#401a38,color:#e8e0d4
    style T fill:#6b2d5b,stroke:#401a38,color:#e8e0d4
    style U fill:#6b2d5b,stroke:#401a38,color:#e8e0d4
    style V fill:#8b4513,stroke:#5c2d0a,color:#e8e0d4
    style W fill:#7a2020,stroke:#4a1010,color:#e8e0d4
    style X fill:#7a2020,stroke:#4a1010,color:#e8e0d4
    style Y fill:#7a2020,stroke:#4a1010,color:#e8e0d4

    click A "01-market-foundations/"
    click B "02-data-pipeline/"
    click C "03-alternative-data/"
    click D "04-alpha-factors/"
    click E "05-portfolio-optimization/"
    click F "06-trading-strategies/"
    click G "07-backtesting/"
    click H "08-linear-models/"
    click I "09-time-series-models/"
    click J "10-bayesian-ml/"
    click K "11-tree-ensembles/"
    click L "12-unsupervised-learning/"
    click M "13-nlp-trading/"
    click N "14-topic-modeling/"
    click O "15-word-embeddings/"
    click P "16-deep-learning/"
    click Q "17-cnns-for-trading/"
    click R "18-rnns-for-trading/"
    click S "19-autoencoders/"
    click T "20-synthetic-data-gans/"
    click U "21-deep-rl/"
    click V "22-ai-assisted-trading/"
    click W "23-live-trading/"
    click X "24-risk-management/"
    click Y "25-monitoring-analytics/"
```

## Parts

| Part | Topic | What You'll Learn |
|------|-------|-------------------|
| [1. Market Foundations](01-market-foundations/) | How markets work | Exchanges, order books, asset classes, market microstructure |
| [2. Data Pipeline](02-data-pipeline/) | Getting market data | Data providers, caching, preprocessing, HDF5/Parquet storage |
| [3. Alternative Data](03-alternative-data/) | Non-traditional data | Web scraping, earnings calls, alternative data evaluation |
| [4. Alpha Factors](04-alpha-factors/) | Predictive signals | TA-Lib, Kalman filter, wavelets, Alphalens, WorldQuant alphas |
| [5. Portfolio Optimization](05-portfolio-optimization/) | Building portfolios | Mean-variance, risk parity, HRP, pyfolio tearsheets |
| [6. Trading Strategies](06-trading-strategies/) | Classical strategies | Momentum, mean reversion, stat arb, market making |
| [7. Backtesting](07-backtesting/) | Testing strategies | Event-driven backtester, walk-forward analysis |
| [8. Linear Models](08-linear-models/) | Linear ML | OLS, ridge, lasso, Fama-French factor models |
| [9. Time Series Models](09-time-series-models/) | Time series | ARIMA, VAR, GARCH, cointegration, pairs trading |
| [10. Bayesian ML](10-bayesian-ml/) | Bayesian methods | PyMC, Bayesian Sharpe, stochastic volatility |
| [11. Tree Ensembles](11-tree-ensembles/) | Gradient boosting | Random forests, XGBoost, LightGBM, CatBoost, SHAP |
| [12. Unsupervised Learning](12-unsupervised-learning/) | Clustering & PCA | Eigenportfolios, k-means, hierarchical clustering |
| [13. NLP for Trading](13-nlp-trading/) | Text analysis | spaCy, TF-IDF, naive Bayes, sentiment analysis |
| [14. Topic Modeling](14-topic-modeling/) | Document topics | LSI, LDA, pyLDAvis, earnings call analysis |
| [15. Word Embeddings](15-word-embeddings/) | Semantic analysis | word2vec, GloVe, doc2vec, BERT, SEC filings |
| [16. Deep Learning](16-deep-learning/) | Neural networks | Feedforward NNs, PyTorch, TensorFlow, TensorBoard |
| [17. CNNs for Trading](17-cnns-for-trading/) | Convolutions | 1D CNN, CNN-TA image approach, transfer learning |
| [18. RNNs for Trading](18-rnns-for-trading/) | Sequence models | LSTM, GRU, stacked RNNs, sentiment classification |
| [19. Autoencoders](19-autoencoders/) | Feature extraction | Denoising AE, VAE, conditional AE for pricing |
| [20. GANs & Synthetic Data](20-synthetic-data-gans/) | Data generation | TimeGAN, synthetic financial time series |
| [21. Deep RL](21-deep-rl/) | RL agents | Q-learning, DQN, DDQN, PPO, trading agents |
| [22. AI-Assisted Trading](22-ai-assisted-trading/) | LLM-powered trading | Sentiment, news signals, AI agent portfolio mgmt |
| [23. Live Trading](23-live-trading/) | Going live | Paper trading, broker integration, order management |
| [24. Risk Management](24-risk-management/) | Managing risk | Position sizing, stop losses, VaR, portfolio controls |
| [25. Monitoring & Analytics](25-monitoring-analytics/) | Tracking performance | Dashboards, trade logs, P&L attribution |

## Prerequisites

- Python 3.11+
- Basic Python programming knowledge
- Basic understanding of financial markets (helpful but not required)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/MichaelTien8901/puffin.git
cd puffin

# Install core dependencies
pip install -e .

# Install all optional dependencies (ML, NLP, AI, dashboard, dev tools)
pip install -e ".[all]"

# Copy environment template and add your API keys
cp .env.example .env
```

## Project Structure

```
puffin/
├── docs/            # This tutorial site (Jekyll + Just the Docs)
├── puffin/          # Python package
│   ├── data/        # Data pipeline
│   ├── factors/     # Alpha factor research
│   ├── portfolio/   # Portfolio optimization
│   ├── strategies/  # Trading strategies
│   ├── backtest/    # Backtesting engine
│   ├── models/      # Linear & time series models
│   ├── ensembles/   # Tree ensemble models
│   ├── unsupervised/# PCA, clustering
│   ├── nlp/         # NLP for trading
│   ├── deep/        # Deep learning (CNN, RNN, AE, GAN)
│   ├── rl/          # Deep reinforcement learning
│   ├── ml/          # ML utilities
│   ├── ai/          # AI-assisted trading (LLMs)
│   ├── broker/      # Live trading
│   ├── risk/        # Risk management
│   └── monitor/     # Monitoring & analytics
├── tests/           # Test suite
└── notebooks/       # Interactive Jupyter notebooks
```
