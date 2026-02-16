# Puffin - Algorithmic Trading Guide & System

A comprehensive, hands-on guide to building algorithmic trading systems — from market fundamentals through machine learning, deep learning, and AI-assisted trading. Based on Stefan Jansen's *Machine Learning for Algorithmic Trading* (2nd Edition) and extended with modern LLM-powered trading.

**Tutorial Site**: [michaeltien8901.github.io/puffin](https://michaeltien8901.github.io/puffin/)

## What You'll Learn

| Section | Topics |
|---------|--------|
| **Data & Alpha Factors** (Parts 1-5) | Market microstructure, data pipelines, alternative data, alpha factors, portfolio optimization |
| **Trading & Backtesting** (Parts 6-7) | Momentum, mean reversion, stat arb, event-driven backtesting |
| **ML Models** (Parts 8-12) | Linear models, time series (ARIMA/GARCH), Bayesian ML, tree ensembles, unsupervised learning |
| **NLP for Trading** (Parts 13-15) | Sentiment analysis, topic modeling, word embeddings (BERT) |
| **Deep Learning** (Parts 16-21) | CNNs, RNNs/LSTMs, autoencoders, GANs, deep reinforcement learning |
| **Production** (Parts 22-25) | AI-assisted trading, live trading, risk management, monitoring |

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
├── docs/            # Tutorial site (Jekyll + Just the Docs)
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
│   ├── ai/          # AI-assisted trading (LLMs)
│   ├── broker/      # Live trading
│   ├── risk/        # Risk management
│   └── monitor/     # Monitoring & analytics
├── tests/
└── notebooks/
```

## Tech Stack

- **Language**: Python 3.11+
- **Data**: yfinance, Alpaca (paper/live), SQLite cache
- **ML**: scikit-learn, PyTorch, stable-baselines3, gymnasium
- **AI**: Anthropic Claude API, OpenAI API
- **Dashboards**: Streamlit
- **Brokers**: Alpaca (primary), Interactive Brokers (advanced)

## Prerequisites

- Python 3.11+
- Basic Python programming knowledge
- Basic understanding of financial markets (helpful but not required)

## License

Content is provided for educational purposes.
