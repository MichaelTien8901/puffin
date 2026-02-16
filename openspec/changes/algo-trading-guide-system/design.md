## Context

This is a greenfield project ("Puffin") that combines educational tutorial content with a production-capable algorithmic trading system. The target audience ranges from beginners learning market fundamentals to intermediate developers building ML/AI-powered trading systems. The project has two parallel tracks: a GitHub Pages tutorial site that teaches concepts progressively, and a Python codebase that implements each concept as working, reusable modules. Each tutorial chapter maps to a code module, so learners build the system as they learn.

There are no existing systems to integrate with — this is built from scratch. The primary constraint is that the tutorial must be self-contained and progressable: a learner should be able to follow chapters in order and have a working system at each stage.

## Goals / Non-Goals

**Goals:**
- Progressive learning path: each module builds on the previous, with working code at every step
- Tutorial site deployable via GitHub Pages with minimal configuration
- Modular Python architecture where each capability is independently usable
- Support for both paper trading (safe learning) and live trading (real usage)
- Cover the full spectrum: classical strategies → ML models → AI-assisted trading
- Reproducible environments via Docker and pinned dependencies

**Non-Goals:**
- High-frequency trading (HFT) or ultra-low-latency execution — focus is on medium-frequency strategies
- Building a proprietary data vendor — we use existing free/affordable data APIs
- Mobile app or native desktop GUI — web dashboards and CLI are sufficient
- Multi-asset exotic derivatives — focus on equities, ETFs, and crypto
- Institutional-grade compliance/regulatory tooling

## Decisions

### 1. Tutorial Site: Jekyll + Just the Docs
**Choice**: Jekyll with Just the Docs theme
**Rationale**: Consistent with the existing Rust guide tutorial project (rust-guide-tutorial), which uses the same stack with a proven Docker-based development workflow. Just the Docs provides built-in search, dark mode, callouts, Mermaid diagram support, and clean navigation. GitHub Pages has native Jekyll support, requiring no custom deploy scripts. Reusing the same stack reduces toolchain fragmentation and leverages existing knowledge.
**Alternatives**: MkDocs Material (Python-native but adds Ruby/Python toolchain split), Hugo (fast builds but Go dependency), Jupyter Book (good for notebooks but heavier).

### 2. Project Structure: Monorepo with Module-per-Capability
**Choice**: Single repository with `puffin/` Python package, each capability as a subpackage, and `docs/` for MkDocs content.
```
puffin/
├── docs/                    # Jekyll tutorial site (GitHub Pages)
│   ├── _config.yml          # Jekyll + Just the Docs config
│   ├── Gemfile              # Ruby dependencies
│   ├── Dockerfile           # Docker dev environment for docs
│   ├── docker-compose.yml
│   ├── _includes/
│   ├── _layouts/
│   ├── _sass/
│   ├── assets/
│   ├── index.md
│   ├── 01-market-foundations/
│   ├── 02-data-pipeline/
│   ├── 03-alternative-data/
│   ├── 04-alpha-factors/
│   ├── 05-portfolio-optimization/
│   ├── 06-trading-strategies/
│   ├── 07-backtesting/
│   ├── 08-linear-models/
│   ├── 09-time-series-models/
│   ├── 10-bayesian-ml/
│   ├── 11-tree-ensembles/
│   ├── 12-unsupervised-learning/
│   ├── 13-nlp-trading/
│   ├── 14-topic-modeling/
│   ├── 15-word-embeddings/
│   ├── 16-deep-learning/
│   ├── 17-cnns-for-trading/
│   ├── 18-rnns-for-trading/
│   ├── 19-autoencoders/
│   ├── 20-synthetic-data-gans/
│   ├── 21-deep-rl/
│   ├── 22-ai-assisted-trading/
│   ├── 23-live-trading/
│   ├── 24-risk-management/
│   └── 25-monitoring-analytics/
├── puffin/                  # Python package
│   ├── data/                # data-pipeline + alternative data
│   ├── factors/             # alpha-factors
│   ├── portfolio/           # portfolio-optimization
│   ├── strategies/          # strategy-library
│   ├── backtest/            # backtesting-engine
│   ├── models/              # linear, time-series, bayesian models
│   ├── ensembles/           # tree-based ensembles
│   ├── unsupervised/        # PCA, clustering, eigenportfolios
│   ├── nlp/                 # NLP, topic modeling, embeddings
│   ├── deep/                # deep learning (CNNs, RNNs, autoencoders, GANs)
│   ├── rl/                  # deep reinforcement learning
│   ├── ml/                  # ML utilities, features
│   ├── ai/                  # ai-assisted-trading (LLM)
│   ├── broker/              # live-trading
│   ├── risk/                # risk-management
│   └── monitor/             # monitoring-analytics
├── tests/
├── notebooks/               # Jupyter notebooks for interactive learning
├── pyproject.toml
└── Dockerfile               # Python dev environment
```
**Rationale**: Monorepo keeps tutorial content and code tightly coupled. Module-per-capability allows independent development and testing. Part-numbered docs folders (matching the Rust guide convention) enforce the learning progression. Jekyll docs get their own Dockerfile (as in the Rust guide) separate from the Python Dockerfile.
**Alternatives**: Multi-repo (tutorial separate from code — breaks the coupling that makes this project valuable), flat module structure (loses organization at scale).

### 3. Data Layer: yfinance + Alpaca for Data, SQLite for Local Storage
**Choice**: yfinance for free historical data (tutorials/backtesting), Alpaca API for real-time and paper trading data, SQLite for local data caching.
**Rationale**: yfinance is free and requires no API key — perfect for tutorials where setup friction must be minimal. Alpaca offers free paper trading with real market data. SQLite requires no server setup and stores locally, keeping the project self-contained. For production use, the data layer abstracts the source so users can swap in Polygon.io or other providers.
**Alternatives**: Polygon.io (better data quality but paid), PostgreSQL/TimescaleDB (better for production but requires server setup — can be added as an advanced topic), CSV files (too primitive for anything beyond basics).

### 4. Backtesting: Custom Lightweight Engine over Zipline/Backtrader
**Choice**: Build a lightweight backtesting engine as part of the tutorial.
**Rationale**: The primary goal is education. Using an existing framework (Zipline, Backtrader) hides the mechanics that learners need to understand: event loops, order matching, slippage modeling, performance calculation. Building it teaches these concepts. The engine will be simple (~500-1000 lines) but extensible. Advanced users can later integrate with established frameworks.
**Alternatives**: Zipline (abandoned/difficult to install), Backtrader (good but complex API), vectorbt (fast but abstracts too much for learning).

### 5. ML Framework: scikit-learn + statsmodels + PyTorch + TensorFlow 2
**Choice**: Four-tier approach — statsmodels for econometrics (linear models, time series, Fama-French), scikit-learn for classical ML (classification, regression, clustering), PyTorch for deep learning and RL, TensorFlow 2 for specific architectures (autoencoders, GANs, TimeGAN).
**Rationale**: The reference book uses all four frameworks. statsmodels is essential for proper statistical inference (p-values, confidence intervals) in linear and time-series models. scikit-learn handles classical ML. PyTorch for deep RL and custom architectures. TensorFlow 2 for GANs/autoencoders where the book provides TF2 implementations. Supporting both DL frameworks is necessary to be comprehensive.
**Alternatives**: Single framework (loses book compatibility), JAX (too advanced for tutorial context).

### 5a. NLP Stack: spaCy + Gensim + Transformers
**Choice**: spaCy for NLP pipeline (tokenization, NER), Gensim for topic modeling and word2vec/doc2vec, HuggingFace transformers for BERT and pretrained models.
**Rationale**: These are the standard tools used in the reference book. spaCy provides industrial-strength NLP processing, Gensim is the go-to for topic models and embeddings, and HuggingFace is the standard for transformer models.
**Alternatives**: NLTK (older, less performant), all-transformers (overkill for bag-of-words and LDA tutorials).

### 5b. Gradient Boosting: XGBoost + LightGBM + CatBoost
**Choice**: Support all three major gradient boosting libraries with comparative tutorials.
**Rationale**: The reference book covers all three with comparative benchmarks. In practice, LightGBM is fastest, CatBoost handles categorical features best, and XGBoost is the most established. Showing all three lets learners make informed choices.
**Alternatives**: Single library (loses the comparison value that makes this topic useful).

### 5c. Bayesian ML: PyMC
**Choice**: PyMC (v5+) for Bayesian modeling tutorials.
**Rationale**: PyMC is the standard Python library for Bayesian inference and matches the reference book. It supports MCMC sampling, variational inference, and has good visualization tools via ArviZ.
**Alternatives**: Stan/PyStan (more complex setup), NumPyro (JAX-based, less accessible).

### 5d. Model Interpretability: SHAP
**Choice**: SHAP (SHapley Additive exPlanations) for model interpretation.
**Rationale**: SHAP provides unified feature importance across all model types (tree, linear, deep learning). The reference book uses SHAP extensively for interpreting gradient boosting results. Essential for understanding what drives trading signals.
**Alternatives**: LIME (less comprehensive), eli5 (less maintained).

### 6. AI-Assisted Trading: Claude API as Primary LLM
**Choice**: Anthropic Claude API for sentiment analysis, news interpretation, and AI agent capabilities, with OpenAI as a documented alternative.
**Rationale**: Claude excels at nuanced text analysis and structured reasoning — both critical for financial news interpretation. The project will use a provider-agnostic wrapper so users can swap LLM providers. Tutorial content will use Claude but document how to switch.
**Alternatives**: OpenAI GPT (viable alternative, documented), local models via Ollama (added as advanced topic for users who want no API dependency).

### 7. Live Trading Broker: Alpaca as Primary, IBKR as Advanced
**Choice**: Alpaca for paper and live trading in tutorials, Interactive Brokers (IBKR) as an advanced integration.
**Rationale**: Alpaca has the simplest API, offers commission-free trading, has excellent paper trading, and requires minimal account setup. IBKR has broader market access but a complex API — better suited as an advanced chapter.
**Alternatives**: TD Ameritrade (API deprecated), Robinhood (no official API), Binance (crypto-only, good for crypto chapter).

### 8. Dashboards: Streamlit over Dash/Grafana
**Choice**: Streamlit for performance dashboards and monitoring UI.
**Rationale**: Streamlit is pure Python, requires no frontend knowledge, and produces interactive dashboards with minimal code — ideal for a tutorial context. Dash requires more boilerplate, Grafana requires separate infrastructure.
**Alternatives**: Dash (more customizable but more complex), Grafana (production-grade but requires separate service), Panel (less ecosystem support).

## Risks / Trade-offs

- **Free data quality** → yfinance data can be delayed/incomplete for some tickers. Mitigation: document limitations, provide Alpaca/Polygon upgrade path, include data validation utilities.

- **Tutorial scope — 25 capabilities** → Expanded to cover the full reference book (23 chapters) plus AI-assisted trading. Mitigation: each module is independently valuable; organize into tiers: Tier 1 (foundations, data, strategies, backtesting), Tier 2 (linear models, time series, tree ensembles), Tier 3 (NLP, deep learning, RL, AI). Ship tier by tier.

- **API key management for learners** → Multiple APIs (Alpaca, LLM providers) require keys. Mitigation: use `.env` files with `.env.example` templates, document setup clearly, provide fallback to offline/cached data where possible.

- **Broker API changes** → Third-party APIs can change without notice. Mitigation: abstract broker interactions behind an interface; pin API client versions.

- **Custom backtester limitations** → A tutorial-grade backtester won't handle all edge cases (corporate actions, splits, dividends). Mitigation: clearly document limitations, provide guidance on graduating to production frameworks.

- **LLM cost for AI modules** → API calls cost money. Mitigation: cache responses, use smaller models for development, provide pre-cached example outputs in tutorials so learners can follow along without API access.

## Migration Plan

Not applicable — greenfield project. Deployment steps:

1. Initialize repository with `pyproject.toml`, MkDocs config, and basic project structure
2. Deploy empty tutorial site to GitHub Pages via GitHub Actions
3. Build and deploy capabilities in tutorial order (foundations first)
4. Each chapter merges to main and auto-deploys to the tutorial site

## Open Questions

- **Which crypto exchanges to support?** Binance is largest but has regulatory complexity. Coinbase has a cleaner API. Could defer crypto to a dedicated chapter.
- **Should notebooks be primary or supplementary?** Jupyter notebooks are great for interactive learning but harder to maintain than markdown + code files. Current plan: notebooks as supplementary hands-on exercises, not the primary tutorial format.
- **Grading/progress tracking?** Should the tutorial site track learner progress (quizzes, checkpoints)? This adds complexity but improves the learning experience. Defer to a later iteration.
