## 1. Stub Chapters (Quick Wins)

- [x] 1.1 Expand 02-data-pipeline (148 lines) into index.md + 3 sub-pages: data providers, caching & storage, preprocessing
- [x] 1.2 Expand 07-backtesting (158 lines) into index.md + 3 sub-pages: event-driven engine, execution models, walk-forward analysis
- [x] 1.3 Expand 24-risk-management (376 lines) into index.md + 3 sub-pages: position sizing, stop losses, portfolio risk controls

## 2. Data Cluster (Chapters 03–05)

- [x] 2.1 Expand 03-alternative-data into index.md + 2 sub-pages: web scraping & transcript parsing, alt data evaluation
- [x] 2.2 Expand 04-alpha-factors into index.md + 3 sub-pages: momentum/value/volatility factors, Kalman & wavelets, factor evaluation
- [x] 2.3 Expand 05-portfolio-optimization into index.md + 3 sub-pages: mean-variance, risk parity, hierarchical risk parity

## 3. ML Cluster (Chapters 08–12)

- [x] 3.1 Expand 08-linear-models into index.md + 3 sub-pages: OLS/ridge/lasso, logistic regression, Fama-French factor models
- [x] 3.2 Expand 09-time-series-models into index.md + 3 sub-pages: diagnostics & stationarity, ARIMA/VAR/GARCH, cointegration & pairs trading
- [x] 3.3 Expand 10-bayesian-ml into index.md + 3 sub-pages: PyMC fundamentals, Bayesian Sharpe ratio, stochastic volatility
- [x] 3.4 Expand 11-tree-ensembles into index.md + 4 sub-pages: random forests, gradient boosting (XGB/LGBM/CatBoost), SHAP interpretation, long-short strategy
- [x] 3.5 Expand 12-unsupervised-learning into index.md + 3 sub-pages: PCA & eigenportfolios, clustering methods, data-driven risk factors

## 4. NLP Cluster (Chapters 13–15)

- [x] 4.1 Expand 13-nlp-trading into index.md + 3 sub-pages: NLP pipeline & tokenization, bag-of-words & TF-IDF, sentiment classification
- [x] 4.2 Expand 14-topic-modeling into index.md + 2 sub-pages: LSI & LDA, earnings call topic analysis
- [x] 4.3 Expand 15-word-embeddings into index.md + 3 sub-pages: word2vec & GloVe, doc2vec & SEC filings, transformer embeddings

## 5. Deep Learning Cluster (Chapters 16–21)

- [x] 5.1 Expand 16-deep-learning into index.md + 3 sub-pages: feedforward NNs, training utilities, TensorBoard integration
- [x] 5.2 Expand 17-cnns-for-trading into index.md + 2 sub-pages: 1D CNN for time series, CNN-TA & transfer learning
- [x] 5.3 Expand 18-rnns-for-trading into index.md + 3 sub-pages: LSTM fundamentals, stacked LSTM & GRU, sentiment RNN
- [x] 5.4 Expand 19-autoencoders into index.md + 3 sub-pages: standard & denoising AE, variational AE, conditional AE for asset pricing
- [x] 5.5 Expand 20-synthetic-data-gans into index.md + 3 sub-pages: GAN architecture, TimeGAN, synthetic data evaluation
- [x] 5.6 Expand 21-deep-rl into index.md + 3 sub-pages: Q-learning, DQN & DDQN, PPO & trading environment

## 6. Operational Cluster (Chapters 22, 25)

- [x] 6.1 Expand 22-ai-assisted-trading into index.md + 3 sub-pages: LLM providers, sentiment & signal generation, AI agent portfolio manager
- [x] 6.2 Expand 25-monitoring-analytics into index.md + 3 sub-pages: trade logging & P&L, benchmark comparison, Streamlit dashboard

## 7. Mermaid Diagrams

- [x] 7.1 Add overview/architecture diagrams to index.md pages for chapters 02–05 (data cluster)
- [x] 7.2 Add overview/architecture diagrams to index.md pages for chapters 07–12 (ML cluster + backtesting)
- [x] 7.3 Add overview/architecture diagrams to index.md pages for chapters 13–15 (NLP cluster)
- [x] 7.4 Add overview/architecture diagrams to index.md pages for chapters 16–21 (deep learning cluster)
- [x] 7.5 Add overview/architecture diagrams to index.md pages for chapters 22, 24–25 (operational cluster)
- [x] 7.6 Add focused sub-page diagrams where algorithms/pipelines benefit from visualization (at least 1 per cluster)

## 8. Cross-Links

- [x] 8.1 Add Related Chapters sections to all data cluster index pages (02–05) with bidirectional links
- [x] 8.2 Add Related Chapters sections to all ML cluster index pages (07–12) with bidirectional links
- [x] 8.3 Add Related Chapters sections to all NLP cluster index pages (13–15) with bidirectional links
- [x] 8.4 Add Related Chapters sections to all deep learning cluster index pages (16–21) with bidirectional links
- [x] 8.5 Add Related Chapters sections to all operational cluster index pages (22, 24–25) with bidirectional links
- [x] 8.6 Add inline cross-references in sub-pages where concepts from other chapters are mentioned
- [x] 8.7 Verify bidirectional link consistency across all chapters

## 9. Cluster Notebooks

- [x] 9.1 Create `notebooks/data_pipeline.ipynb` covering chapters 02–05 with puffin data/factors/portfolio imports
- [x] 9.2 Create `notebooks/ml_models.ipynb` covering chapters 08–12 with puffin models/ensembles/unsupervised imports
- [x] 9.3 Create `notebooks/nlp_trading.ipynb` covering chapters 13–15 with puffin nlp imports
- [x] 9.4 Create `notebooks/deep_learning.ipynb` covering chapters 16–21 with puffin deep/rl imports
- [x] 9.5 Create `notebooks/operational.ipynb` covering chapters 06–07, 22–25 with puffin strategies/backtest/broker/monitor imports
- [x] 9.6 Validate all 5 notebooks execute cleanly via `jupyter nbconvert --execute` in Docker

## 10. Verification

- [x] 10.1 Run `bundle exec jekyll build` and verify zero build errors
- [x] 10.2 Validate all internal links (cross-chapter, sub-page, anchor links) resolve correctly using html-proofer or manual crawl of built `_site/`
- [x] 10.3 Validate all external links (GitHub source links, paper references, tool websites) return 200 status codes
- [x] 10.4 Verify all 22 expanded chapters have index.md with has_children + ≥2 child pages
- [x] 10.5 Verify all chapters have at least one Mermaid diagram
- [x] 10.6 Verify all chapters have a Related Chapters section with 2–4 links
- [x] 10.7 Spot-check 5 chapters for working puffin import code examples
- [x] 10.8 Verify nav_order values are sequential and consistent across all index.md and child pages (no duplicates, no gaps, correct parent-child ordering)
