## ADDED Requirements

### Requirement: One Jupyter notebook SHALL be created per chapter cluster
The following 5 cluster notebooks SHALL be created in `notebooks/`:

1. `data_pipeline.ipynb` — covers chapters 02–05 (data pipeline, alt data, alpha factors, portfolio optimization)
2. `ml_models.ipynb` — covers chapters 08–12 (linear models, time series, Bayesian, tree ensembles, unsupervised)
3. `nlp_trading.ipynb` — covers chapters 13–15 (NLP, topic modeling, word embeddings)
4. `deep_learning.ipynb` — covers chapters 16–21 (DL fundamentals, CNN, RNN, autoencoders, GANs, deep RL)
5. `operational.ipynb` — covers chapters 06–07, 22–25 (strategies, backtesting, AI-assisted, live trading, risk, monitoring)

#### Scenario: Notebook exists for each cluster
- **WHEN** the `notebooks/` directory is listed
- **THEN** it SHALL contain all 5 cluster notebooks listed above

### Requirement: Notebooks SHALL be auto-validatable
Each notebook SHALL execute without errors when run via `jupyter nbconvert --execute` inside the project's Docker environment.

#### Scenario: Clean execution
- **WHEN** `jupyter nbconvert --to notebook --execute notebooks/<name>.ipynb` is run in the Docker container
- **THEN** the command SHALL exit with code 0 and produce no cell errors

#### Scenario: No external API dependencies for validation
- **WHEN** a notebook is executed for validation
- **THEN** it SHALL NOT require live API keys (Alpaca, OpenAI, etc.) — use synthetic data or mocks for examples that would need external services

### Requirement: Notebooks SHALL use puffin module imports
Each notebook SHALL demonstrate the corresponding chapters' puffin modules with working import statements and realistic usage examples.

#### Scenario: Module coverage
- **WHEN** a cluster notebook is reviewed
- **THEN** it SHALL import from at least 3 distinct `puffin` subpackages relevant to its cluster

### Requirement: Notebooks SHALL link back to tutorial chapters
Each notebook SHALL include a markdown cell at the top listing the tutorial chapters it covers, with links to the docs site.

#### Scenario: Chapter reference header
- **WHEN** a notebook is opened
- **THEN** the first markdown cell SHALL list the covered chapters with descriptive links
