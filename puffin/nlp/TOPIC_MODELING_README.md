# Topic Modeling for Financial Text Analysis

This module provides topic modeling capabilities for analyzing earnings calls, financial news, and other financial documents.

## Features

- **LSI (Latent Semantic Indexing)**: Fast topic discovery using SVD
- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic modeling with Gensim or sklearn
- **Earnings Analysis**: Specialized tools for earnings call transcripts
- **Visualization**: Multiple plotting functions for topic exploration
- **Interactive**: pyLDAvis integration for interactive exploration

## Quick Start

### Installation

```bash
# Minimum requirements
pip install scikit-learn numpy

# Recommended (for LDA and coherence)
pip install gensim

# Optional (for visualization)
pip install matplotlib pyldavis
```

### Basic Usage

#### LSI Topic Model

```python
from puffin.nlp import LSIModel

# Prepare documents
documents = [
    "revenue growth exceeded expectations",
    "market volatility increased significantly",
    "technology innovation driving future"
]

# Fit model
model = LSIModel()
model.fit(documents, n_topics=3)

# Get topics
topics = model.get_topics(n_words=5)
for topic_id, words in topics:
    print(f"Topic {topic_id}: {[w for w, _ in words]}")

# Transform new documents
weights = model.transform(["strong revenue growth"])
```

#### LDA Topic Model

```python
from puffin.nlp import LDAModel

# Fit LDA model
model = LDAModel(use_gensim=True)
model.fit(documents, n_topics=3, passes=15)

# Get topics
topics = model.get_topics(n_words=5)

# Get topic distributions
distributions = model.transform(documents)

# Calculate coherence
coherence = model.coherence_score()
print(f"Coherence: {coherence:.3f}")
```

#### Earnings Call Analysis

```python
from puffin.nlp import EarningsTopicAnalyzer

# Load earnings transcripts
transcripts = [...]
dates = ['2024-01-31', '2024-04-30', ...]

# Analyze
analyzer = EarningsTopicAnalyzer(n_topics=5, model_type='lda')
results = analyzer.analyze(transcripts, dates=dates)

# Detect topic shifts
shifts = analyzer.detect_topic_shifts(
    transcripts,
    dates,
    window_size=3,
    threshold=0.3
)

# Analyze sentiment by topic
sentiment_map = analyzer.topic_sentiment(transcripts, sentiment_scores)
```

## Module Structure

### topic_models.py
- `LSIModel`: LSI implementation using TruncatedSVD
- `LDAModel`: LDA implementation with Gensim/sklearn backend
- `find_optimal_topics()`: Find optimal number of topics

### topic_viz.py
- `plot_topic_distribution()`: Bar chart of topic weights
- `plot_topic_evolution()`: Topic weights over time
- `plot_topic_heatmap()`: Document-topic heatmap
- `plot_topic_words()`: Top words visualization
- `plot_coherence_scores()`: Coherence vs number of topics
- `prepare_pyldavis()`: Prepare interactive visualization

### earnings_topics.py
- `EarningsTopicAnalyzer`: Specialized earnings analysis
- `compare_earnings_topics()`: Compare topic models

## Examples

### Find Optimal Topics

```python
from puffin.nlp import find_optimal_topics, plot_coherence_scores

optimal_n, scores = find_optimal_topics(
    documents,
    min_topics=2,
    max_topics=20,
    step=2,
    method='lda'
)

print(f"Optimal: {optimal_n} topics")

fig = plot_coherence_scores(scores)
fig.savefig('coherence.png')
```

### Visualize Topic Evolution

```python
from puffin.nlp import LDAModel, plot_topic_evolution

model = LDAModel()
model.fit(transcripts, n_topics=5)

fig = plot_topic_evolution(
    model,
    transcripts,
    dates,
    topic_ids=[0, 1, 2],
    rolling_window=3
)
fig.savefig('evolution.png')
```

### Interactive Exploration

```python
from puffin.nlp import LDAModel, prepare_pyldavis
import pyLDAvis

model = LDAModel(use_gensim=True)
model.fit(documents, n_topics=5)

vis_data = prepare_pyldavis(model)
pyLDAvis.display(vis_data)
```

### Label and Summarize Topics

```python
analyzer = EarningsTopicAnalyzer(n_topics=5)
results = analyzer.analyze(transcripts)

# Assign labels
analyzer.label_topics({
    0: "Financial Performance",
    1: "Technology Innovation",
    2: "Market Conditions",
    3: "Operational Efficiency",
    4: "Strategic Initiatives"
})

# Get summary
summary = analyzer.get_topic_summary(0)
print(f"{summary['label']}")
print(f"Top words: {summary['top_words'][:5]}")
```

### Compare Time Periods

```python
from puffin.nlp import compare_earnings_topics

# Analyze two periods
early_analyzer = EarningsTopicAnalyzer(n_topics=3)
early_analyzer.analyze(transcripts_2020_2021)

late_analyzer = EarningsTopicAnalyzer(n_topics=3)
late_analyzer.analyze(transcripts_2023_2024)

# Compare
similarity = compare_earnings_topics(early_analyzer, late_analyzer)
print("Topic similarity matrix:")
print(similarity)
```

## API Reference

### LSIModel

- `fit(documents, n_topics=10)` - Fit model to documents
- `transform(documents)` - Transform to topic weights
- `get_topics(n_words=10)` - Get top words per topic
- `explained_variance_ratio()` - Variance explained by topics

### LDAModel

- `fit(documents, n_topics=10, passes=15)` - Fit LDA model
- `transform(documents)` - Get topic distributions
- `get_topics(n_words=10)` - Get top words per topic
- `coherence_score(coherence_type='c_v')` - Calculate coherence
- `perplexity()` - Calculate perplexity

### EarningsTopicAnalyzer

- `analyze(transcripts, dates=None)` - Comprehensive analysis
- `detect_topic_shifts(transcripts, dates, window_size=4, threshold=0.3)` - Find shifts
- `topic_sentiment(transcripts, sentiment_scores=None)` - Sentiment by topic
- `label_topics(labels)` - Assign labels
- `get_topic_summary(topic_id, n_words=10)` - Topic summary

## Best Practices

### Preprocessing

Topic models work best with clean, preprocessed text:

```python
import re

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    return text

documents = [preprocess(doc) for doc in raw_documents]
```

### Choosing Number of Topics

- Start with 5-10 topics for exploration
- Use `find_optimal_topics()` for data-driven selection
- Consider domain knowledge
- Check coherence scores (0.3-0.7 is typically good)

### Model Selection

**Use LSI when:**
- Speed is critical
- Exploratory analysis
- Limited computational resources

**Use LDA when:**
- Need interpretable probabilities
- Building production systems
- Have sufficient data (100+ documents)
- Want coherence metrics

### Handling Small Datasets

With <50 documents:
- Use fewer topics (2-5)
- Prefer LSI for stability
- Consider pooling similar documents

## Testing

Run tests with:

```bash
pytest tests/nlp/test_topic_models.py -v
pytest tests/nlp/test_earnings_topics.py -v
```

## Documentation

Full tutorial available at:
`docs/14-topic-modeling/01-topic-modeling.md`

## Dependencies

**Required:**
- numpy
- scikit-learn

**Optional:**
- gensim (for better LDA)
- matplotlib (for visualization)
- pyldavis (for interactive viz)

All dependencies handle gracefully with informative warnings.

## License

Part of the Puffin algorithmic trading framework.
