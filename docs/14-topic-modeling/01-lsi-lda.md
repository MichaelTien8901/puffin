---
layout: default
title: "LSI & LDA"
parent: "Part 14: Topic Modeling"
nav_order: 1
---

# LSI & LDA Topic Models

This page covers the two core topic modeling algorithms available in Puffin: Latent Semantic Indexing (LSI) and Latent Dirichlet Allocation (LDA). We also cover finding the optimal number of topics, visualization utilities, and interactive exploration with pyLDAvis.

## LSI (Latent Semantic Indexing)

LSI uses Singular Value Decomposition (SVD) on a TF-IDF matrix to discover latent semantic structure in text.

### Basic LSI Example

```python
from puffin.nlp import LSIModel

# Sample earnings call excerpts
documents = [
    "revenue growth exceeded expectations strong quarter improved margins",
    "digital transformation cloud computing technology investment innovation",
    "market conditions challenging headwinds competitive pressure",
    "customer acquisition retention loyalty engagement metrics improved",
    "earnings beat analyst estimates guidance raised outlook positive"
]

# Fit LSI model
model = LSIModel()
model.fit(documents, n_topics=3)

# Get discovered topics
topics = model.get_topics(n_words=5)

for topic_id, words in topics:
    print(f"Topic {topic_id}:")
    for word, weight in words:
        print(f"  {word}: {weight:.3f}")
```

### Transform Documents to Topic Weights

```python
# Transform documents to topic space
topic_weights = model.transform(documents)

print(f"Shape: {topic_weights.shape}")  # (5, 3)

# Analyze specific document
doc_weights = topic_weights[0]
dominant_topic = doc_weights.argmax()
print(f"Document 1 dominant topic: {dominant_topic}")
print(f"Topic weights: {doc_weights}")
```

### Explained Variance

LSI provides explained variance ratios showing how much information each topic captures:

```python
variance = model.explained_variance_ratio()

print("Explained variance by topic:")
for i, var in enumerate(variance):
    print(f"  Topic {i}: {var:.2%}")

print(f"Total variance explained: {variance.sum():.2%}")
```

{: .note }
> LSI topics are ordered by explained variance, so the first topic captures the most information. This makes LSI useful for dimensionality reduction as well as topic discovery.

## LDA (Latent Dirichlet Allocation)

LDA is a probabilistic topic model that represents documents as mixtures of topics and topics as mixtures of words.

### Basic LDA with Gensim

```python
from puffin.nlp import LDAModel

# Prepare documents
documents = [
    "revenue growth profit margin operating income increased",
    "cloud computing digital transformation technology platform",
    "supply chain logistics inventory management efficiency",
    "market share competitive advantage industry leadership",
    "customer satisfaction retention loyalty engagement"
]

# Fit LDA model (uses Gensim by default)
model = LDAModel(use_gensim=True)
model.fit(documents, n_topics=3, passes=15)

# Get topics
topics = model.get_topics(n_words=5)

for topic_id, words in topics:
    print(f"\nTopic {topic_id}:")
    for word, weight in words:
        print(f"  {word}: {weight:.4f}")
```

### Topic Distributions

LDA produces probability distributions over topics for each document:

```python
# Get topic distributions
distributions = model.transform(documents)

# Each row sums to 1.0
print(f"Distribution shape: {distributions.shape}")
print(f"Row sums: {distributions.sum(axis=1)}")

# Analyze document-topic relationships
for i, dist in enumerate(distributions):
    dominant = dist.argmax()
    print(f"Doc {i}: Topic {dominant} ({dist[dominant]:.2%})")
```

### Topic Coherence

Coherence measures how interpretable topics are:

```python
# Calculate coherence score (requires Gensim)
coherence = model.coherence_score()
print(f"Coherence score: {coherence:.4f}")

# Higher is better (typically 0.3-0.7 is good)
```

{: .important }
> Coherence scores are only available when using the Gensim backend (`use_gensim=True`). The sklearn fallback does not support coherence scoring.

### Sklearn Fallback

When Gensim isn't available, Puffin falls back to sklearn:

```python
# Use sklearn explicitly
model = LDAModel(use_gensim=False)
model.fit(documents, n_topics=3)

# Same interface
topics = model.get_topics(n_words=5)
distributions = model.transform(documents)
```

## Finding Optimal Number of Topics

Use coherence or variance to find the optimal number of topics:

```python
from puffin.nlp import find_optimal_topics

# Load earnings transcripts
transcripts = load_earnings_calls()

# Find optimal topics for LDA
optimal_n, scores = find_optimal_topics(
    transcripts,
    min_topics=2,
    max_topics=20,
    step=2,
    method='lda'
)

print(f"Optimal number of topics: {optimal_n}")

# Plot coherence scores
from puffin.nlp import plot_coherence_scores

fig = plot_coherence_scores(
    scores,
    title='LDA Coherence vs Number of Topics'
)
fig.savefig('coherence_scores.png')
```

{: .note }
> **Guidelines for selecting topic count:** Start with 5-10 topics for initial exploration. Use `find_optimal_topics()` for data-driven selection. Consider domain knowledge (e.g., known strategic themes). Examine topic coherence and interpretability. Look for topic overlap or redundancy.

## Topic Visualization

### Topic Distribution for a Document

```python
from puffin.nlp import plot_topic_distribution

# Analyze a specific earnings call
doc = "strong revenue growth margins improved profitability exceeded expectations"

fig = plot_topic_distribution(
    model,
    doc,
    title='Q1 2024 Earnings Call - Topic Distribution'
)
fig.savefig('topic_distribution.png')
```

### Topic Evolution Over Time

```python
from puffin.nlp import plot_topic_evolution
import pandas as pd

# Load quarterly earnings calls with dates
df = pd.read_csv('earnings_history.csv')

fig = plot_topic_evolution(
    model,
    documents=df['transcript'].tolist(),
    dates=df['date'].tolist(),
    topic_ids=[0, 1, 2],  # Track specific topics
    rolling_window=3,  # Smooth with 3-quarter average
    title='Topic Evolution: AAPL 2020-2024'
)
fig.savefig('topic_evolution.png')
```

### Document-Topic Heatmap

```python
from puffin.nlp import plot_topic_heatmap

fig = plot_topic_heatmap(
    model,
    documents=transcripts,
    labels=['Q1', 'Q2', 'Q3', 'Q4'],
    title='Quarterly Earnings Topics'
)
fig.savefig('topic_heatmap.png')
```

### Top Words for a Topic

```python
from puffin.nlp import plot_topic_words

fig = plot_topic_words(
    model,
    topic_id=0,
    n_words=15,
    title='Topic 0: Financial Performance'
)
fig.savefig('topic_words.png')
```

## Interactive Visualization with pyLDAvis

pyLDAvis creates interactive visualizations for exploring LDA topics:

```python
from puffin.nlp import prepare_pyldavis, save_pyldavis_html

# Prepare visualization (requires Gensim-based LDA)
vis_data = prepare_pyldavis(model)

if vis_data:
    # Save to HTML file
    save_pyldavis_html(vis_data, 'lda_visualization.html')

    # Or display in Jupyter
    import pyLDAvis
    pyLDAvis.display(vis_data)
```

The interactive visualization allows you to:
- Explore topics in 2D space
- See top words for each topic
- Adjust relevance metric (lambda parameter)
- Examine topic prevalence and relationships

## Source Code

Browse the implementation: [`puffin/nlp/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/nlp)
