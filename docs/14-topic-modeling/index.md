---
layout: default
title: "Part 14: Topic Modeling"
nav_order: 15
---

# Topic Modeling for Financial Text

Topic modeling is an unsupervised machine learning technique that discovers latent themes in large collections of documents. In algorithmic trading, topic modeling helps analyze earnings calls, financial news, and analyst reports to identify key themes and track how they evolve over time.

## Introduction to Topic Modeling

Topic modeling algorithms automatically identify topics (clusters of related words) in a corpus of documents. Unlike keyword searches or rule-based approaches, topic models discover patterns without prior knowledge of what topics exist.

**Key Applications in Trading:**
- Analyzing earnings call transcripts to identify strategic themes
- Tracking topic evolution across multiple quarters
- Comparing topics across different companies or sectors
- Detecting shifts in corporate strategy or market focus
- Sentiment analysis at the topic level

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
- Adjust relevance metric (λ parameter)
- Examine topic prevalence and relationships

## Earnings Call Topic Analysis

The `EarningsTopicAnalyzer` provides specialized tools for analyzing earnings calls:

```python
from puffin.nlp import EarningsTopicAnalyzer

# Load earnings transcripts
transcripts = [
    "Q1 2023: revenue growth margin expansion strong demand",
    "Q2 2023: digital transformation cloud migration progress",
    "Q3 2023: supply chain improvements cost optimization",
    "Q4 2023: market headwinds competitive pressure challenges",
    "Q1 2024: innovation pipeline new products customer growth",
]

dates = ['2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31', '2024-03-31']

# Create analyzer
analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lda')

# Analyze transcripts
results = analyzer.analyze(transcripts, dates=dates)

# Examine results
print(f"Discovered {len(results['topics'])} topics")
print(f"Coherence score: {results.get('coherence_score', 'N/A')}")

# Dominant topic per call
for i, topic_id in enumerate(results['dominant_topics']):
    print(f"{dates[i]}: Topic {topic_id}")
```

### Labeling Topics

Assign human-readable labels after examining top words:

```python
# Review topics
topics = results['topics']
for topic_id, words in topics:
    print(f"\nTopic {topic_id}:")
    print([word for word, _ in words[:5]])

# Assign labels
analyzer.label_topics({
    0: "Financial Performance",
    1: "Digital Transformation",
    2: "Operational Challenges"
})

# Get labeled summary
summary = analyzer.get_topic_summary(0)
print(f"Topic: {summary['label']}")
print(f"Top words: {summary['top_words'][:5]}")
```

### Detecting Topic Shifts

Identify when companies shift strategic focus:

```python
# Detect topic shifts over time
shifts = analyzer.detect_topic_shifts(
    transcripts,
    dates,
    window_size=3,  # Compare 3-quarter windows
    threshold=0.3    # Minimum change magnitude
)

for shift in shifts:
    print(f"\nShift detected on {shift['date']}:")
    print(f"  From Topic {shift['old_topic']} to Topic {shift['new_topic']}")
    print(f"  Magnitude: {shift['change_magnitude']:.3f}")
```

### Topic-Level Sentiment

Analyze sentiment associated with each topic:

```python
# Provide sentiment scores (or let analyzer compute them)
sentiment_scores = [0.8, 0.6, 0.3, -0.2, 0.7]

sentiment_map = analyzer.topic_sentiment(
    transcripts,
    sentiment_scores
)

for topic_id, stats in sentiment_map.items():
    label = analyzer.topic_labels.get(topic_id, f"Topic {topic_id}")
    print(f"\n{label}:")
    print(f"  Average sentiment: {stats['avg_sentiment']:.3f}")
    print(f"  Weighted sentiment: {stats['weighted_sentiment']:.3f}")
    print(f"  Document count: {stats['document_count']}")
```

### Temporal Trends

Examine how topics evolve:

```python
trends = results['temporal_trends']

for topic_id, trend_data in trends.items():
    label = analyzer.topic_labels.get(topic_id, f"Topic {topic_id}")
    print(f"\n{label}:")
    print(f"  Trend: {trend_data['trend_direction']}")
    print(f"  Avg weight: {trend_data['avg_weight']:.3f}")
    print(f"  Range: {trend_data['min_weight']:.3f} - {trend_data['max_weight']:.3f}")
```

## Comparing Topic Models

Compare topics across different time periods or companies:

```python
from puffin.nlp import compare_earnings_topics

# Analyze two different periods
early_analyzer = EarningsTopicAnalyzer(n_topics=3)
early_analyzer.analyze(transcripts_2020_2021)

late_analyzer = EarningsTopicAnalyzer(n_topics=3)
late_analyzer.analyze(transcripts_2023_2024)

# Compare topic similarity
similarity = compare_earnings_topics(
    early_analyzer,
    late_analyzer,
    method='cosine'  # or 'euclidean'
)

print("Topic similarity matrix:")
print(similarity)
```

## Complete Trading Workflow

Here's a complete example of using topic modeling in a trading strategy:

```python
import pandas as pd
from puffin.nlp import EarningsTopicAnalyzer, plot_topic_evolution
from puffin.data import DataFetcher

# 1. Fetch earnings transcripts
fetcher = DataFetcher()
tickers = ['AAPL', 'MSFT', 'GOOGL']

all_transcripts = []
all_dates = []
all_tickers = []

for ticker in tickers:
    transcripts = fetcher.get_earnings_transcripts(ticker, years=2)

    for date, text in transcripts:
        all_transcripts.append(text)
        all_dates.append(date)
        all_tickers.append(ticker)

# 2. Analyze topics
analyzer = EarningsTopicAnalyzer(n_topics=5, model_type='lda')
results = analyzer.analyze(all_transcripts, dates=all_dates)

# 3. Label topics
analyzer.label_topics({
    0: "Revenue Growth",
    1: "AI & Innovation",
    2: "Cost Management",
    3: "Market Expansion",
    4: "Regulatory Environment"
})

# 4. Detect strategic shifts
shifts = analyzer.detect_topic_shifts(
    all_transcripts,
    all_dates,
    window_size=4
)

# 5. Generate trading signals
signals = []

for shift in shifts:
    # Example: Buy on shift to innovation topic
    if shift['new_topic'] == 1:  # AI & Innovation
        signals.append({
            'date': shift['date'],
            'action': 'BUY',
            'reason': 'Shift to innovation focus'
        })

    # Example: Reduce exposure on cost management focus
    if shift['new_topic'] == 2:  # Cost Management
        signals.append({
            'date': shift['date'],
            'action': 'REDUCE',
            'reason': 'Defensive posture detected'
        })

# 6. Combine with sentiment
sentiment_map = analyzer.topic_sentiment(all_transcripts)

for topic_id, stats in sentiment_map.items():
    if stats['weighted_sentiment'] < -0.3:
        label = analyzer.topic_labels[topic_id]
        print(f"Warning: Negative sentiment on {label}")

# 7. Visualize trends
fig = plot_topic_evolution(
    analyzer.model,
    all_transcripts,
    all_dates,
    topic_ids=[0, 1, 4],
    title='Key Topic Trends: Tech Leaders 2022-2024'
)
fig.savefig('tech_topic_trends.png')
```

## Best Practices

### Preprocessing

Good preprocessing is crucial for topic modeling:

```python
import re
from nltk.corpus import stopwords

def preprocess_for_topics(text):
    """Prepare text for topic modeling."""
    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Remove stopwords (optional, models do this internally)
    # stop_words = set(stopwords.words('english'))
    # words = [w for w in text.split() if w not in stop_words]
    # text = ' '.join(words)

    return text

# Apply preprocessing
clean_transcripts = [preprocess_for_topics(t) for t in transcripts]
```

### Choosing Number of Topics

Guidelines for selecting topic count:

1. Start with 5-10 topics for initial exploration
2. Use `find_optimal_topics()` for data-driven selection
3. Consider domain knowledge (e.g., known strategic themes)
4. Examine topic coherence and interpretability
5. Look for topic overlap or redundancy

### Model Selection: LSI vs LDA

**Use LSI when:**
- You want fast results
- Topics don't need to be probability distributions
- You're doing exploratory analysis
- Computational resources are limited

**Use LDA when:**
- You need interpretable probability distributions
- You want to measure topic coherence
- You're building production systems
- You have sufficient data (100+ documents)

### Handling Small Datasets

With fewer than 50 documents:

```python
# Use fewer topics
analyzer = EarningsTopicAnalyzer(n_topics=3)

# Use LSI for stability
analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')

# Consider pooling data across similar companies
all_tech_transcripts = []
for ticker in tech_tickers:
    transcripts = fetch_transcripts(ticker)
    all_tech_transcripts.extend(transcripts)
```

## Performance Considerations

### Parallel Processing with LDA

Gensim's LdaMulticore uses multiple CPU cores:

```python
# Automatically uses multiple cores
model = LDAModel(use_gensim=True)
model.fit(documents, n_topics=10, passes=15)
```

### Caching Results

Cache fitted models to avoid retraining:

```python
import pickle

# Save model
with open('earnings_lda_model.pkl', 'wb') as f:
    pickle.dump(analyzer, f)

# Load model
with open('earnings_lda_model.pkl', 'rb') as f:
    analyzer = pickle.load(f)

# Use cached model
new_weights = analyzer.model.transform(new_documents)
```

### Batch Processing

Process large document sets efficiently:

```python
batch_size = 100

for i in range(0, len(all_transcripts), batch_size):
    batch = all_transcripts[i:i+batch_size]

    # Process batch
    weights = model.transform(batch)

    # Store or analyze results
    process_batch_results(weights)
```

## Next Steps

Topic modeling provides powerful insights into financial text. In the next chapters, we'll explore:

- Word embeddings for semantic similarity
- Deep learning approaches to NLP
- Combining topics with price data
- Real-time topic tracking systems

## Summary

Key takeaways:

- LSI and LDA discover latent topics without supervision
- Topic models reveal strategic themes in earnings calls
- Track topic evolution to detect strategic shifts
- Combine topics with sentiment for richer signals
- Use visualization to interpret and validate topics
- Consider preprocessing, topic count, and model choice carefully

Topic modeling transforms unstructured text into structured signals for algorithmic trading strategies.
