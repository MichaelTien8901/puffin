---
layout: default
title: "Earnings Call Topic Analysis"
parent: "Part 14: Topic Modeling"
nav_order: 2
---

# Earnings Call Topic Analysis

The `EarningsTopicAnalyzer` provides specialized tools for analyzing earnings call transcripts. It builds on the core LSI and LDA models to offer domain-specific features: topic labeling, shift detection, topic-level sentiment, temporal trends, and cross-period comparison.

## EarningsTopicAnalyzer

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

{: .note }
> The `analyze()` method fits the underlying topic model and returns a dictionary containing discovered topics, document-topic weights, dominant topics per document, and temporal trends when dates are provided.

## Labeling Topics

After examining the top words for each discovered topic, assign human-readable labels to make downstream analysis more interpretable:

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

{: .important }
> Always review the top words before labeling. Labels are subjective interpretations -- ensure they genuinely reflect the word clusters the model discovered.

## Detecting Topic Shifts

Identify when companies shift strategic focus between quarters. The shift detector compares topic distributions across sliding windows and flags significant changes:

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

{: .warning }
> A low `threshold` will surface many small shifts, which can generate noisy signals. Start with 0.3 and tune based on your backtest results.

## Topic-Level Sentiment

Analyze sentiment associated with each topic. This combines the topic weights with document-level sentiment scores to produce topic-specific sentiment metrics:

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

The `weighted_sentiment` metric accounts for how strongly each document is associated with the topic, giving more weight to documents where the topic is dominant.

## Temporal Trends

Examine how topics evolve over the time periods covered by your transcripts:

```python
trends = results['temporal_trends']

for topic_id, trend_data in trends.items():
    label = analyzer.topic_labels.get(topic_id, f"Topic {topic_id}")
    print(f"\n{label}:")
    print(f"  Trend: {trend_data['trend_direction']}")
    print(f"  Avg weight: {trend_data['avg_weight']:.3f}")
    print(f"  Range: {trend_data['min_weight']:.3f} - {trend_data['max_weight']:.3f}")
```

## Comparing Topic Models Across Periods

Compare topics across different time periods or companies using cosine or Euclidean similarity:

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

{: .note }
> High similarity between topics from different periods suggests a consistent strategic focus, while low similarity indicates the company has shifted its narrative substantially.

## Complete Trading Workflow

Here is a complete example of using topic modeling in a trading strategy -- from fetching transcripts through generating signals:

```python
import pandas as pd
from puffin.nlp import EarningsTopicAnalyzer, plot_topic_evolution
from puffin.data import DataProvider

# 1. Fetch earnings transcripts
fetcher = DataProvider()
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

{: .note }
> When pooling transcripts across companies, be aware that company-specific jargon may form its own topics. Consider adding company name to your stopwords list to prevent this.

## Source Code

Browse the implementation: [`puffin/nlp/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/nlp)
