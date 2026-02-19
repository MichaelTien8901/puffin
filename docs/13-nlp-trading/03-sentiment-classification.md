---
layout: default
title: "Sentiment Classification"
parent: "Part 13: NLP for Trading"
nav_order: 3
permalink: /13-nlp-trading/03-sentiment-classification
---

# Sentiment Classification

With text preprocessed and vectorized, the final step is classifying sentiment. Puffin provides three complementary approaches: `NewsClassifier` for supervised classification, `RuleSentiment` for Loughran-McDonald lexicon scoring, and `LexiconSentiment` for custom weighted lexicons. Combining multiple methods produces more robust trading signals.

## News Classification with Naive Bayes

The `NewsClassifier` uses a Naive Bayes model to classify news articles as bullish, bearish, or neutral. It handles vectorization internally, so you pass raw text directly.

### Training a Classifier

```python
from puffin.nlp import NewsClassifier

# Training data
train_texts = [
    "Stock surged on strong earnings beat and raised guidance.",
    "Shares rallied after positive analyst upgrade.",
    "Stock plunged on disappointing results.",
    "Shares tumbled after weak outlook.",
    "Company announced routine board meeting.",
    "Stock traded flat in line with market.",
]

train_labels = [
    "bullish", "bullish",
    "bearish", "bearish",
    "neutral", "neutral"
]

# Train classifier
clf = NewsClassifier(
    max_features=5000,
    ngram_range=(1, 2),
    alpha=1.0  # Smoothing parameter
)

clf.fit(train_texts, train_labels)
```

{: .note }
> The `alpha` parameter controls Laplace smoothing. Higher values (e.g., 1.0) prevent zero probabilities for unseen terms but reduce sensitivity. Lower values (e.g., 0.1) are more sensitive but may overfit. Tune this via cross-validation.

### Making Predictions

```python
# Predict labels
test_texts = [
    "Earnings exceeded forecasts, boosting confidence.",
    "Results fell short, disappointing investors."
]

predictions = clf.predict(test_texts)
print(predictions)
# ['bullish', 'bearish']

# Get probabilities
probabilities = clf.predict_proba(test_texts)
print(probabilities)
# [[0.75, 0.15, 0.10],  # bullish, bearish, neutral
#  [0.10, 0.80, 0.10]]
```

### Evaluating Performance

```python
# Evaluate on test set
metrics = clf.evaluate(test_texts, test_labels)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")

print("\nConfusion Matrix:")
print(metrics['confusion_matrix'])

print("\nDetailed Report:")
print(metrics['classification_report'])
```

### Feature Importance

Understanding which words drive predictions is essential for validating that the model is learning meaningful patterns rather than spurious correlations.

```python
# Get top features for each class
importance = clf.get_feature_importance(n=10)

for class_label, features in importance.items():
    print(f"\n{class_label.upper()}:")
    for feature, weight in features:
        print(f"  {feature}: {weight:.3f}")
```

### Explaining Predictions

For individual prediction transparency:

```python
text = "Stock jumped on excellent earnings results."
explanation = clf.get_prediction_explanation(text, n_features=5)

print(f"Prediction: {explanation['prediction']}")
print(f"\nProbabilities:")
for label, prob in explanation['probabilities'].items():
    print(f"  {label}: {prob:.3f}")

print(f"\nTop Features:")
for feature, weight in explanation['top_features']:
    print(f"  {feature}: {weight:.3f}")
```

### Saving and Loading

```python
# Save trained model
clf.save("models/news_classifier.pkl")

# Load later
from puffin.nlp import NewsClassifier
clf = NewsClassifier.load("models/news_classifier.pkl")
```

## Rule-Based Sentiment with Loughran-McDonald

The Loughran-McDonald lexicon is designed specifically for financial text, avoiding issues with general-purpose sentiment dictionaries. For example, "liability" is negative in finance but neutral in everyday language.

```python
from puffin.nlp import RuleSentiment

# Initialize with Loughran-McDonald lexicon
sentiment = RuleSentiment()

# Score text (-1 to +1)
text = "Strong earnings beat expectations, profit margins expanded."
score = sentiment.score(text)
print(f"Sentiment: {score:.3f}")
# 0.125 (positive/bullish)

text2 = "Disappointing results, guidance lowered, losses mounted."
score2 = sentiment.score(text2)
print(f"Sentiment: {score2:.3f}")
# -0.200 (negative/bearish)
```

{: .important }
> General-purpose sentiment tools (like VADER or TextBlob) perform poorly on financial text because they do not account for domain-specific word meanings. Always use a financial lexicon like Loughran-McDonald as your baseline.

### Detailed Sentiment Analysis

```python
text = "Excellent performance despite some challenges and risks."
analysis = sentiment.analyze(text)

print(f"Score: {analysis['score']:.3f}")
print(f"Label: {analysis['label']}")  # bullish, bearish, or neutral
print(f"Positive words: {analysis['positive_words']}")
print(f"Negative words: {analysis['negative_words']}")
print(f"Total words: {analysis['total_words']}")
```

### Batch Sentiment Scoring

```python
texts = [
    "Record profits and strong growth.",
    "Significant losses and declining revenue.",
    "Trading in line with expectations.",
]

scores = sentiment.batch_score(texts)
for text, score in zip(texts, scores):
    label = "bullish" if score > 0.05 else ("bearish" if score < -0.05 else "neutral")
    print(f"{label:8s} ({score:+.3f}): {text}")
```

### Custom Lexicon

Add domain-specific words for specialized use cases:

```python
positive_words = {"mooning", "breakout", "bullrun"}
negative_words = {"dumping", "crash", "capitulation"}

sentiment = RuleSentiment(
    positive_words=positive_words,
    negative_words=negative_words
)

score = sentiment.score("Bitcoin mooning, altcoins following.")
print(score)  # Positive
```

## Weighted Lexicon Sentiment

For more control over how individual words contribute to scores, use `LexiconSentiment` with explicit word weights:

```python
from puffin.nlp import LexiconSentiment

# Words with importance weights
positive_words = {
    "excellent": 3.0,
    "strong": 2.0,
    "good": 1.0,
    "beat": 2.0,
}

negative_words = {
    "terrible": 3.0,
    "weak": 2.0,
    "miss": 2.0,
    "decline": 1.5,
}

sentiment = LexiconSentiment(
    positive_words=positive_words,
    negative_words=negative_words
)

score = sentiment.score("Excellent results beat expectations.")
# Higher score due to weighted words
```

### Dynamic Lexicon Updates

```python
# Start with default Loughran-McDonald
sentiment = LexiconSentiment()

# Add custom financial terms
sentiment.add_positive_word("uplift", weight=1.5)
sentiment.add_positive_word("outperform", weight=2.0)
sentiment.add_negative_word("headwind", weight=1.5)
sentiment.add_negative_word("deteriorate", weight=2.0)

# Use updated lexicon
score = sentiment.score("Business faces headwinds but may outperform.")
```

## Complete Example: News Sentiment Pipeline

Combine all three approaches into a single pipeline that processes news articles end-to-end:

```python
from puffin.nlp import NLPPipeline, NewsClassifier, RuleSentiment
import pandas as pd

# Initialize components
pipeline = NLPPipeline()
sentiment = RuleSentiment()
classifier = NewsClassifier()

# Assume classifier is pre-trained
# classifier.fit(train_texts, train_labels)

def analyze_news(articles):
    results = []

    for article in articles:
        # Extract entities
        entities = pipeline.extract_entities(article)

        # Compute sentiment
        sent_score = sentiment.score(article)

        # Classify sentiment
        prediction = classifier.predict([article])[0]
        proba = classifier.predict_proba([article])[0]

        # Extract financial terms
        terms = pipeline.extract_financial_terms(article)

        results.append({
            'text': article[:100] + '...',
            'entities': entities,
            'sentiment_score': sent_score,
            'classification': prediction,
            'confidence': max(proba),
            'financial_terms': terms,
        })

    return pd.DataFrame(results)

# Example usage
news = [
    "Apple reported record iPhone sales, beating analyst expectations.",
    "Tesla shares fell on production delays and supply chain issues.",
    "Microsoft announced its quarterly dividend payment.",
]

results_df = analyze_news(news)
print(results_df[['classification', 'sentiment_score', 'confidence']])
```

## Best Practices

### Combine Signals

Don't rely on a single method. Ensemble multiple approaches for more robust scores:

```python
def combined_sentiment(text):
    # Lexicon-based
    lex_score = sentiment.score(text)

    # Classifier-based
    clf_pred = classifier.predict([text])[0]
    clf_score = {"bullish": 1, "neutral": 0, "bearish": -1}[clf_pred]

    # Weighted average
    final_score = 0.6 * lex_score + 0.4 * clf_score

    return final_score
```

### Time Decay

Weight recent news more heavily. Stale sentiment loses predictive power rapidly:

```python
import numpy as np
from datetime import datetime, timedelta

def time_weighted_sentiment(news_df):
    # news_df has columns: timestamp, text
    now = datetime.now()

    scores = []
    for _, row in news_df.iterrows():
        score = sentiment.score(row['text'])

        # Exponential decay: half-life of 7 days
        age_days = (now - row['timestamp']).days
        weight = np.exp(-0.1 * age_days)

        scores.append(score * weight)

    return np.mean(scores)
```

{: .warning }
> Sentiment signals decay quickly. A positive earnings surprise from two weeks ago has much less predictive value than one from today. Always apply time weighting when aggregating sentiment across multiple articles.

## Limitations

### Sarcasm and Context

Lexicon methods cannot detect sarcasm or nuanced context:

```python
# This might score positive, but context suggests otherwise
text = "Great job missing earnings again."
# Manual review or advanced models needed
```

### Domain Shifts

Financial language evolves. Update lexicons regularly:

```python
# Terms gain/lose importance
# "crypto", "ESG", "work-from-home" rose in prominence
# Retrain classifiers periodically
```

### Short Text

Tweets and headlines have limited context:

```python
# Hard to analyze
"AAPL 🚀"

# Better
"Apple stock surges 5% on strong iPhone demand in China."
```

{: .note }
> For short-text analysis (tweets, headlines), consider using character-level or subword models rather than word-level BOW/TF-IDF. Transformer-based models (covered in later chapters) handle short text significantly better.

## Source Code

Browse the implementation: [`puffin/nlp/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/nlp)

## Next Steps

- **Part 14**: Topic modeling with LDA and LSI
- **Part 15**: Word embeddings for semantic similarity
- **Part 16**: Transformer models for advanced NLP
