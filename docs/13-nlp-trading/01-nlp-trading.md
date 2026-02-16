---
layout: default
title: "Chapter 1: NLP for Trading"
parent: "Part 13: NLP for Trading"
nav_order: 1
---

# NLP for Trading

## Overview

Natural Language Processing (NLP) extracts signals from unstructured text like news articles, earnings call transcripts, SEC filings, and social media. Unlike numerical data, text requires specialized techniques for tokenization, feature extraction, and sentiment analysis.

In trading, NLP helps with:

- **Sentiment analysis**: Gauge market mood from news and social media
- **Event detection**: Identify significant corporate events from filings
- **Topic modeling**: Track evolving themes in earnings calls
- **Classification**: Categorize news as bullish, bearish, or neutral
- **Named entity recognition**: Extract company names, money amounts, dates

The Puffin NLP module provides financial-specific tools tuned for trading applications, not general-purpose text analysis.

## Text Preprocessing with NLPPipeline

The `NLPPipeline` class handles text processing using spaCy (with automatic fallback if unavailable).

### Basic Usage

```python
from puffin.nlp import NLPPipeline

# Initialize pipeline
pipeline = NLPPipeline()

# Process text
text = "Apple Inc. reported Q3 earnings of $19.4B, up 8% YoY."
doc = pipeline.process(text)

# Access linguistic features
print(doc.tokens)
# ['Apple', 'Inc.', 'reported', 'Q3', 'earnings', ...]

print(doc.lemmas)
# ['apple', 'inc.', 'report', 'q3', 'earnings', ...]

print(doc.entities)
# [('Apple Inc.', 'ORG'), ('Q3', 'DATE'), ('$19.4B', 'MONEY'), ('8%', 'PERCENT')]

print(doc.sentences)
# ['Apple Inc. reported Q3 earnings of $19.4B, up 8% YoY.']
```

### Named Entity Recognition for Finance

Extract entities relevant to financial analysis:

```python
# Focus on ORG, MONEY, PERCENT, DATE entities
entities = pipeline.extract_entities(text)

for entity_text, entity_label in entities:
    print(f"{entity_label}: {entity_text}")
# ORG: Apple Inc.
# DATE: Q3
# MONEY: $19.4B
# PERCENT: 8%
```

### Extract Financial Terms

Identify trading and finance keywords:

```python
text = "The stock rallied on strong earnings, with analysts raising price targets."
terms = pipeline.extract_financial_terms(text)

print(terms)
# ['stock', 'rallied', 'earnings', 'analysts']
```

### Batch Processing

Process multiple documents efficiently:

```python
texts = [
    "Tesla shares gained 5% on delivery beat.",
    "Microsoft cloud revenue grew 20% YoY.",
    "Amazon acquired robotics startup.",
]

docs = pipeline.batch_process(texts)

for doc in docs:
    print(f"Tokens: {len(doc.tokens)}, Entities: {len(doc.entities)}")
```

## Bag-of-Words and TF-IDF

Convert text to numerical features for machine learning.

### Bag-of-Words

Count term frequencies:

```python
from puffin.nlp import build_bow

documents = [
    "Stock prices increased on strong earnings.",
    "Earnings beat expectations, stock rallied.",
    "Market volatility increased after announcement.",
]

# Build BOW matrix
matrix, features = build_bow(documents, max_features=100)

print(f"Matrix shape: {matrix.shape}")
# (3, 50)  # 3 documents, 50 features

print(f"Features: {features[:10]}")
# ['after', 'announcement', 'beat', 'earnings', 'expectations', ...]
```

### TF-IDF with N-grams

Weight terms by importance using TF-IDF (Term Frequency - Inverse Document Frequency):

```python
from puffin.nlp import build_tfidf

# Build TF-IDF with unigrams and bigrams
matrix, features = build_tfidf(
    documents,
    max_features=100,
    ngram_range=(1, 2)  # Include both single words and pairs
)

print(features[:15])
# ['after', 'announcement', 'beat', 'beat expectations',
#  'earnings', 'earnings beat', 'increased', 'market', ...]
```

### DocumentTermMatrix Class

More flexible interface for vectorization:

```python
from puffin.nlp import DocumentTermMatrix

# Initialize with TF-IDF
dtm = DocumentTermMatrix(
    method="tfidf",
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,  # Term must appear in at least 2 docs
    max_df=0.8  # Term can't appear in more than 80% of docs
)

# Fit and transform
matrix = dtm.fit_transform(documents)

# Transform new documents
new_docs = ["Stock volatility remains elevated."]
new_matrix = dtm.transform(new_docs)

# Get top terms across all documents
top_terms = dtm.get_top_terms(n=10)
for term, weight in top_terms:
    print(f"{term}: {weight:.3f}")

# Get top terms for a specific document
doc_terms = dtm.get_top_terms(n=5, doc_idx=0)
```

## News Classification with Naive Bayes

Classify news articles as bullish, bearish, or neutral.

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

See which words drive predictions:

```python
# Get top features for each class
importance = clf.get_feature_importance(n=10)

for class_label, features in importance.items():
    print(f"\n{class_label.upper()}:")
    for feature, weight in features:
        print(f"  {feature}: {weight:.3f}")
```

### Explaining Predictions

Understand why a specific prediction was made:

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

## Financial Sentiment Analysis

Lexicon-based sentiment tuned for financial text.

### Rule-Based Sentiment with Loughran-McDonald

The Loughran-McDonald lexicon is designed for financial text, avoiding issues with general-purpose sentiment dictionaries (e.g., "liability" is negative in finance but neutral in general use).

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

Add domain-specific words:

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

For more control, use `LexiconSentiment` with word weights:

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

Combine multiple NLP techniques:

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

### 1. Financial-Specific Preprocessing

```python
# Use financial stopwords
financial_stopwords = {
    'inc', 'corp', 'ltd', 'co', 'llc',  # Company suffixes
    'said', 'according', 'reported',     # Common verbs
}

# Keep domain terms that general stopwords remove
keep_words = {
    'up', 'down', 'above', 'below',     # Direction words
    'more', 'less', 'most', 'least',    # Comparison
}
```

### 2. Combine Signals

Don't rely on single methods:

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

### 3. Time Decay

Weight recent news more heavily:

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

### 4. Entity-Specific Analysis

Analyze sentiment per company:

```python
def entity_sentiment(articles):
    entity_scores = {}

    for article in articles:
        # Extract organizations
        entities = pipeline.extract_entities(article)
        orgs = [e[0] for e in entities if e[1] == 'ORG']

        # Compute sentiment
        score = sentiment.score(article)

        # Attribute to each mentioned org
        for org in orgs:
            if org not in entity_scores:
                entity_scores[org] = []
            entity_scores[org].append(score)

    # Average per entity
    return {
        org: np.mean(scores)
        for org, scores in entity_scores.items()
    }
```

## Limitations

### Sarcasm and Context

Lexicon methods can't detect sarcasm or nuanced context:

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

## Summary

- **NLPPipeline**: Tokenization, NER, financial term extraction
- **Vectorizers**: Convert text to BOW or TF-IDF features
- **NewsClassifier**: Naive Bayes for bullish/bearish/neutral classification
- **RuleSentiment**: Loughran-McDonald lexicon for financial sentiment
- **LexiconSentiment**: Custom weighted sentiment analysis

Combine multiple approaches and validate against market outcomes. Text signals work best as part of a broader feature set, not as sole predictors.

## Next Steps

- **Chapter 2**: Topic modeling with LDA and LSI
- **Chapter 3**: Word embeddings for semantic similarity
- **Chapter 4**: Transformer models for advanced NLP
