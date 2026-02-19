---
layout: default
title: "Bag-of-Words & TF-IDF"
parent: "Part 13: NLP for Trading"
nav_order: 2
permalink: /13-nlp-trading/02-bag-of-words-tfidf
---

# Bag-of-Words & TF-IDF

Once text has been preprocessed with `NLPPipeline`, the next step is converting it into numerical features that machine learning models can consume. The two foundational approaches are Bag-of-Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF). Puffin provides `build_bow`, `build_tfidf`, and the more flexible `DocumentTermMatrix` class for this purpose.

## Bag-of-Words

Bag-of-Words represents each document as a vector of term counts. It ignores word order and grammar, capturing only which words appear and how often.

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

{: .note }
> The `max_features` parameter limits the vocabulary to the N most frequent terms. This prevents the feature matrix from becoming unwieldy when processing large corpora. Start with a reasonable limit (500-5000) and tune based on model performance.

### When to Use BOW

BOW works well when:

- You need a simple, interpretable baseline
- Document length is relatively consistent
- You have a large training set relative to vocabulary size
- Word frequency is the primary signal (not word importance)

BOW struggles when:

- Rare but important terms get drowned out by common words
- You need to distinguish between documents that share many common terms but differ in key words

## TF-IDF with N-grams

TF-IDF addresses the main weakness of BOW by weighting terms by their importance. Terms that appear frequently in a single document but rarely across the corpus receive higher weights, while terms that appear everywhere (like "the" or "company") get down-weighted.

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

### N-gram Ranges

The `ngram_range` parameter controls which word combinations to include:

| Range | Description | Example Features |
|-------|-------------|-----------------|
| `(1, 1)` | Unigrams only | "earnings", "beat", "strong" |
| `(1, 2)` | Unigrams + bigrams | "earnings", "earnings beat", "strong earnings" |
| `(1, 3)` | Up to trigrams | "earnings", "earnings beat", "earnings beat expectations" |

{: .important }
> Bigrams capture valuable two-word phrases like "earnings beat," "price target," and "revenue growth" that carry more specific meaning than individual words. However, higher n-grams dramatically increase the feature space. For financial text, `(1, 2)` is usually the best trade-off.

## DocumentTermMatrix Class

For more control over vectorization, use the `DocumentTermMatrix` class. It provides a scikit-learn-compatible interface with fit/transform semantics, making it easy to integrate into ML pipelines.

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

### Key Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `method` | Vectorization method (`"bow"` or `"tfidf"`) | `"tfidf"` for most tasks |
| `max_features` | Maximum vocabulary size | 1000-5000 |
| `ngram_range` | Tuple of (min_n, max_n) for n-grams | `(1, 2)` |
| `min_df` | Minimum document frequency (int or float) | 2-5 or 0.01 |
| `max_df` | Maximum document frequency (float) | 0.8-0.95 |

{: .warning }
> The `min_df` and `max_df` parameters are critical for filtering noise. Setting `min_df=2` removes terms that only appear once (likely typos or irrelevant), while `max_df=0.8` removes terms that appear in 80%+ of documents (too common to be informative).

### Fit vs Transform

The `DocumentTermMatrix` follows scikit-learn conventions:

- **`fit_transform(docs)`**: Learn the vocabulary from the training documents and transform them in one step
- **`transform(docs)`**: Transform new documents using the already-learned vocabulary
- **`get_top_terms(n, doc_idx)`**: Inspect the most important terms overall or for a specific document

```python
# Training phase
train_matrix = dtm.fit_transform(train_documents)

# Production phase -- use the same vocabulary
new_matrix = dtm.transform(incoming_articles)
```

{: .note }
> Always call `fit_transform` on your training set and `transform` on new data. Calling `fit_transform` on test or production data would create a different vocabulary, making the features incompatible with your trained model.

## Choosing Between BOW and TF-IDF

For most financial NLP tasks, TF-IDF is the better default:

| Criterion | BOW | TF-IDF |
|-----------|-----|--------|
| Simplicity | Simpler | Slightly more complex |
| Term weighting | Equal weight | Weighted by importance |
| Rare term handling | Under-weighted | Properly elevated |
| Common term handling | Over-weighted | Properly suppressed |
| Financial text | Baseline only | Recommended default |

In practice, TF-IDF with `(1, 2)` n-grams and sensible `min_df`/`max_df` filtering provides a strong baseline for financial text classification and sentiment scoring.

## Integration with Downstream Models

The feature matrices produced by these vectorizers feed directly into classifiers like `NewsClassifier` or custom scikit-learn models:

```python
from puffin.nlp import DocumentTermMatrix, NewsClassifier
from sklearn.linear_model import LogisticRegression

# Vectorize
dtm = DocumentTermMatrix(method="tfidf", max_features=2000, ngram_range=(1, 2))
X_train = dtm.fit_transform(train_texts)
X_test = dtm.transform(test_texts)

# Use with any scikit-learn classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)
predictions = model.predict(X_test)
```

## Source Code

Browse the implementation: [`puffin/nlp/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/nlp)

## Next Steps

Continue to [Sentiment Classification](03-sentiment-classification) to learn how to classify news as bullish, bearish, or neutral, and how to build lexicon-based sentiment scoring for financial text.
