---
layout: default
title: "Transformer Embeddings"
parent: "Part 15: Word Embeddings"
nav_order: 3
permalink: /15-word-embeddings/03-transformer-embeddings
---

# Transformer Embeddings

BERT (Bidirectional Encoder Representations from Transformers) and its variants provide contextualized embeddings where the same word receives different representations based on its surrounding context. This is a fundamental advance over static embeddings like Word2Vec and GloVe.

## Using BERT Embeddings

The `TransformerEmbedder` class provides a unified interface for encoding text with transformer models.

```python
from puffin.nlp.transformer_embeddings import TransformerEmbedder

# Initialize embedder
embedder = TransformerEmbedder()

# Encode texts
texts = [
    'The company reported strong quarterly earnings.',
    'Market volatility increased amid economic uncertainty.',
    'Stock prices declined following the announcement.'
]

# Get embeddings (default: DistilBERT)
embeddings = embedder.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")  # (3, 768)

# Calculate similarity
sim = embedder.similarity(texts[0], texts[1])
print(f"Similarity: {sim:.3f}")
```

{: .note }
> The default model is DistilBERT, which is 60% faster than BERT-base while retaining 97% of its language understanding capability. For production systems where latency matters, DistilBERT is an excellent choice.

### Comparing with Traditional Embeddings

Transformer embeddings differ fundamentally from static embeddings in how they handle polysemy and context:

```python
# Traditional Word2Vec embedding
w2v_doc_vec = trainer.document_vector(texts[0].split())

# BERT embedding (contextualized)
bert_embedding = embedder.encode(texts[0])

print(f"Word2Vec dimension: {len(w2v_doc_vec)}")
print(f"BERT dimension: {len(bert_embedding[0])}")

# BERT captures context better
texts_polysemy = [
    'The bank approved the loan.',
    'The river bank was flooded.'
]

# Word2Vec: same "bank" vector in both
# BERT: different "bank" vectors based on context
bert_embs = embedder.encode(texts_polysemy)
```

## FinBERT for Financial Text

FinBERT is BERT pre-trained on financial text, providing better representations for financial documents. It understands domain-specific vocabulary and the nuanced meaning of financial language.

```python
# Use FinBERT (or fallback to DistilBERT)
fin_embeddings = embedder.encode_financial([
    'The company exceeded earnings expectations.',
    'Revenue guidance was lowered for next quarter.',
])
```

{: .important }
> FinBERT is specifically trained on financial communication including analyst reports, earnings calls, and financial news. It significantly outperforms general-purpose BERT on financial sentiment classification and semantic similarity tasks.

### Semantic Search

Use transformer embeddings for semantic search over financial documents:

```python
# Semantic search
corpus = [
    'Strong earnings growth driven by product sales.',
    'Regulatory challenges impacting operations.',
    'Market share gains in key segments.',
    'Cost reduction initiatives underway.',
]

query = 'financial performance improvement'
results = embedder.semantic_search(query, corpus, top_k=3)

for result in results:
    print(f"{result['text'][:50]}... (score: {result['score']:.3f})")
```

Semantic search goes beyond keyword matching. The query "financial performance improvement" will match documents about "earnings growth" and "market share gains" even though they share no words in common, because the transformer understands the semantic relationship.

### Clustering Financial Documents

Group similar documents together using transformer embeddings as features:

```python
# Cluster earnings call transcripts
documents = [
    # ... load earnings transcripts ...
]

labels = embedder.cluster_texts(documents, n_clusters=5)

# Analyze clusters
for i in range(5):
    cluster_docs = [doc for doc, label in zip(documents, labels) if label == i]
    print(f"\nCluster {i} ({len(cluster_docs)} documents):")
    print(cluster_docs[0][:100] + '...')
```

## Practical Applications

### 1. News Similarity Detection

Detect duplicate or semantically similar news articles to avoid acting on the same information twice:

```python
# Detect duplicate or similar news articles
news_articles = [
    'Apple announces new iPhone with advanced features.',
    'Apple unveils latest iPhone model with enhanced capabilities.',
    'Tesla reports record quarterly deliveries.',
]

embeddings = embedder.encode(news_articles)

# Compute pairwise similarities
from scipy.spatial.distance import pdist, squareform
similarities = 1 - squareform(pdist(embeddings, metric='cosine'))

print("Similarity matrix:")
print(similarities)

# Articles 0 and 1 are highly similar (duplicates)
```

{: .note }
> In a live trading system, news deduplication prevents overreacting to the same event reported by multiple outlets. A similarity threshold of 0.85-0.90 typically works well for identifying duplicate articles.

### 2. Topic-Based Portfolio Selection

Use embeddings to find stocks whose business descriptions align with a specific investment theme:

```python
# Find stocks related to specific themes
theme = "artificial intelligence and machine learning"
company_descriptions = {
    'AAPL': 'Consumer electronics and software services',
    'NVDA': 'Graphics processing units for AI and gaming',
    'GOOGL': 'Internet services and AI research',
    'XOM': 'Oil and gas exploration and production',
}

from scipy.spatial.distance import cosine

theme_emb = embedder.encode(theme)[0]

scores = {}
for ticker, description in company_descriptions.items():
    desc_emb = embedder.encode(description)[0]
    score = 1 - cosine(theme_emb, desc_emb)
    scores[ticker] = score

# Rank by relevance
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print("AI-related stocks:")
for ticker, score in ranked:
    print(f"{ticker}: {score:.3f}")

# Output:
# NVDA: 0.782
# GOOGL: 0.745
# AAPL: 0.621
# XOM: 0.234
```

### 3. Earnings Call Sentiment Tracking

Track sentiment changes across quarterly earnings calls by measuring embedding distance to positive and negative reference texts:

```python
# Track sentiment changes across earnings calls
calls = {
    'Q1 2023': 'Strong performance across all segments...',
    'Q2 2023': 'Challenging macroeconomic environment...',
    'Q3 2023': 'Improved outlook and growth momentum...',
    'Q4 2023': 'Record revenue and expanding margins...',
}

positive_ref = "excellent results, strong growth, exceeded expectations"
negative_ref = "challenges, headwinds, below expectations"

pos_emb = embedder.encode(positive_ref)[0]
neg_emb = embedder.encode(negative_ref)[0]

for quarter, text in calls.items():
    text_emb = embedder.encode(text)[0]

    pos_sim = 1 - cosine(text_emb, pos_emb)
    neg_sim = 1 - cosine(text_emb, neg_emb)

    sentiment = pos_sim - neg_sim
    print(f"{quarter}: {sentiment:+.3f}")
```

{: .warning }
> Embedding-based sentiment tracking is a complement to, not a replacement for, dedicated sentiment models like FinBERT's sentiment classifier. The reference-text approach shown here is useful for rapid prototyping, but production systems should use fine-tuned sentiment models for higher accuracy.

## Best Practices

### 1. Choosing Embedding Dimensions

Different embedding methods produce different dimensionalities:

- **Word2Vec/GloVe**: 100-300 dimensions (100 for small datasets, 300 for large)
- **Doc2Vec**: 50-200 dimensions
- **BERT**: 768 (base) or 1024 (large) -- fixed by model architecture

### 2. Preprocessing

```python
import re

def preprocess_financial_text(text):
    """Preprocess financial text for embedding."""
    # Lowercase
    text = text.lower()

    # Remove special characters (keep periods for sentence boundaries)
    text = re.sub(r'[^a-z0-9\s\.]', ' ', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Tokenize
    tokens = text.split()

    # Remove stopwords (optional - may hurt for some tasks)
    # stopwords = set(['the', 'a', 'an', 'and', 'or', ...])
    # tokens = [t for t in tokens if t not in stopwords]

    return tokens
```

{: .note }
> For transformer models (BERT, FinBERT), minimal preprocessing is best. These models have their own tokenizers and benefit from seeing punctuation and capitalization. The preprocessing function above is primarily for Word2Vec, GloVe, and Doc2Vec.

### 3. Model Selection

| Task | Recommended Model | Reason |
|------|------------------|---------|
| Word similarity | Word2Vec (Skip-gram) | Good for rare words |
| Document classification | Doc2Vec or BERT | Direct document embeddings |
| Semantic search | BERT/FinBERT | Contextualized representations |
| Financial text | FinBERT | Domain-specific pre-training |
| Large-scale | GloVe | Efficient, good performance |

### 4. Evaluation

Evaluate embedding quality by comparing model similarities against human judgments:

```python
# Evaluate embeddings on word similarity
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

# Human similarity ratings (0-10)
word_pairs = [
    ('bull', 'bear', 3.2),
    ('stock', 'equity', 9.1),
    ('profit', 'loss', 2.1),
    # ... more pairs ...
]

# Model similarities
model_sims = []
human_sims = []

for w1, w2, human_score in word_pairs:
    vec1 = trainer.word_vector(w1)
    vec2 = trainer.word_vector(w2)
    model_score = 1 - cosine(vec1, vec2)

    model_sims.append(model_score)
    human_sims.append(human_score / 10)  # Normalize to 0-1

# Spearman correlation
correlation, p_value = spearmanr(model_sims, human_sims)
print(f"Correlation with human judgments: {correlation:.3f}")
```

## Source Code

Browse the implementation: [`puffin/nlp/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/nlp)
