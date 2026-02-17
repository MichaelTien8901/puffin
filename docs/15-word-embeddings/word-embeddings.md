---
layout: default
title: "Part 15: Word Embeddings"
nav_order: 16
permalink: /15-word-embeddings/
---

# Word Embeddings

Word embeddings are dense vector representations of words that capture semantic relationships and contextual meaning. Unlike traditional one-hot encodings, embeddings place semantically similar words close together in a continuous vector space, making them powerful tools for financial text analysis.

## Introduction to Word Embeddings

Word embeddings map words from a discrete vocabulary to continuous vectors in a high-dimensional space. Words with similar meanings or contexts have similar vector representations, enabling mathematical operations on language.

**Key Properties:**
- **Semantic similarity**: Related words have similar vectors
- **Compositionality**: Vector arithmetic captures relationships (e.g., "king" - "man" + "woman" ≈ "queen")
- **Dimensionality reduction**: From sparse one-hot vectors to dense representations
- **Transfer learning**: Pre-trained embeddings can be used across tasks

## Word2Vec

Word2Vec learns word embeddings by training a shallow neural network to predict words from their context (CBOW) or context from words (Skip-gram).

### Skip-Gram Model

The Skip-gram model predicts context words given a target word. It works well for rare words and larger datasets.

**Training objective**: Maximize probability of context words given target word:

$$P(w_{context} | w_{target}) = \frac{\exp(v_{context}^T v_{target})}{\sum_{w \in V} \exp(v_w^T v_{target})}$$

### CBOW Model

Continuous Bag of Words (CBOW) predicts a target word from its context words. It's faster and works well for frequent words.

### Training Word2Vec with Gensim

```python
from puffin.nlp.embeddings import Word2VecTrainer
import pandas as pd

# Load and preprocess financial news
data = pd.read_csv('financial_news.csv')

# Tokenize documents
documents = [text.lower().split() for text in data['text']]

# Train Word2Vec
trainer = Word2VecTrainer()
model = trainer.train(
    documents,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window size
    min_count=5,          # Ignore rare words
    sg=1,                 # 1=Skip-gram, 0=CBOW
    workers=4,            # Parallel processing
    epochs=10
)

# Get word vector
market_vec = trainer.word_vector('market')
print(f"Vector shape: {market_vec.shape}")

# Find similar words
similar = trainer.similar_words('volatility', topn=10)
for word, similarity in similar:
    print(f"{word}: {similarity:.3f}")

# Output:
# uncertainty: 0.852
# fluctuation: 0.831
# instability: 0.809
# ...
```

### Document Embeddings

Average word vectors to create document-level embeddings:

```python
# Get document vector
doc = ['market', 'volatility', 'increased', 'significantly']
doc_vec = trainer.document_vector(doc)

# Compare documents
doc1 = ['bull', 'market', 'rising', 'prices']
doc2 = ['bear', 'market', 'falling', 'prices']

vec1 = trainer.document_vector(doc1)
vec2 = trainer.document_vector(doc2)

from scipy.spatial.distance import cosine
similarity = 1 - cosine(vec1, vec2)
print(f"Document similarity: {similarity:.3f}")
```

### Word Analogies

Solve analogies using vector arithmetic:

```python
# king - man + woman = queen
# bull market - bull + bear = bear market

result = trainer.analogy(
    positive=['bull', 'market'],
    negative=['bull'],
    topn=5
)

print(result)
# [('bear', 0.756), ('declining', 0.632), ...]
```

## GloVe: Global Vectors

GloVe learns embeddings by factorizing the word co-occurrence matrix, capturing global corpus statistics.

### Loading Pre-trained GloVe

```python
from puffin.nlp.embeddings import GloVeLoader

# Download GloVe from: https://nlp.stanford.edu/projects/glove/
loader = GloVeLoader()
loader.load('glove.6B.100d.txt')

# Get word vector
vec = loader.word_vector('stock')

# Find similar words
similar = loader.similar_words('earnings', topn=5)

# Document embedding
doc = ['quarterly', 'earnings', 'exceeded', 'expectations']
doc_vec = loader.document_vector(doc)
```

### GloVe vs Word2Vec

**GloVe advantages:**
- Captures global co-occurrence statistics
- More efficient on large corpora
- Better performance on word analogy tasks

**Word2Vec advantages:**
- Better at capturing local context
- Online learning (can update with new data)
- Better for rare words (Skip-gram)

## Doc2Vec (Paragraph Vectors)

Doc2Vec extends Word2Vec to learn document-level embeddings directly, rather than averaging word vectors.

### Training Doc2Vec

```python
from puffin.nlp.embeddings import Doc2VecTrainer

# Prepare documents
documents = [
    ['market', 'volatility', 'increased', 'today'],
    ['stock', 'prices', 'rose', 'sharply'],
    ['bond', 'yields', 'declined', 'slightly'],
]

# Train Doc2Vec
trainer = Doc2VecTrainer()
model = trainer.train(
    documents,
    vector_size=100,
    window=5,
    min_count=2,
    dm=1,              # 1=PV-DM, 0=PV-DBOW
    epochs=20
)

# Infer vector for new document
new_doc = ['equity', 'markets', 'rallied']
doc_vec = trainer.infer_vector(new_doc)

# Find similar documents
similar_docs = trainer.similar_documents(new_doc, topn=3)
for doc_id, similarity in similar_docs:
    print(f"Document {doc_id}: {similarity:.3f}")
```

### PV-DM vs PV-DBOW

**PV-DM (Distributed Memory):**
- Concatenates document vector with word vectors
- Similar to CBOW
- Better for most tasks

**PV-DBOW (Distributed Bag of Words):**
- Predicts words from document vector only
- Similar to Skip-gram
- Faster training

## SEC Filing Analysis

Apply embeddings to analyze SEC filings and track corporate language changes over time.

### Analyzing 10-K Filings

```python
from puffin.nlp.sec_analysis import SECFilingAnalyzer
from puffin.nlp.embeddings import Word2VecTrainer

# Load SEC filings
with open('aapl_10k_2023.txt') as f:
    text_2023 = f.read()

with open('aapl_10k_2022.txt') as f:
    text_2022 = f.read()

# Train embedding model on financial corpus
# (In practice, use pre-trained model or larger corpus)
trainer = Word2VecTrainer()
# ... training code ...

# Analyze filing
analyzer = SECFilingAnalyzer()
result = analyzer.analyze_10k(
    text_2023,
    trainer.model,
    prior_text=text_2022
)

print(f"Similarity to prior year: {result['similarity_to_prior']:.3f}")

# Extract embeddings for key sections
risk_emb = result['risk_factors_embedding']
mda_emb = result['mda_embedding']
```

### Comparing Multiple Filings

```python
# Load multiple years of filings
texts = [text_2021, text_2022, text_2023]
dates = ['2021-12-31', '2022-12-31', '2023-12-31']

# Compare over time
comparison = analyzer.compare_filings(texts, dates, trainer.model)

print(comparison[['date', 'similarity_to_previous', 'change_magnitude']])

# Output:
#         date  similarity_to_previous  change_magnitude
# 0 2021-12-31                    NaN               NaN
# 1 2022-12-31                  0.892             0.108
# 2 2023-12-31                  0.857             0.143

# Visualize changes
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(comparison['date'], comparison['similarity_to_previous'], marker='o')
plt.xlabel('Filing Date')
plt.ylabel('Similarity to Previous Filing')
plt.title('SEC Filing Language Similarity Over Time')
plt.grid(True)
plt.show()
```

### Detecting Significant Changes

```python
# Detect language changes
changes = analyzer.detect_language_changes(
    texts,
    dates,
    threshold=0.15  # 15% vocabulary change
)

for change in changes:
    print(f"\nDate: {change['date']}")
    print(f"Change score: {change['change_score']:.3f}")
    print(f"New terms: {change['new_terms'][:5]}")
    print(f"Removed terms: {change['removed_terms'][:5]}")

# Output:
# Date: 2023-12-31
# Change score: 0.182
# New terms: ['macroeconomic', 'headwinds', 'geopolitical', ...]
# Removed terms: ['pandemic', 'lockdown', 'stimulus', ...]
```

### Risk Sentiment Analysis

```python
# Extract risk-related sentiment
sentiment = analyzer.extract_risk_sentiment(
    text_2023,
    trainer.model,
    risk_keywords=['risk', 'uncertainty', 'volatile', 'litigation']
)

print(f"Risk score: {sentiment['risk_score']:.3f}")
print("\nTop risk terms:")
for term, score in sentiment['top_risk_terms'][:10]:
    print(f"  {term}: {score:.3f}")
```

## BERT and Transformer Embeddings

BERT (Bidirectional Encoder Representations from Transformers) provides contextualized embeddings where the same word has different representations based on context.

### Using BERT Embeddings

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

### FinBERT for Financial Text

FinBERT is BERT pre-trained on financial text, providing better representations for financial documents.

```python
# Use FinBERT (or fallback to DistilBERT)
fin_embeddings = embedder.encode_financial([
    'The company exceeded earnings expectations.',
    'Revenue guidance was lowered for next quarter.',
])

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

### Comparing with Traditional Embeddings

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

### Clustering Financial Documents

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

### 2. Topic-Based Portfolio Selection

```python
# Find stocks related to specific themes
theme = "artificial intelligence and machine learning"
company_descriptions = {
    'AAPL': 'Consumer electronics and software services',
    'NVDA': 'Graphics processing units for AI and gaming',
    'GOOGL': 'Internet services and AI research',
    'XOM': 'Oil and gas exploration and production',
}

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

## Best Practices

### 1. Choosing Embedding Dimensions

- **Word2Vec/GloVe**: 100-300 dimensions (100 for small datasets, 300 for large)
- **Doc2Vec**: 50-200 dimensions
- **BERT**: 768 (base) or 1024 (large) - fixed

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

### 3. Model Selection

| Task | Recommended Model | Reason |
|------|------------------|---------|
| Word similarity | Word2Vec (Skip-gram) | Good for rare words |
| Document classification | Doc2Vec or BERT | Direct document embeddings |
| Semantic search | BERT/FinBERT | Contextualized representations |
| Financial text | FinBERT | Domain-specific pre-training |
| Large-scale | GloVe | Efficient, good performance |

### 4. Evaluation

```python
# Evaluate embeddings on word similarity
from scipy.stats import spearmanr

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

## Summary

Word embeddings provide powerful representations for financial text analysis:

- **Word2Vec**: Learn embeddings from local context (Skip-gram, CBOW)
- **GloVe**: Global co-occurrence statistics
- **Doc2Vec**: Direct document embeddings
- **BERT/FinBERT**: Contextualized embeddings for state-of-the-art performance

Applications include:
- SEC filing analysis and change detection
- News similarity and deduplication
- Theme-based portfolio selection
- Sentiment tracking across documents

Next, we'll explore deep learning models for time series prediction and trading.

## Further Reading

- [Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781)
- [Pennington et al. (2014). "GloVe: Global Vectors for Word Representation"](https://aclanthology.org/D14-1162/)
- Le and Mikolov (2014). "Distributed Representations of Sentences and Documents"
- [Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional Transformers"](https://arxiv.org/abs/1810.04805)
- [Araci (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"](https://arxiv.org/abs/1908.10063)
