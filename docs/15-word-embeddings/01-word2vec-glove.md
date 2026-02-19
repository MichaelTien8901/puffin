---
layout: default
title: "Word2Vec & GloVe"
parent: "Part 15: Word Embeddings"
nav_order: 1
permalink: /15-word-embeddings/01-word2vec-glove
---

# Word2Vec & GloVe

Static word embedding methods learn fixed vector representations for each word in a vocabulary. Word2Vec uses local context windows while GloVe leverages global co-occurrence statistics. Both produce dense, low-dimensional vectors that capture semantic relationships between words.

## Word2Vec

Word2Vec learns word embeddings by training a shallow neural network to predict words from their context (CBOW) or context from words (Skip-gram).

### Skip-Gram Model

The Skip-gram model predicts context words given a target word. It works well for rare words and larger datasets.

**Training objective**: Maximize probability of context words given target word:

$$P(w_{context} | w_{target}) = \frac{\exp(v_{context}^T v_{target})}{\sum_{w \in V} \exp(v_w^T v_{target})}$$

{: .note }
> Skip-gram tends to outperform CBOW on smaller datasets and with rare words, making it a good default choice for financial corpora where domain-specific terms may appear infrequently.

### CBOW Model

Continuous Bag of Words (CBOW) predicts a target word from its context words. It is faster to train and works well for frequent words.

### Training Word2Vec with Gensim

The `Word2VecTrainer` class wraps Gensim's Word2Vec implementation with convenience methods for financial text analysis.

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

{: .warning }
> Averaging word vectors to produce document embeddings is a simple baseline but can lose important information about word order and emphasis. For more robust document representations, see the [Doc2Vec](02-doc2vec-sec-filings) section.

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

Word analogies demonstrate that embeddings capture meaningful linear relationships. In financial contexts, this enables discovery of related concepts (e.g., finding that "inflation" relates to "rates" similarly to how "growth" relates to "earnings").

## GloVe: Global Vectors

GloVe learns embeddings by factorizing the word co-occurrence matrix, capturing global corpus statistics rather than relying on local context windows alone.

### Loading Pre-trained GloVe

The `GloVeLoader` class provides methods for loading pre-trained GloVe vectors and using them for financial text analysis.

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

{: .note }
> Pre-trained GloVe vectors are available in several sizes: 50d, 100d, 200d, and 300d. For financial applications, 100d or 300d provides a good balance between quality and computational cost.

### GloVe vs Word2Vec

Understanding when to choose each method is important for building effective NLP pipelines.

**GloVe advantages:**
- Captures global co-occurrence statistics
- More efficient on large corpora
- Better performance on word analogy tasks

**Word2Vec advantages:**
- Better at capturing local context
- Online learning (can update with new data)
- Better for rare words (Skip-gram)

| Criterion | Word2Vec | GloVe |
|-----------|----------|-------|
| Training data | Local context windows | Global co-occurrence matrix |
| Incremental updates | Yes (online learning) | No (requires retraining) |
| Rare words | Better (Skip-gram) | Weaker |
| Training speed | Fast | Fast (matrix factorization) |
| Memory usage | Lower | Higher (stores full matrix) |

For financial applications where the corpus changes frequently (e.g., daily news), Word2Vec's online learning capability is valuable. For static corpora like historical filings, GloVe's global statistics may produce higher quality embeddings.

## Source Code

Browse the implementation: [`puffin/nlp/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/nlp)
