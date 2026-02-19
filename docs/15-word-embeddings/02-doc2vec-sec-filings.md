---
layout: default
title: "Doc2Vec & SEC Filings"
parent: "Part 15: Word Embeddings"
nav_order: 2
permalink: /15-word-embeddings/02-doc2vec-sec-filings
---

# Doc2Vec & SEC Filings

Doc2Vec extends Word2Vec to learn document-level embeddings directly, rather than averaging word vectors. Combined with the `SECFilingAnalyzer`, these techniques enable powerful analysis of corporate disclosures including filing comparison and language change detection over time.

## Doc2Vec (Paragraph Vectors)

Doc2Vec introduces a document vector that is trained alongside word vectors, producing a fixed-length representation for documents of any length.

### Training Doc2Vec

The `Doc2VecTrainer` class provides a convenient wrapper around Gensim's Doc2Vec implementation.

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

{: .note }
> When inferring vectors for new documents, Doc2Vec runs additional gradient descent steps. Increasing `epochs` during inference (e.g., 50-100 steps) can improve the quality of inferred vectors, especially for short documents.

### PV-DM vs PV-DBOW

Doc2Vec offers two training architectures, each with different strengths.

**PV-DM (Distributed Memory):**
- Concatenates document vector with word vectors
- Similar to CBOW in Word2Vec
- Better for most tasks, especially when word order matters
- Preserves more semantic information

**PV-DBOW (Distributed Bag of Words):**
- Predicts words from document vector only
- Similar to Skip-gram in Word2Vec
- Faster training
- Works well when combined with PV-DM

{: .important }
> For financial document analysis, PV-DM (dm=1) is generally recommended as the default. Le and Mikolov found that combining PV-DM and PV-DBOW vectors often produces the best results. You can concatenate the vectors from both models for downstream tasks.

## SEC Filing Analysis

The `SECFilingAnalyzer` class applies embedding techniques to analyze SEC filings and track corporate language changes over time. This is particularly useful for detecting shifts in risk disclosure, management tone, and business strategy.

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

The `analyze_10k` method extracts key sections (Risk Factors, MD&A, Business Description) and computes embedding-based metrics for each. The `prior_text` argument enables year-over-year comparison.

### Comparing Multiple Filings

Track how corporate disclosures evolve over multiple filing periods:

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

{: .warning }
> A sudden drop in filing similarity (e.g., from 0.92 to 0.75) may indicate a material change in business risk or strategy. However, formatting changes and boilerplate updates can also cause fluctuations. Always combine quantitative metrics with qualitative review.

### Detecting Significant Language Changes

Identify specific vocabulary shifts between filings to surface emerging risks or strategic pivots:

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

The appearance of new risk terms or the disappearance of previously emphasized terms can signal important shifts in a company's operating environment.

### Risk Sentiment Analysis

Extract risk-specific sentiment from filing text using embedding similarity to risk-related seed words:

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

{: .note }
> The risk sentiment score is computed by measuring the average cosine similarity between the filing's vocabulary and the provided risk keywords in embedding space. A higher score indicates more risk-related language throughout the document.

## Source Code

Browse the implementation: [`puffin/nlp/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/nlp)
