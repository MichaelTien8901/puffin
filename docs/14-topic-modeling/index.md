---
layout: default
title: "Part 14: Topic Modeling"
nav_order: 15
has_children: true
permalink: /14-topic-modeling/
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

## Topic Modeling Workflow

```mermaid
flowchart TD
    A[Raw Documents] --> B[Preprocessing]
    B --> C[Tokenization & Cleaning]
    C --> D{Model Selection}
    D -->|Fast / Exploratory| E[LSI Model]
    D -->|Probabilistic| F[LDA Model]
    E --> G[SVD Decomposition]
    F --> H[Dirichlet Sampling]
    G --> I[Topic-Word Matrix]
    H --> I
    I --> J[Document-Topic Weights]
    J --> K[Find Optimal Topics]
    K --> L[Label & Interpret]
    L --> M[Earnings Topic Analysis]
    M --> N[Detect Topic Shifts]
    N --> O[Topic Sentiment]
    O --> P[Trading Signals]

    classDef source fill:#2d5016,stroke:#1a3a1a,color:#e8e0d4
    classDef process fill:#1a3a5c,stroke:#0d2137,color:#e8e0d4
    classDef model fill:#6b2d5b,stroke:#4a1a4a,color:#e8e0d4
    classDef output fill:#8b4513,stroke:#5a2d0a,color:#e8e0d4
    classDef decision fill:#4a4a2d,stroke:#33331a,color:#e8e0d4
    classDef signal fill:#2d6b5b,stroke:#1a4a3a,color:#e8e0d4

    class A,B,C source
    class D decision
    class E,F,G,H model
    class I,J,K,L process
    class M,N,O output
    class P signal

    linkStyle default stroke:#4a5568,stroke-width:2px
```

## Model Selection: LSI vs LDA

| Feature | LSI | LDA |
|---------|-----|-----|
| **Method** | SVD on TF-IDF | Probabilistic generative |
| **Output** | Continuous weights | Probability distributions |
| **Speed** | Fast | Slower (iterative) |
| **Coherence** | No built-in metric | Coherence scoring |
| **Best for** | Exploratory analysis | Production systems |
| **Min documents** | ~20 | ~100 |

{: .note }
> **LSI** is ideal for quick exploratory analysis, while **LDA** excels in production systems where interpretable probability distributions and coherence metrics matter.

## Chapter Contents

This chapter covers two main areas:

1. **[LSI & LDA](01-lsi-lda.html)** -- Core topic modeling algorithms, finding optimal topics, visualization techniques, and interactive exploration with pyLDAvis.

2. **[Earnings Call Topic Analysis](02-earnings-topic-analysis.html)** -- The `EarningsTopicAnalyzer` for specialized earnings call analysis, topic shift detection, topic-level sentiment, and a complete trading workflow.

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

{: .tip }
> **Notebook**: Run the examples interactively in [`nlp_trading.ipynb`](https://github.com/MichaelTien8901/puffin/blob/main/notebooks/nlp_trading.ipynb)

## Related Chapters

- [Part 13: NLP for Trading]({{ site.baseurl }}/13-nlp-trading/) -- NLP fundamentals provide the tokenization and vectorization foundation that topic models build upon
- [Part 15: Word Embeddings]({{ site.baseurl }}/15-word-embeddings/) -- Word embeddings offer an alternative dense representation compared to topic model bag-of-words features
- [Part 3: Alternative Data]({{ site.baseurl }}/03-alternative-data/) -- Earnings call transcripts and other alternative data sources serve as primary input for topic analysis

## Source Code

Browse the implementation: [`puffin/nlp/`](https://github.com/MichaelTien8901/puffin/tree/main/puffin/nlp)

## Summary

Key takeaways:

- LSI and LDA discover latent topics without supervision
- Topic models reveal strategic themes in earnings calls
- Track topic evolution to detect strategic shifts
- Combine topics with sentiment for richer signals
- Use visualization to interpret and validate topics
- Consider preprocessing, topic count, and model choice carefully

Topic modeling transforms unstructured text into structured signals for algorithmic trading strategies.

## Next Steps

Topic modeling provides powerful insights into financial text. In the next chapters, we'll explore:

- Word embeddings for semantic similarity
- Deep learning approaches to NLP
- Combining topics with price data
- Real-time topic tracking systems
