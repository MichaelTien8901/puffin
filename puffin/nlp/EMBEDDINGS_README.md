# Word Embeddings Module

This module provides word and document embeddings for financial text analysis, including Word2Vec, GloVe, Doc2Vec, and transformer-based models (BERT/FinBERT).

## Installation

### Core Dependencies

```bash
pip install gensim numpy scipy pandas
```

### For Transformer Embeddings (Optional)

```bash
pip install transformers torch
# Or for CPU-only (smaller install):
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

### For Sentence Transformers (Optional)

```bash
pip install sentence-transformers
```

### All Dependencies

```bash
pip install -r requirements-embeddings.txt
```

## Modules

### 1. embeddings.py

Core embedding models: Word2Vec, GloVe, and Doc2Vec.

**Classes:**
- `Word2VecTrainer`: Train and use Word2Vec embeddings (Skip-gram or CBOW)
- `GloVeLoader`: Load and use pre-trained GloVe vectors
- `Doc2VecTrainer`: Train and use Doc2Vec for document embeddings

**Example:**

```python
from puffin.nlp.embeddings import Word2VecTrainer

# Prepare tokenized documents
documents = [
    ['market', 'volatility', 'increased'],
    ['stock', 'prices', 'rallied'],
]

# Train Word2Vec
trainer = Word2VecTrainer()
model = trainer.train(
    documents,
    vector_size=100,
    window=5,
    min_count=5,
    sg=1  # 1=Skip-gram, 0=CBOW
)

# Get word vector
vector = trainer.word_vector('market')

# Find similar words
similar = trainer.similar_words('market', topn=10)

# Get document vector
doc_vec = trainer.document_vector(['stock', 'market'])
```

### 2. sec_analysis.py

Analyze SEC filings using word embeddings.

**Classes:**
- `SECFilingAnalyzer`: Extract and analyze SEC filing sections, compare filings over time

**Example:**

```python
from puffin.nlp.sec_analysis import SECFilingAnalyzer
from puffin.nlp.embeddings import Word2VecTrainer

# Load SEC filings
with open('10k_2023.txt') as f:
    text_2023 = f.read()

with open('10k_2022.txt') as f:
    text_2022 = f.read()

# Train embedding model (or load pre-trained)
trainer = Word2VecTrainer()
# ... training code ...

# Analyze filing
analyzer = SECFilingAnalyzer()
result = analyzer.analyze_10k(text_2023, trainer.model, prior_text=text_2022)

print(f"Similarity to prior year: {result['similarity_to_prior']:.3f}")

# Compare multiple filings
texts = [text_2021, text_2022, text_2023]
dates = ['2021-12-31', '2022-12-31', '2023-12-31']
comparison = analyzer.compare_filings(texts, dates, trainer.model)

# Detect language changes
changes = analyzer.detect_language_changes(texts, dates, threshold=0.15)
```

### 3. transformer_embeddings.py

BERT, FinBERT, and other transformer-based embeddings.

**Classes:**
- `TransformerEmbedder`: Use BERT/FinBERT for contextualized embeddings
- `SentenceTransformerEmbedder`: Optimized sentence embeddings (requires sentence-transformers)

**Example:**

```python
from puffin.nlp.transformer_embeddings import TransformerEmbedder

# Initialize embedder
embedder = TransformerEmbedder()

# Encode texts
texts = [
    'The company reported strong earnings.',
    'Market volatility increased.',
]

embeddings = embedder.encode(texts)  # Uses DistilBERT by default

# Use FinBERT for financial text
fin_embeddings = embedder.encode_financial(texts)

# Calculate similarity
sim = embedder.similarity(texts[0], texts[1])

# Semantic search
corpus = ['Revenue grew 20%', 'Costs decreased', 'Margins improved']
results = embedder.semantic_search('financial performance', corpus, top_k=2)
```

## Usage Examples

### 1. Financial News Similarity

```python
from puffin.nlp.embeddings import Word2VecTrainer
from scipy.spatial.distance import cosine

# Train on news corpus
trainer = Word2VecTrainer()
trainer.train(news_documents, vector_size=100, sg=1)

# Compare two articles
article1 = ['market', 'volatility', 'increased']
article2 = ['market', 'uncertainty', 'rose']

vec1 = trainer.document_vector(article1)
vec2 = trainer.document_vector(article2)

similarity = 1 - cosine(vec1, vec2)
print(f"Similarity: {similarity:.3f}")
```

### 2. Topic-Based Stock Screening

```python
from puffin.nlp.transformer_embeddings import TransformerEmbedder

embedder = TransformerEmbedder()

# Define investment theme
theme = "artificial intelligence and machine learning"
theme_emb = embedder.encode(theme)[0]

# Company descriptions
companies = {
    'NVDA': 'Graphics processing units for AI and gaming',
    'GOOGL': 'Internet services and AI research',
    'XOM': 'Oil and gas exploration',
}

# Score relevance
scores = {}
for ticker, description in companies.items():
    desc_emb = embedder.encode(description)[0]
    score = 1 - cosine(theme_emb, desc_emb)
    scores[ticker] = score

# Rank by relevance
ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 3. Earnings Call Analysis

```python
from puffin.nlp.embeddings import Doc2VecTrainer

# Train on historical earnings calls
trainer = Doc2VecTrainer()
trainer.train(earnings_transcripts, vector_size=100)

# Compare new call to historical
new_call = ['strong', 'growth', 'momentum', 'expanding', 'markets']
new_vec = trainer.infer_vector(new_call)

# Find similar historical calls
similar = trainer.similar_documents(new_call, topn=5)
```

### 4. SEC Filing Change Detection

```python
from puffin.nlp.sec_analysis import SECFilingAnalyzer

analyzer = SECFilingAnalyzer()

# Track changes over 5 years
filings = load_filings(['2019', '2020', '2021', '2022', '2023'])
dates = ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31']

# Detect significant changes
changes = analyzer.detect_language_changes(
    filings,
    dates,
    threshold=0.20  # 20% vocabulary change
)

for change in changes:
    print(f"{change['date']}: {change['change_score']:.1%} change")
    print(f"New terms: {change['new_terms'][:10]}")
```

## Pre-trained Models

### GloVe

Download pre-trained GloVe vectors:
- [GloVe 6B](https://nlp.stanford.edu/projects/glove/) - 6 billion tokens
- Dimensions: 50d, 100d, 200d, 300d

```python
from puffin.nlp.embeddings import GloVeLoader

loader = GloVeLoader()
loader.load('glove.6B.100d.txt')
```

### FinBERT

Pre-trained BERT for financial text:
- Model: `ProsusAI/finbert` or `yiyanghkust/finbert-tone`
- Automatically downloaded by transformers

```python
from puffin.nlp.transformer_embeddings import TransformerEmbedder

embedder = TransformerEmbedder()
embeddings = embedder.encode(texts, model_name='ProsusAI/finbert')
```

## Best Practices

### 1. Choosing an Embedding Method

| Use Case | Recommended Method | Reason |
|----------|-------------------|---------|
| Word similarity | Word2Vec (Skip-gram) | Good for rare words |
| Document classification | Doc2Vec or BERT | Direct doc embeddings |
| Semantic search | BERT/FinBERT | Contextualized |
| Financial text | FinBERT | Domain-specific |
| Large-scale offline | GloVe | Fast, pre-trained |
| Small dataset | Pre-trained GloVe/BERT | Transfer learning |

### 2. Hyperparameters

**Word2Vec:**
- `vector_size`: 100-300 (100 for small data, 300 for large)
- `window`: 5-10 (5 for syntactic, 10 for semantic)
- `min_count`: 5-10 (filter rare words)
- `sg`: 1 for rare words, 0 for frequent words

**Doc2Vec:**
- `vector_size`: 50-200
- `dm`: 1 (PV-DM) usually better, 0 (PV-DBOW) faster
- `epochs`: 10-50 (more for small datasets)

**BERT:**
- `max_length`: 512 (BERT limit)
- Use `batch_size=32` for GPU, 8-16 for CPU

### 3. Preprocessing

```python
import re

def preprocess_financial_text(text):
    """Preprocess text for embedding."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = ' '.join(text.split())
    return text.split()
```

### 4. Memory Optimization

For large corpora:

```python
# Use generators for Word2Vec
def document_generator(file_path):
    with open(file_path) as f:
        for line in f:
            yield preprocess_financial_text(line)

# Train without loading all data
trainer = Word2VecTrainer()
model = trainer.train(document_generator('large_corpus.txt'))
```

## Performance Considerations

### Model Size

| Model | Parameters | Disk Size | RAM | Speed |
|-------|-----------|-----------|-----|-------|
| Word2Vec | ~10M | ~40 MB | Low | Fast |
| GloVe (pre-trained) | ~400K words | ~500 MB | Medium | Fast |
| Doc2Vec | ~15M | ~60 MB | Low | Fast |
| DistilBERT | 66M | ~250 MB | High | Medium |
| BERT-base | 110M | ~440 MB | High | Slow |
| FinBERT | 110M | ~440 MB | High | Slow |

### Speed Benchmarks

On CPU (Intel i7, 8 cores):
- Word2Vec: ~10K docs/sec (training)
- Doc2Vec: ~5K docs/sec (training)
- GloVe: ~100K lookups/sec
- DistilBERT: ~10 docs/sec (inference)
- BERT: ~5 docs/sec (inference)

On GPU (NVIDIA RTX 3080):
- DistilBERT: ~100 docs/sec (batch=32)
- BERT: ~50 docs/sec (batch=32)

## Testing

Run tests:

```bash
# Test embeddings
pytest tests/nlp/test_embeddings.py -v

# Test SEC analysis
pytest tests/nlp/test_sec_analysis.py -v

# Run all NLP tests
pytest tests/nlp/ -v
```

## Troubleshooting

### Issue: Gensim not found

```bash
pip install gensim
```

### Issue: Transformers/PyTorch not found

```bash
# For GPU
pip install transformers torch

# For CPU only (smaller)
pip install transformers torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Out of memory with BERT

```python
# Reduce batch size
embeddings = embedder.encode(texts, batch_size=8)

# Use DistilBERT (smaller, faster)
embeddings = embedder.encode(texts, model_name='distilbert-base-uncased')

# Process on CPU
embedder = TransformerEmbedder(device='cpu')
```

### Issue: Word not in vocabulary

```python
# Use try-except for Word2Vec
try:
    vector = trainer.word_vector('rare_word')
except KeyError:
    vector = np.zeros(trainer.model.vector_size)

# GloVe returns None for missing words
vector = loader.word_vector('rare_word')
if vector is None:
    vector = np.zeros(loader.vector_size)
```

## Further Reading

- [Word2Vec paper](https://arxiv.org/abs/1301.3781) - Mikolov et al., 2013
- [GloVe paper](https://nlp.stanford.edu/pubs/glove.pdf) - Pennington et al., 2014
- [Doc2Vec paper](https://arxiv.org/abs/1405.4053) - Le and Mikolov, 2014
- [BERT paper](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- [FinBERT paper](https://arxiv.org/abs/1908.10063) - Araci, 2019
- [Gensim documentation](https://radimrehurek.com/gensim/)
- [HuggingFace documentation](https://huggingface.co/docs)

## License

Part of the Puffin algorithmic trading framework.
