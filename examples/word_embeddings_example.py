"""
Example usage of word embeddings for financial text analysis.

This example demonstrates:
1. Training Word2Vec on financial news
2. Using GloVe embeddings
3. Training Doc2Vec for document similarity
4. Analyzing SEC filings
5. Using BERT/FinBERT for modern embeddings
"""

import numpy as np
from pathlib import Path

# Sample financial news documents
FINANCIAL_NEWS = [
    "Market volatility increased today as investors reacted to inflation data",
    "Stock prices rallied on strong earnings reports from major tech companies",
    "Bond yields declined amid concerns about economic growth",
    "The Federal Reserve signaled potential interest rate increases",
    "Oil prices surged following supply disruption news",
    "Technology stocks led market gains in volatile trading session",
    "Central bank policy decisions impact currency markets significantly",
    "Investors seek safe haven assets during market uncertainty",
    "Commodity prices fluctuated on global demand concerns",
    "Corporate earnings exceeded analyst expectations driving market optimism",
]


def example_word2vec():
    """Example: Training and using Word2Vec."""
    print("\n" + "="*60)
    print("Word2Vec Example")
    print("="*60)

    from puffin.nlp.embeddings import Word2VecTrainer

    # Tokenize documents
    documents = [text.lower().split() for text in FINANCIAL_NEWS]

    # Train Word2Vec
    print("\nTraining Word2Vec model...")
    trainer = Word2VecTrainer()
    model = trainer.train(
        documents,
        vector_size=50,
        window=5,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=20
    )

    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Vector dimension: {model.vector_size}")

    # Get word vector
    word = "market"
    vector = trainer.word_vector(word)
    print(f"\nVector for '{word}': {vector[:5]}... (showing first 5 dims)")

    # Find similar words
    print(f"\nWords similar to '{word}':")
    similar = trainer.similar_words(word, topn=5)
    for word, similarity in similar:
        print(f"  {word}: {similarity:.3f}")

    # Document embedding
    doc = ["stock", "market", "volatility"]
    doc_vec = trainer.document_vector(doc)
    print(f"\nDocument vector for {doc}: {doc_vec[:5]}...")

    # Word analogy
    print("\nWord analogy (market + stock - volatility):")
    result = trainer.analogy(
        positive=["market", "stock"],
        negative=["volatility"],
        topn=3
    )
    for word, score in result:
        print(f"  {word}: {score:.3f}")


def example_glove():
    """Example: Using GloVe embeddings."""
    print("\n" + "="*60)
    print("GloVe Example")
    print("="*60)

    from puffin.nlp.embeddings import GloVeLoader
    import tempfile

    # Create a small mock GloVe file for demonstration
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        # In practice, download from: https://nlp.stanford.edu/projects/glove/
        f.write("market 0.1 0.2 0.3 0.4 0.5\n")
        f.write("stock 0.2 0.3 0.4 0.5 0.6\n")
        f.write("bond 0.3 0.4 0.5 0.6 0.7\n")
        f.write("price 0.4 0.5 0.6 0.7 0.8\n")
        f.write("volatility 0.15 0.25 0.35 0.45 0.55\n")
        glove_path = f.name

    try:
        print(f"\nLoading GloVe vectors from: {glove_path}")
        loader = GloVeLoader()
        loader.load(glove_path)

        print(f"Loaded {len(loader)} word vectors")
        print(f"Vector dimension: {loader.vector_size}")

        # Get word vector
        word = "market"
        vector = loader.word_vector(word)
        if vector is not None:
            print(f"\nVector for '{word}': {vector}")

        # Document embedding
        doc = ["stock", "market", "price"]
        doc_vec = loader.document_vector(doc)
        print(f"\nDocument vector for {doc}: {doc_vec}")

    finally:
        Path(glove_path).unlink()


def example_doc2vec():
    """Example: Training and using Doc2Vec."""
    print("\n" + "="*60)
    print("Doc2Vec Example")
    print("="*60)

    from puffin.nlp.embeddings import Doc2VecTrainer

    # Tokenize documents
    documents = [text.lower().split() for text in FINANCIAL_NEWS]

    # Train Doc2Vec
    print("\nTraining Doc2Vec model...")
    trainer = Doc2VecTrainer()
    model = trainer.train(
        documents,
        vector_size=50,
        window=5,
        min_count=1,
        epochs=20
    )

    print(f"Trained on {len(documents)} documents")
    print(f"Vector dimension: {model.vector_size}")

    # Get document vector
    doc_id = "0"
    doc_vec = trainer.document_vector(doc_id)
    print(f"\nDocument {doc_id} vector: {doc_vec[:5]}...")

    # Infer vector for new document
    new_doc = ["stock", "prices", "increased", "today"]
    inferred_vec = trainer.infer_vector(new_doc)
    print(f"\nInferred vector for new document: {inferred_vec[:5]}...")

    # Find similar documents
    print(f"\nDocuments similar to {new_doc}:")
    similar = trainer.similar_documents(new_doc, topn=3)
    for doc_id, similarity in similar:
        doc_idx = int(doc_id)
        print(f"  Doc {doc_id}: {similarity:.3f}")
        print(f"    -> {FINANCIAL_NEWS[doc_idx][:60]}...")


def example_sec_analysis():
    """Example: Analyzing SEC filings."""
    print("\n" + "="*60)
    print("SEC Filing Analysis Example")
    print("="*60)

    from puffin.nlp.sec_analysis import SECFilingAnalyzer
    from puffin.nlp.embeddings import Word2VecTrainer

    # Sample 10-K texts
    filing_2022 = """
    ITEM 1A. RISK FACTORS

    Our business faces risks from market volatility, regulatory changes,
    and competitive pressures. Economic uncertainty could impact our operations.

    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

    Revenue increased 10% this year. Market conditions were favorable.
    We maintained strong profitability through operational efficiency.
    """

    filing_2023 = """
    ITEM 1A. RISK FACTORS

    Significant risks include market volatility, regulatory compliance,
    litigation, and competitive dynamics. Global economic conditions remain uncertain.

    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

    Revenue grew 15% driven by strong demand. Despite macroeconomic headwinds,
    we achieved record profitability. Operations expanded into new markets.
    """

    # Train embedding model on sample corpus
    all_text = filing_2022 + " " + filing_2023
    documents = [all_text.lower().split()]

    print("\nTraining embedding model...")
    trainer = Word2VecTrainer()
    trainer.train(documents, vector_size=50, min_count=1, epochs=10)

    # Analyze filing
    print("\nAnalyzing 10-K filing...")
    analyzer = SECFilingAnalyzer()
    result = analyzer.analyze_10k(
        filing_2023,
        trainer.model,
        prior_text=filing_2022
    )

    print(f"Similarity to prior year: {result['similarity_to_prior']:.3f}")
    print(f"Risk factors embedding shape: {result['risk_factors_embedding'].shape}")
    print(f"MD&A embedding shape: {result['mda_embedding'].shape}")

    # Compare multiple filings
    print("\nComparing filings over time...")
    texts = [filing_2022, filing_2023]
    dates = ['2022-12-31', '2023-12-31']

    comparison = analyzer.compare_filings(texts, dates, trainer.model)
    print("\nComparison results:")
    print(comparison[['date', 'similarity_to_previous', 'change_magnitude']])

    # Detect language changes
    print("\nDetecting language changes...")
    changes = analyzer.detect_language_changes(texts, dates, threshold=0.1)

    for change in changes:
        print(f"\nDate: {change['date']}")
        print(f"Change score: {change['change_score']:.3f}")
        if change['new_terms']:
            print(f"Sample new terms: {change['new_terms'][:5]}")

    # Risk sentiment
    print("\nExtracting risk sentiment...")
    sentiment = analyzer.extract_risk_sentiment(filing_2023, trainer.model)
    print(f"Risk score: {sentiment['risk_score']:.3f}")


def example_transformer_embeddings():
    """Example: Using BERT/FinBERT embeddings."""
    print("\n" + "="*60)
    print("Transformer Embeddings Example")
    print("="*60)

    try:
        from puffin.nlp.transformer_embeddings import TransformerEmbedder

        print("\nInitializing transformer embedder...")
        embedder = TransformerEmbedder()

        # Encode texts
        texts = [
            "The company reported strong quarterly earnings.",
            "Market volatility increased amid economic uncertainty.",
            "Stock prices declined following the announcement.",
        ]

        print("\nEncoding texts with DistilBERT...")
        embeddings = embedder.encode(texts, model_name='distilbert-base-uncased')
        print(f"Embeddings shape: {embeddings.shape}")

        # Calculate similarity
        sim = embedder.similarity(texts[0], texts[1])
        print(f"\nSimilarity between texts 0 and 1: {sim:.3f}")

        # Semantic search
        corpus = [
            "Strong earnings growth driven by product sales.",
            "Regulatory challenges impacting operations.",
            "Market share gains in key segments.",
            "Cost reduction initiatives underway.",
        ]

        query = "financial performance improvement"
        print(f"\nSemantic search for: '{query}'")
        results = embedder.semantic_search(query, corpus, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['text'][:60]}...")
            print(f"   Score: {result['score']:.3f}")

        # Find similar texts
        print("\nFinding texts similar to first text...")
        similar = embedder.find_similar(texts[0], texts[1:], top_k=2)
        for text, score, idx in similar:
            print(f"  Score: {score:.3f} - {text[:60]}...")

    except ImportError as e:
        print(f"\nTransformers not available: {e}")
        print("Install with: pip install transformers torch")


def main():
    """Run all examples."""
    print("Word Embeddings Examples")
    print("=" * 60)

    # Check if gensim is available
    try:
        import gensim
        example_word2vec()
        example_doc2vec()
        example_sec_analysis()
    except ImportError:
        print("\nGensim not available. Install with: pip install gensim")
        print("Skipping Word2Vec, Doc2Vec, and SEC analysis examples.")

    example_glove()
    example_transformer_embeddings()

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
