"""Tests for NLP pipeline."""

import pytest

from puffin.nlp.pipeline import NLPPipeline, ProcessedDoc, SimpleTokenizer


@pytest.fixture
def sample_text():
    return (
        "Apple Inc. announced record earnings of $5.2 billion, up 15% from last year. "
        "The stock rallied on the news."
    )


@pytest.fixture
def sample_texts():
    return [
        "Tesla shares gained 10% after strong delivery numbers.",
        "Microsoft reported a 20% decline in cloud revenue.",
        "Amazon acquired a robotics startup for $100M.",
    ]


def test_simple_tokenizer():
    tokenizer = SimpleTokenizer()
    text = "The stock price increased by 5%."

    tokens = tokenizer.tokenize(text)
    assert "stock" in tokens
    assert "price" in tokens
    assert "increased" in tokens
    assert len(tokens) > 0

    sentences = tokenizer.split_sentences(text)
    assert len(sentences) > 0


def test_nlp_pipeline_initialization():
    pipeline = NLPPipeline()
    assert pipeline is not None
    # Pipeline will use fallback if spaCy not installed


def test_process_text(sample_text):
    pipeline = NLPPipeline()
    doc = pipeline.process(sample_text)

    assert isinstance(doc, ProcessedDoc)
    assert doc.text == sample_text
    assert len(doc.tokens) > 0
    assert len(doc.sentences) > 0


def test_extract_entities(sample_text):
    pipeline = NLPPipeline()
    entities = pipeline.extract_entities(sample_text)

    assert isinstance(entities, list)
    # Should find at least money and percent entities
    entity_texts = [ent[0] for ent in entities]
    entity_labels = [ent[1] for ent in entities]

    # Check if we found any financial entities
    assert any(label in ["MONEY", "PERCENT", "ORG"] for label in entity_labels)


def test_extract_financial_terms(sample_text):
    pipeline = NLPPipeline()
    terms = pipeline.extract_financial_terms(sample_text)

    assert isinstance(terms, list)
    # Should find terms like "earnings", "stock", etc.
    assert any(term in ["earnings", "stock", "rallied"] for term in terms)


def test_batch_process(sample_texts):
    pipeline = NLPPipeline()
    docs = pipeline.batch_process(sample_texts)

    assert len(docs) == len(sample_texts)
    assert all(isinstance(doc, ProcessedDoc) for doc in docs)
    assert all(len(doc.tokens) > 0 for doc in docs)


def test_processed_doc_attributes():
    doc = ProcessedDoc(
        text="Test text",
        tokens=["test", "text"],
        lemmas=["test", "text"],
        entities=[("Entity", "ORG")],
        sentences=["Test text."],
        pos_tags=["NOUN", "NOUN"]
    )

    assert doc.text == "Test text"
    assert len(doc.tokens) == 2
    assert len(doc.entities) == 1
    assert doc.entities[0][0] == "Entity"
    assert doc.entities[0][1] == "ORG"


def test_financial_terms_vocabulary():
    pipeline = NLPPipeline()

    # Test various financial texts
    texts = [
        "The stock price increased with high volume.",
        "Earnings beat expectations, profit margins expanded.",
        "The merger was announced, triggering a rally.",
        "Bearish sentiment dominated as volatility spiked.",
    ]

    for text in texts:
        terms = pipeline.extract_financial_terms(text)
        assert len(terms) > 0, f"No financial terms found in: {text}"


def test_empty_text():
    pipeline = NLPPipeline()
    doc = pipeline.process("")

    assert isinstance(doc, ProcessedDoc)
    assert doc.text == ""
    assert len(doc.tokens) == 0


def test_special_characters():
    pipeline = NLPPipeline()
    text = "Stock up 5.5%! Revenue: $100M++"

    doc = pipeline.process(text)
    assert len(doc.tokens) > 0

    entities = pipeline.extract_entities(text)
    # Should handle special characters gracefully
    assert isinstance(entities, list)
