"""Tests for text vectorizers."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from puffin.nlp.vectorizers import DocumentTermMatrix, build_bow, build_tfidf


@pytest.fixture
def sample_documents():
    return [
        "The stock market rallied today on strong earnings.",
        "Stock prices declined amid economic concerns.",
        "The earnings report exceeded market expectations.",
        "Market volatility increased after the announcement.",
        "Strong economic data boosted investor confidence.",
    ]


def test_build_bow(sample_documents):
    matrix, features = build_bow(sample_documents, max_features=50)

    assert isinstance(matrix, csr_matrix)
    assert isinstance(features, list)
    assert matrix.shape[0] == len(sample_documents)
    assert matrix.shape[1] == len(features)
    assert len(features) > 0


def test_build_tfidf(sample_documents):
    matrix, features = build_tfidf(sample_documents, max_features=50)

    assert isinstance(matrix, csr_matrix)
    assert isinstance(features, list)
    assert matrix.shape[0] == len(sample_documents)
    assert matrix.shape[1] == len(features)
    assert len(features) > 0


def test_build_tfidf_with_ngrams(sample_documents):
    matrix, features = build_tfidf(
        sample_documents,
        max_features=100,
        ngram_range=(1, 2)
    )

    # Should include both unigrams and bigrams
    assert any(" " in feat for feat in features), "No bigrams found"
    assert any(" " not in feat for feat in features), "No unigrams found"


def test_document_term_matrix_bow(sample_documents):
    dtm = DocumentTermMatrix(method="bow", max_features=50)

    matrix = dtm.fit_transform(sample_documents)

    assert isinstance(matrix, csr_matrix)
    assert matrix.shape[0] == len(sample_documents)
    assert dtm.get_vocabulary_size() == matrix.shape[1]


def test_document_term_matrix_tfidf(sample_documents):
    dtm = DocumentTermMatrix(method="tfidf", max_features=50, ngram_range=(1, 2))

    matrix = dtm.fit_transform(sample_documents)

    assert isinstance(matrix, csr_matrix)
    assert matrix.shape[0] == len(sample_documents)


def test_transform_new_documents(sample_documents):
    dtm = DocumentTermMatrix(method="tfidf")

    # Fit on original documents
    dtm.fit_transform(sample_documents)

    # Transform new documents
    new_docs = [
        "The market showed strong performance today.",
        "Economic indicators suggest continued growth.",
    ]

    new_matrix = dtm.transform(new_docs)

    assert new_matrix.shape[0] == len(new_docs)
    assert new_matrix.shape[1] == dtm.get_vocabulary_size()


def test_get_top_terms(sample_documents):
    dtm = DocumentTermMatrix(method="tfidf")
    dtm.fit_transform(sample_documents)

    # Get top terms across all documents
    top_terms = dtm.get_top_terms(n=10)

    assert isinstance(top_terms, list)
    assert len(top_terms) <= 10
    assert all(isinstance(term, tuple) for term in top_terms)
    assert all(isinstance(term[0], str) and isinstance(term[1], float) for term in top_terms)


def test_get_top_terms_for_document(sample_documents):
    dtm = DocumentTermMatrix(method="tfidf")
    dtm.fit_transform(sample_documents)

    # Get top terms for first document
    top_terms = dtm.get_top_terms(n=5, doc_idx=0)

    assert isinstance(top_terms, list)
    assert len(top_terms) <= 5


def test_get_feature_names(sample_documents):
    dtm = DocumentTermMatrix(method="bow")
    dtm.fit_transform(sample_documents)

    features = dtm.get_feature_names()

    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(f, str) for f in features)


def test_get_document_length(sample_documents):
    dtm = DocumentTermMatrix(method="bow")
    dtm.fit_transform(sample_documents)

    length = dtm.get_document_length(0)

    assert isinstance(length, int)
    assert length > 0


def test_get_term_frequency(sample_documents):
    dtm = DocumentTermMatrix(method="bow")
    dtm.fit_transform(sample_documents)

    # "market" appears in multiple documents
    freq = dtm.get_term_frequency("market")

    assert isinstance(freq, int)
    assert freq >= 0


def test_to_dense(sample_documents):
    dtm = DocumentTermMatrix(method="tfidf")
    dtm.fit_transform(sample_documents)

    dense = dtm.to_dense()

    assert isinstance(dense, np.ndarray)
    assert dense.shape[0] == len(sample_documents)
    assert dense.shape[1] == dtm.get_vocabulary_size()


def test_invalid_method():
    with pytest.raises(ValueError, match="Unknown method"):
        DocumentTermMatrix(method="invalid")


def test_transform_before_fit(sample_documents):
    dtm = DocumentTermMatrix(method="tfidf")

    with pytest.raises(ValueError, match="Must call fit_transform"):
        dtm.transform(sample_documents)


def test_get_top_terms_before_fit():
    dtm = DocumentTermMatrix(method="tfidf")

    with pytest.raises(ValueError, match="Must call fit_transform"):
        dtm.get_top_terms()


def test_min_max_df(sample_documents):
    # Test filtering by document frequency
    dtm = DocumentTermMatrix(
        method="tfidf",
        min_df=2,  # Appear in at least 2 documents
        max_df=0.8,  # Appear in at most 80% of documents
    )

    matrix = dtm.fit_transform(sample_documents)

    assert matrix.shape[1] > 0  # Should have some features


def test_repr(sample_documents):
    dtm = DocumentTermMatrix(method="tfidf")

    # Before fitting
    repr_before = repr(dtm)
    assert "not fitted" in repr_before

    # After fitting
    dtm.fit_transform(sample_documents)
    repr_after = repr(dtm)
    assert "docs=" in repr_after
    assert "features=" in repr_after


def test_empty_documents():
    dtm = DocumentTermMatrix(method="bow")

    empty_docs = ["", "", ""]
    matrix = dtm.fit_transform(empty_docs)

    assert matrix.shape[0] == len(empty_docs)
    # Vocabulary size might be 0 for empty documents
    assert matrix.shape[1] >= 0
