"""Text vectorization using bag-of-words and TF-IDF."""

from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_bow(
    documents: list[str],
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
    **kwargs: Any
) -> tuple[csr_matrix, list[str]]:
    """
    Build bag-of-words representation of documents.

    Args:
        documents: List of text documents
        max_features: Maximum number of features (vocabulary size)
        min_df: Minimum document frequency for a term
        max_df: Maximum document frequency (as fraction)
        **kwargs: Additional arguments passed to CountVectorizer

    Returns:
        Tuple of (sparse_matrix, feature_names)
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        **kwargs
    )

    sparse_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out().tolist()

    return sparse_matrix, feature_names


def build_tfidf(
    documents: list[str],
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
    **kwargs: Any
) -> tuple[csr_matrix, list[str]]:
    """
    Build TF-IDF representation of documents.

    Args:
        documents: List of text documents
        max_features: Maximum number of features (vocabulary size)
        ngram_range: Range of n-grams to extract (default: unigrams and bigrams)
        min_df: Minimum document frequency for a term
        max_df: Maximum document frequency (as fraction)
        **kwargs: Additional arguments passed to TfidfVectorizer

    Returns:
        Tuple of (sparse_matrix, feature_names)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        **kwargs
    )

    sparse_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out().tolist()

    return sparse_matrix, feature_names


class DocumentTermMatrix:
    """
    Document-term matrix wrapper for sklearn vectorizers.

    Supports both bag-of-words (CountVectorizer) and TF-IDF (TfidfVectorizer).
    """

    def __init__(
        self,
        method: str = "tfidf",
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        **kwargs: Any
    ):
        """
        Initialize document-term matrix.

        Args:
            method: Vectorization method ("bow" or "tfidf")
            max_features: Maximum number of features
            ngram_range: N-gram range for feature extraction
            min_df: Minimum document frequency
            max_df: Maximum document frequency (as fraction)
            **kwargs: Additional arguments for vectorizer
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.kwargs = kwargs

        if method == "bow":
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                **kwargs
            )
        elif method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'bow' or 'tfidf'.")

        self.matrix_: csr_matrix | None = None
        self.feature_names_: list[str] | None = None

    def fit_transform(self, documents: list[str]) -> csr_matrix:
        """
        Fit vectorizer on documents and transform them.

        Args:
            documents: List of text documents

        Returns:
            Document-term sparse matrix
        """
        try:
            self.matrix_ = self.vectorizer.fit_transform(documents)
            self.feature_names_ = self.vectorizer.get_feature_names_out().tolist()
        except ValueError:
            # Empty vocabulary (e.g., all documents are empty/stop words)
            from scipy.sparse import csr_matrix as _csr
            self.matrix_ = _csr((len(documents), 0))
            self.feature_names_ = []
        return self.matrix_

    def transform(self, documents: list[str]) -> csr_matrix:
        """
        Transform new documents using fitted vectorizer.

        Args:
            documents: List of text documents

        Returns:
            Document-term sparse matrix
        """
        if self.feature_names_ is None:
            raise ValueError("Must call fit_transform before transform.")

        return self.vectorizer.transform(documents)

    def get_feature_names(self) -> list[str]:
        """Get feature names (vocabulary)."""
        if self.feature_names_ is None:
            raise ValueError("Must call fit_transform before getting feature names.")
        return self.feature_names_

    def get_top_terms(self, n: int = 20, doc_idx: int | None = None) -> list[tuple[str, float]]:
        """
        Get top terms by weight.

        Args:
            n: Number of top terms to return
            doc_idx: If provided, get top terms for this document index.
                    Otherwise, get top terms across all documents.

        Returns:
            List of (term, weight) tuples sorted by weight descending
        """
        if self.matrix_ is None or self.feature_names_ is None:
            raise ValueError("Must call fit_transform before getting top terms.")

        if doc_idx is not None:
            # Top terms for a specific document
            if doc_idx >= self.matrix_.shape[0]:
                raise IndexError(f"Document index {doc_idx} out of range.")

            doc_vector = self.matrix_[doc_idx].toarray().flatten()
            top_indices = np.argsort(doc_vector)[-n:][::-1]
            top_terms = [
                (self.feature_names_[idx], doc_vector[idx])
                for idx in top_indices
                if doc_vector[idx] > 0
            ]
        else:
            # Top terms across all documents (by sum or mean)
            term_scores = np.asarray(self.matrix_.sum(axis=0)).flatten()
            top_indices = np.argsort(term_scores)[-n:][::-1]
            top_terms = [
                (self.feature_names_[idx], term_scores[idx])
                for idx in top_indices
            ]

        return top_terms

    def get_document_length(self, doc_idx: int) -> int:
        """
        Get the total term count for a document (for BOW).

        Args:
            doc_idx: Document index

        Returns:
            Total term count
        """
        if self.matrix_ is None:
            raise ValueError("Must call fit_transform first.")

        return int(self.matrix_[doc_idx].sum())

    def get_vocabulary_size(self) -> int:
        """Get the vocabulary size."""
        if self.feature_names_ is None:
            raise ValueError("Must call fit_transform first.")
        return len(self.feature_names_)

    def get_term_frequency(self, term: str) -> int:
        """
        Get document frequency of a term (number of documents containing it).

        Args:
            term: Term to look up

        Returns:
            Number of documents containing the term
        """
        if self.matrix_ is None or self.feature_names_ is None:
            raise ValueError("Must call fit_transform first.")

        if term not in self.feature_names_:
            return 0

        term_idx = self.feature_names_.index(term)
        # Count non-zero entries in the term column
        return int((self.matrix_[:, term_idx] > 0).sum())

    def to_dense(self) -> np.ndarray:
        """Convert sparse matrix to dense numpy array."""
        if self.matrix_ is None:
            raise ValueError("Must call fit_transform first.")
        return self.matrix_.toarray()

    def __repr__(self) -> str:
        if self.matrix_ is None:
            return f"DocumentTermMatrix(method={self.method}, not fitted)"
        return (
            f"DocumentTermMatrix(method={self.method}, "
            f"docs={self.matrix_.shape[0]}, "
            f"features={self.matrix_.shape[1]})"
        )
