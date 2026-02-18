"""
Tests for topic modeling functionality.
"""

import pytest
import numpy as np
from puffin.nlp.topic_models import LSIModel, LDAModel, find_optimal_topics


@pytest.fixture
def sample_documents():
    """Sample financial documents for testing."""
    return [
        "earnings growth revenue profit margin improved strong quarter",
        "revenue increased sales growth market share expansion",
        "profit margin declined costs increased expenses higher",
        "market volatility risk uncertainty economic conditions",
        "technology innovation digital transformation cloud computing",
        "customer satisfaction retention loyalty engagement metrics",
        "supply chain logistics inventory management efficiency",
        "earnings beat expectations analyst estimates exceeded",
        "revenue guidance raised outlook positive momentum",
        "market conditions challenging headwinds pressure competitive"
    ]


@pytest.fixture
def simple_documents():
    """Very simple documents for basic testing."""
    return [
        "cat dog animal pet",
        "dog cat pet animal",
        "car truck vehicle automobile",
        "truck car automobile vehicle"
    ]


class TestLSIModel:
    """Test LSI topic model."""

    def test_init(self):
        """Test LSI model initialization."""
        model = LSIModel()
        assert model.vectorizer is None
        assert model.svd is None
        assert model.n_topics is None

    def test_fit(self, sample_documents):
        """Test fitting LSI model."""
        model = LSIModel()
        result = model.fit(sample_documents, n_topics=3)

        assert result is model  # Check fluent interface
        assert model.n_topics == 3
        assert model.vectorizer is not None
        assert model.svd is not None
        assert model.feature_names is not None

    def test_transform(self, sample_documents):
        """Test transforming documents to topic weights."""
        model = LSIModel()
        model.fit(sample_documents, n_topics=3)

        # Transform same documents
        weights = model.transform(sample_documents)

        assert weights.shape == (len(sample_documents), 3)
        assert isinstance(weights, np.ndarray)

    def test_transform_new_documents(self, sample_documents):
        """Test transforming new documents."""
        model = LSIModel()
        model.fit(sample_documents, n_topics=3)

        new_docs = ["earnings strong revenue growth",
                    "market risk volatility uncertainty"]
        weights = model.transform(new_docs)

        assert weights.shape == (2, 3)

    def test_get_topics(self, sample_documents):
        """Test getting topic words."""
        model = LSIModel()
        model.fit(sample_documents, n_topics=3)

        topics = model.get_topics(n_words=5)

        assert len(topics) == 3
        for topic_id, words in topics:
            assert isinstance(topic_id, int)
            assert len(words) == 5
            for word, weight in words:
                assert isinstance(word, str)
                assert isinstance(weight, float)

    def test_explained_variance(self, sample_documents):
        """Test explained variance ratio."""
        model = LSIModel()
        model.fit(sample_documents, n_topics=3)

        variance = model.explained_variance_ratio()

        assert len(variance) == 3
        assert np.all(variance >= 0)
        assert np.all(variance <= 1)
        # Variance should be in descending order
        assert np.all(variance[:-1] >= variance[1:])

    def test_transform_before_fit(self, sample_documents):
        """Test that transform fails before fit."""
        model = LSIModel()

        with pytest.raises(ValueError):
            model.transform(sample_documents)

    def test_get_topics_before_fit(self):
        """Test that get_topics fails before fit."""
        model = LSIModel()

        with pytest.raises(ValueError):
            model.get_topics()


class TestLDAModel:
    """Test LDA topic model."""

    def test_init(self):
        """Test LDA model initialization."""
        model = LDAModel()
        assert model.model is None
        assert model.n_topics is None

    def test_fit_gensim(self, sample_documents):
        """Test fitting LDA with Gensim (if available)."""
        model = LDAModel(use_gensim=True)
        result = model.fit(sample_documents, n_topics=3, passes=5)

        assert result is model
        assert model.n_topics == 3
        assert model.model is not None

    def test_fit_sklearn(self, sample_documents):
        """Test fitting LDA with sklearn."""
        model = LDAModel(use_gensim=False)
        result = model.fit(sample_documents, n_topics=3)

        assert result is model
        assert model.n_topics == 3
        assert model.model is not None
        assert model.vectorizer is not None

    def test_transform(self, sample_documents):
        """Test transforming documents to topic distributions."""
        model = LDAModel(use_gensim=False)  # Use sklearn for consistency
        model.fit(sample_documents, n_topics=3)

        distributions = model.transform(sample_documents)

        assert distributions.shape == (len(sample_documents), 3)
        # Distributions should sum to ~1 for LDA
        sums = np.sum(distributions, axis=1)
        assert np.allclose(sums, 1.0, atol=0.01)

    def test_get_topics(self, sample_documents):
        """Test getting topic words."""
        model = LDAModel(use_gensim=False)
        model.fit(sample_documents, n_topics=3)

        topics = model.get_topics(n_words=5)

        assert len(topics) == 3
        for topic_id, words in topics:
            assert isinstance(topic_id, int)
            assert len(words) == 5
            for word, weight in words:
                assert isinstance(word, str)
                assert isinstance(weight, float)
                assert weight >= 0

    def test_coherence_score_sklearn(self, sample_documents):
        """Test coherence score with sklearn (should return 0)."""
        model = LDAModel(use_gensim=False)
        model.fit(sample_documents, n_topics=3)

        score = model.coherence_score()
        assert score == 0.0  # sklearn returns 0

    def test_perplexity(self, sample_documents):
        """Test perplexity calculation."""
        model = LDAModel(use_gensim=False)
        model.fit(sample_documents, n_topics=3)

        perplexity = model.perplexity()
        assert isinstance(perplexity, float)

    def test_transform_before_fit(self, sample_documents):
        """Test that transform fails before fit."""
        model = LDAModel()

        with pytest.raises(ValueError):
            model.transform(sample_documents)


class TestFindOptimalTopics:
    """Test optimal topic number finder."""

    def test_find_optimal_lsi(self, sample_documents):
        """Test finding optimal topics with LSI."""
        optimal_n, scores = find_optimal_topics(
            sample_documents,
            min_topics=2,
            max_topics=5,
            step=1,
            method='lsi'
        )

        assert isinstance(optimal_n, int)
        assert 2 <= optimal_n <= 5
        assert len(scores) == 4  # 2, 3, 4, 5

        # Check scores structure
        for n_topics, score in scores:
            assert isinstance(n_topics, int)
            assert isinstance(score, float)

    def test_find_optimal_lda(self, simple_documents):
        """Test finding optimal topics with LDA."""
        # Use simple docs to speed up test
        optimal_n, scores = find_optimal_topics(
            simple_documents,
            min_topics=2,
            max_topics=3,
            step=1,
            method='lda'
        )

        assert isinstance(optimal_n, int)
        assert 2 <= optimal_n <= 3
        assert len(scores) == 2  # 2, 3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_documents(self):
        """Test with empty document list."""
        model = LSIModel()

        with pytest.raises(Exception):  # Should raise some exception
            model.fit([], n_topics=3)

    def test_single_document(self):
        """Test with single document."""
        model = LSIModel()

        with pytest.raises(Exception):
            model.fit(["single document"], n_topics=3)

    def test_more_topics_than_documents(self, sample_documents):
        """Test requesting more topics than documents."""
        model = LSIModel()

        # Should work but be limited by available dimensions
        model.fit(sample_documents[:5], n_topics=10)

    def test_different_n_words(self, sample_documents):
        """Test getting different numbers of words per topic."""
        model = LSIModel()
        model.fit(sample_documents, n_topics=3)

        topics_5 = model.get_topics(n_words=5)
        topics_10 = model.get_topics(n_words=10)

        assert len(topics_5[0][1]) == 5
        # n_words=10 may return fewer if vocabulary is smaller than 10
        assert len(topics_10[0][1]) <= 10
        assert len(topics_10[0][1]) > len(topics_5[0][1])


class TestIntegration:
    """Integration tests for topic models."""

    def test_lsi_pipeline(self, sample_documents):
        """Test complete LSI pipeline."""
        # Fit model
        model = LSIModel()
        model.fit(sample_documents, n_topics=3)

        # Get topics
        topics = model.get_topics(n_words=5)
        assert len(topics) == 3

        # Transform documents
        weights = model.transform(sample_documents)
        assert weights.shape[0] == len(sample_documents)

        # Transform new document
        new_weights = model.transform(["revenue earnings growth"])
        assert new_weights.shape == (1, 3)

    def test_lda_pipeline(self, sample_documents):
        """Test complete LDA pipeline."""
        # Fit model
        model = LDAModel(use_gensim=False)
        model.fit(sample_documents, n_topics=3)

        # Get topics
        topics = model.get_topics(n_words=5)
        assert len(topics) == 3

        # Transform documents
        distributions = model.transform(sample_documents)
        assert distributions.shape[0] == len(sample_documents)

        # Check distributions sum to 1
        sums = np.sum(distributions, axis=1)
        assert np.allclose(sums, 1.0, atol=0.01)

    def test_compare_models(self, sample_documents):
        """Test comparing LSI and LDA models."""
        lsi_model = LSIModel()
        lsi_model.fit(sample_documents, n_topics=3)

        lda_model = LDAModel(use_gensim=False)
        lda_model.fit(sample_documents, n_topics=3)

        # Both should produce valid topics
        lsi_topics = lsi_model.get_topics(n_words=5)
        lda_topics = lda_model.get_topics(n_words=5)

        assert len(lsi_topics) == 3
        assert len(lda_topics) == 3

        # Both should transform documents
        lsi_weights = lsi_model.transform(sample_documents)
        lda_weights = lda_model.transform(sample_documents)

        assert lsi_weights.shape == (len(sample_documents), 3)
        assert lda_weights.shape == (len(sample_documents), 3)
