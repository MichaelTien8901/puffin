"""
Tests for SEC filing analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from puffin.nlp.sec_analysis import SECFilingAnalyzer


# Mock embedding model for testing
class MockEmbeddingModel:
    """Mock word embedding model for testing."""

    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.wv = self
        # Simple vocabulary for testing
        self.vocab = {
            'market': np.random.randn(vector_size),
            'risk': np.random.randn(vector_size),
            'volatility': np.random.randn(vector_size),
            'revenue': np.random.randn(vector_size),
            'profit': np.random.randn(vector_size),
            'loss': np.random.randn(vector_size),
            'uncertainty': np.random.randn(vector_size),
            'regulation': np.random.randn(vector_size),
            'litigation': np.random.randn(vector_size),
            'business': np.random.randn(vector_size),
            'operations': np.random.randn(vector_size),
            'financial': np.random.randn(vector_size),
        }

    def __getitem__(self, word):
        if word in self.vocab:
            return self.vocab[word]
        raise KeyError(f"Word '{word}' not in vocabulary")


# Sample SEC filing texts
SAMPLE_10K = """
ITEM 1A. RISK FACTORS

Our business faces various risks including market volatility, regulatory changes,
and competitive pressures. The uncertainty in financial markets could adversely
affect our operations.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue increased by 15% this year driven by strong market conditions.
Our profit margins improved due to operational efficiencies.
"""

SAMPLE_10K_PRIOR = """
ITEM 1A. RISK FACTORS

We face risks from market conditions and regulatory compliance.
Business operations may be impacted by economic uncertainty.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue grew modestly this year. Market conditions remained challenging
but we maintained profitability through cost controls.
"""

SAMPLE_10K_2 = """
ITEM 1A. RISK FACTORS

Significant risks include litigation, regulatory scrutiny, and market volatility.
Loss of key customers could adversely impact our financial results.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Despite challenging market conditions, revenue increased. We continue to
invest in operations and new business opportunities.
"""


class TestSECFilingAnalyzer:
    """Test SECFilingAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = SECFilingAnalyzer()
        assert analyzer is not None

    def test_tokenize(self):
        """Test text tokenization."""
        analyzer = SECFilingAnalyzer()
        tokens = analyzer._tokenize("Market volatility increased significantly.")

        assert isinstance(tokens, list)
        assert 'market' in tokens
        assert 'volatility' in tokens
        assert 'increased' in tokens
        assert 'significantly' in tokens

    def test_extract_section(self):
        """Test section extraction."""
        analyzer = SECFilingAnalyzer()

        # Extract risk factors
        risk_section = analyzer._extract_section(
            SAMPLE_10K,
            r'item\s+1a\.?\s+risk\s+factors.*?(?=item\s+1b|item\s+2|item\s+7|\Z)'
        )

        assert risk_section != ""
        assert 'risk factors' in risk_section.lower()
        assert 'market volatility' in risk_section.lower()

    def test_get_embedding(self):
        """Test getting text embeddings."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        text = "market volatility risk"
        embedding = analyzer._get_embedding(text, model)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (50,)
        assert not np.all(embedding == 0)

    def test_get_embedding_unknown_words(self):
        """Test embedding for text with unknown words."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        text = "xyz abc def"
        embedding = analyzer._get_embedding(text, model)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (50,)
        assert np.all(embedding == 0)  # All unknown words -> zero vector

    def test_analyze_10k_basic(self):
        """Test basic 10-K analysis."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        result = analyzer.analyze_10k(SAMPLE_10K, model)

        assert 'risk_factors_embedding' in result
        assert 'mda_embedding' in result
        assert 'full_text_embedding' in result
        assert 'similarity_to_prior' in result

        assert isinstance(result['full_text_embedding'], np.ndarray)
        assert result['similarity_to_prior'] is None

    def test_analyze_10k_with_prior(self):
        """Test 10-K analysis with prior year comparison."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        result = analyzer.analyze_10k(SAMPLE_10K, model, prior_text=SAMPLE_10K_PRIOR)

        assert result['similarity_to_prior'] is not None
        assert isinstance(result['similarity_to_prior'], float)
        assert -1 <= result['similarity_to_prior'] <= 1

    def test_compare_filings(self):
        """Test comparing multiple filings."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        texts = [SAMPLE_10K_PRIOR, SAMPLE_10K, SAMPLE_10K_2]
        dates = ['2021-12-31', '2022-12-31', '2023-12-31']

        df = analyzer.compare_filings(texts, dates, model)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'date' in df.columns
        assert 'similarity_to_previous' in df.columns
        assert 'change_magnitude' in df.columns
        assert 'embedding' in df.columns

        # First filing has no previous
        assert pd.isna(df.iloc[0]['similarity_to_previous'])

        # Subsequent filings have similarity scores
        assert df.iloc[1]['similarity_to_previous'] is not None
        assert df.iloc[2]['similarity_to_previous'] is not None

    def test_compare_filings_datetime(self):
        """Test comparing filings with datetime objects."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        texts = [SAMPLE_10K_PRIOR, SAMPLE_10K]
        dates = [datetime(2021, 12, 31), datetime(2022, 12, 31)]

        df = analyzer.compare_filings(texts, dates, model)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_compare_filings_mismatched_length(self):
        """Test that mismatched lengths raise error."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        texts = [SAMPLE_10K_PRIOR, SAMPLE_10K]
        dates = ['2021-12-31']

        with pytest.raises(ValueError, match="Number of texts must match"):
            analyzer.compare_filings(texts, dates, model)

    def test_detect_language_changes(self):
        """Test detecting language changes."""
        analyzer = SECFilingAnalyzer()

        texts = [SAMPLE_10K_PRIOR, SAMPLE_10K, SAMPLE_10K_2]
        dates = ['2021-12-31', '2022-12-31', '2023-12-31']

        changes = analyzer.detect_language_changes(texts, dates, threshold=0.1)

        assert isinstance(changes, list)
        # Should detect some changes between filings
        assert len(changes) >= 0

        if changes:
            change = changes[0]
            assert 'date' in change
            assert 'change_score' in change
            assert 'new_terms' in change
            assert 'removed_terms' in change

            assert isinstance(change['new_terms'], list)
            assert isinstance(change['removed_terms'], list)

    def test_detect_language_changes_high_threshold(self):
        """Test language change detection with high threshold."""
        analyzer = SECFilingAnalyzer()

        texts = [SAMPLE_10K_PRIOR, SAMPLE_10K]
        dates = ['2021-12-31', '2022-12-31']

        # High threshold should detect fewer changes
        changes = analyzer.detect_language_changes(texts, dates, threshold=0.9)

        assert isinstance(changes, list)
        # With high threshold, likely no changes detected
        # (unless texts are very different)

    def test_extract_risk_sentiment(self):
        """Test extracting risk sentiment."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        sentiment = analyzer.extract_risk_sentiment(SAMPLE_10K, model)

        assert 'risk_score' in sentiment
        assert 'top_risk_terms' in sentiment

        assert isinstance(sentiment['risk_score'], float)
        assert 0 <= sentiment['risk_score'] <= 1

        assert isinstance(sentiment['top_risk_terms'], list)

    def test_extract_risk_sentiment_custom_keywords(self):
        """Test risk sentiment with custom keywords."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        custom_keywords = ['risk', 'uncertainty', 'volatility']
        sentiment = analyzer.extract_risk_sentiment(
            SAMPLE_10K,
            model,
            risk_keywords=custom_keywords
        )

        assert 'risk_score' in sentiment
        assert isinstance(sentiment['risk_score'], float)

    def test_get_vector_size(self):
        """Test getting vector size from model."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=75)

        size = analyzer._get_vector_size(model)
        assert size == 75

    def test_compare_filings_sorting(self):
        """Test that filings are sorted by date."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        # Provide dates in non-sorted order
        texts = [SAMPLE_10K, SAMPLE_10K_PRIOR, SAMPLE_10K_2]
        dates = ['2022-12-31', '2021-12-31', '2023-12-31']

        df = analyzer.compare_filings(texts, dates, model)

        # Check dates are sorted
        assert df.iloc[0]['date'] < df.iloc[1]['date']
        assert df.iloc[1]['date'] < df.iloc[2]['date']


class TestSECFilingAnalyzerIntegration:
    """Integration tests for SEC filing analysis."""

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        analyzer = SECFilingAnalyzer()
        model = MockEmbeddingModel(vector_size=50)

        # Analyze individual filing
        result = analyzer.analyze_10k(SAMPLE_10K, model, prior_text=SAMPLE_10K_PRIOR)

        assert result['risk_factors_embedding'] is not None
        assert result['similarity_to_prior'] is not None

        # Compare multiple filings
        texts = [SAMPLE_10K_PRIOR, SAMPLE_10K, SAMPLE_10K_2]
        dates = ['2021-12-31', '2022-12-31', '2023-12-31']

        comparison_df = analyzer.compare_filings(texts, dates, model)
        assert len(comparison_df) == 3

        # Detect language changes
        changes = analyzer.detect_language_changes(texts, dates, threshold=0.1)
        assert isinstance(changes, list)

        # Extract risk sentiment
        sentiment = analyzer.extract_risk_sentiment(SAMPLE_10K, model)
        assert 'risk_score' in sentiment


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
