"""
Tests for earnings call topic analysis.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from puffin.nlp.earnings_topics import (
    EarningsTopicAnalyzer,
    compare_earnings_topics
)


@pytest.fixture
def earnings_transcripts():
    """Sample earnings call transcripts."""
    return [
        "revenue growth exceeded expectations strong quarter improved margins profit increased",
        "digital transformation cloud computing technology investment innovation strategy",
        "market conditions challenging headwinds competitive pressure pricing dynamics",
        "customer acquisition retention loyalty engagement improved metrics growth",
        "supply chain logistics efficiency inventory management cost optimization",
        "earnings beat analyst estimates revenue guidance raised outlook positive",
        "economic uncertainty volatility risk management strategic positioning",
        "product innovation pipeline development research investment future growth",
        "operational excellence cost reduction efficiency gains margin improvement",
        "market expansion international growth geographic diversification strategy"
    ]


@pytest.fixture
def earnings_dates():
    """Sample dates for earnings calls."""
    base_date = datetime(2024, 1, 1)
    return [base_date + timedelta(days=90 * i) for i in range(10)]


@pytest.fixture
def sentiment_scores():
    """Sample sentiment scores."""
    return [0.8, 0.6, -0.3, 0.7, 0.2, 0.9, -0.4, 0.5, 0.4, 0.3]


class TestEarningsTopicAnalyzer:
    """Test EarningsTopicAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        analyzer = EarningsTopicAnalyzer(n_topics=5)

        assert analyzer.n_topics == 5
        assert analyzer.model_type == 'lda'
        assert analyzer.model is None
        assert analyzer.topic_labels == {}

    def test_init_lsi(self):
        """Test analyzer with LSI model."""
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')

        assert analyzer.model_type == 'lsi'
        assert analyzer.n_topics == 3

    def test_analyze_basic(self, earnings_transcripts):
        """Test basic analysis of earnings calls."""
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        results = analyzer.analyze(earnings_transcripts)

        # Check result structure
        assert 'topics' in results
        assert 'distributions' in results
        assert 'dominant_topics' in results
        assert 'model' in results

        # Check topics
        topics = results['topics']
        assert len(topics) == 3
        for topic_id, words in topics:
            assert isinstance(topic_id, int)
            assert len(words) == 10  # Default n_words

        # Check distributions
        distributions = results['distributions']
        assert distributions.shape == (len(earnings_transcripts), 3)

        # Check dominant topics
        dominant = results['dominant_topics']
        assert len(dominant) == len(earnings_transcripts)
        assert all(0 <= t < 3 for t in dominant)

    def test_analyze_with_dates(self, earnings_transcripts, earnings_dates):
        """Test analysis with temporal information."""
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        results = analyzer.analyze(earnings_transcripts, dates=earnings_dates)

        assert 'temporal_trends' in results

        trends = results['temporal_trends']
        assert len(trends) == 3  # One per topic

        for topic_id, trend_data in trends.items():
            assert 'weights' in trend_data
            assert 'dates' in trend_data
            assert 'trend_direction' in trend_data
            assert len(trend_data['weights']) == len(earnings_transcripts)

    def test_analyze_empty_transcripts(self):
        """Test analysis with empty transcript list."""
        analyzer = EarningsTopicAnalyzer(n_topics=3)

        with pytest.raises(ValueError):
            analyzer.analyze([])

    def test_detect_topic_shifts(self, earnings_transcripts, earnings_dates):
        """Test detection of topic shifts over time."""
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        analyzer.analyze(earnings_transcripts, dates=earnings_dates)

        shifts = analyzer.detect_topic_shifts(
            earnings_transcripts,
            earnings_dates,
            window_size=3,
            threshold=0.2
        )

        # Shifts is a list of dicts
        assert isinstance(shifts, list)

        for shift in shifts:
            assert 'date' in shift
            assert 'old_topic' in shift
            assert 'new_topic' in shift
            assert 'change_magnitude' in shift
            assert isinstance(shift['date'], datetime)
            assert 0 <= shift['old_topic'] < 3
            assert 0 <= shift['new_topic'] < 3

    def test_detect_shifts_mismatched_lengths(self, earnings_transcripts):
        """Test shift detection with mismatched lengths."""
        analyzer = EarningsTopicAnalyzer(n_topics=3)
        analyzer.analyze(earnings_transcripts)

        dates = [datetime.now() for _ in range(5)]  # Wrong length

        with pytest.raises(ValueError):
            analyzer.detect_topic_shifts(earnings_transcripts, dates)

    def test_topic_sentiment_provided(
        self,
        earnings_transcripts,
        sentiment_scores
    ):
        """Test topic sentiment with provided sentiment scores."""
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        analyzer.analyze(earnings_transcripts)

        sentiment_map = analyzer.topic_sentiment(
            earnings_transcripts,
            sentiment_scores
        )

        assert len(sentiment_map) == 3  # One per topic

        for topic_id, sentiment_data in sentiment_map.items():
            assert 'avg_sentiment' in sentiment_data
            assert 'sentiment_variance' in sentiment_data
            assert 'document_count' in sentiment_data
            assert 'weighted_sentiment' in sentiment_data

            assert isinstance(sentiment_data['avg_sentiment'], float)
            assert isinstance(sentiment_data['document_count'], int)

    def test_topic_sentiment_computed(self, earnings_transcripts):
        """Test topic sentiment with computed sentiment."""
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        analyzer.analyze(earnings_transcripts)

        # No sentiment scores provided
        sentiment_map = analyzer.topic_sentiment(earnings_transcripts)

        assert len(sentiment_map) == 3

        for topic_id, sentiment_data in sentiment_map.items():
            assert 'avg_sentiment' in sentiment_data
            # Computed sentiment should be between -1 and 1
            assert -1.0 <= sentiment_data['avg_sentiment'] <= 1.0

    def test_topic_sentiment_mismatched_lengths(self, earnings_transcripts):
        """Test sentiment with mismatched lengths."""
        analyzer = EarningsTopicAnalyzer(n_topics=3)
        analyzer.analyze(earnings_transcripts)

        wrong_sentiments = [0.5, 0.6]  # Wrong length

        with pytest.raises(ValueError):
            analyzer.topic_sentiment(earnings_transcripts, wrong_sentiments)

    def test_label_topics(self, earnings_transcripts):
        """Test labeling topics with custom names."""
        analyzer = EarningsTopicAnalyzer(n_topics=3)
        analyzer.analyze(earnings_transcripts)

        labels = {
            0: "Financial Performance",
            1: "Technology & Innovation",
            2: "Market Conditions"
        }

        analyzer.label_topics(labels)

        assert analyzer.topic_labels == labels

    def test_get_topic_summary(self, earnings_transcripts):
        """Test getting topic summary."""
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        analyzer.analyze(earnings_transcripts)

        analyzer.label_topics({0: "Custom Label"})

        summary = analyzer.get_topic_summary(0, n_words=5)

        assert summary['topic_id'] == 0
        assert summary['label'] == "Custom Label"
        assert 'top_words' in summary
        assert len(summary['top_words']) == 5

    def test_get_summary_before_fit(self):
        """Test getting summary before fitting."""
        analyzer = EarningsTopicAnalyzer(n_topics=3)

        with pytest.raises(ValueError):
            analyzer.get_topic_summary(0)

    def test_get_summary_invalid_topic(self, earnings_transcripts):
        """Test getting summary for invalid topic."""
        analyzer = EarningsTopicAnalyzer(n_topics=3)
        analyzer.analyze(earnings_transcripts)

        with pytest.raises(ValueError):
            analyzer.get_topic_summary(10)  # Topic doesn't exist


class TestCompareEarningsTopics:
    """Test comparing topic models."""

    def test_compare_two_analyzers(self, earnings_transcripts):
        """Test comparing two analyzers."""
        # Create two analyzers with same transcripts
        analyzer1 = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        analyzer1.analyze(earnings_transcripts[:5])

        analyzer2 = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        analyzer2.analyze(earnings_transcripts[5:])

        similarity = compare_earnings_topics(analyzer1, analyzer2)

        # Should be 3x3 matrix
        assert similarity.shape == (3, 3)

        # Values should be reasonable
        assert np.all(np.isfinite(similarity))

    def test_compare_different_n_topics(self, earnings_transcripts):
        """Test comparing models with different numbers of topics."""
        analyzer1 = EarningsTopicAnalyzer(n_topics=2, model_type='lsi')
        analyzer1.analyze(earnings_transcripts)

        analyzer2 = EarningsTopicAnalyzer(n_topics=4, model_type='lsi')
        analyzer2.analyze(earnings_transcripts)

        similarity = compare_earnings_topics(analyzer1, analyzer2)

        # Should be 2x4 matrix
        assert similarity.shape == (2, 4)

    def test_compare_euclidean(self, earnings_transcripts):
        """Test comparison with euclidean distance."""
        analyzer1 = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        analyzer1.analyze(earnings_transcripts[:5])

        analyzer2 = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        analyzer2.analyze(earnings_transcripts[5:])

        similarity = compare_earnings_topics(
            analyzer1,
            analyzer2,
            method='euclidean'
        )

        assert similarity.shape == (3, 3)
        # Euclidean distances are negative
        assert np.all(similarity <= 0)

    def test_compare_unfitted_analyzer(self, earnings_transcripts):
        """Test comparing with unfitted analyzer."""
        analyzer1 = EarningsTopicAnalyzer(n_topics=3)
        analyzer1.analyze(earnings_transcripts)

        analyzer2 = EarningsTopicAnalyzer(n_topics=3)

        with pytest.raises(ValueError):
            compare_earnings_topics(analyzer1, analyzer2)


class TestTemporalAnalysis:
    """Test temporal analysis features."""

    def test_temporal_trends_structure(self, earnings_transcripts, earnings_dates):
        """Test structure of temporal trends."""
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        results = analyzer.analyze(earnings_transcripts, dates=earnings_dates)

        trends = results['temporal_trends']

        for topic_id in range(3):
            assert topic_id in trends

            trend = trends[topic_id]
            assert 'weights' in trend
            assert 'dates' in trend
            assert 'trend_direction' in trend
            assert 'avg_weight' in trend
            assert 'max_weight' in trend
            assert 'min_weight' in trend

            # Check trend direction is valid
            assert trend['trend_direction'] in [
                'increasing', 'decreasing', 'stable', 'unknown'
            ]

    def test_temporal_with_string_dates(self, earnings_transcripts):
        """Test temporal analysis with ISO string dates."""
        dates = [
            "2024-01-01T00:00:00",
            "2024-04-01T00:00:00",
            "2024-07-01T00:00:00",
            "2024-10-01T00:00:00",
            "2025-01-01T00:00:00",
            "2025-04-01T00:00:00",
            "2025-07-01T00:00:00",
            "2025-10-01T00:00:00",
            "2026-01-01T00:00:00",
            "2026-04-01T00:00:00"
        ]

        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')
        results = analyzer.analyze(earnings_transcripts, dates=dates)

        assert 'temporal_trends' in results
        trends = results['temporal_trends']
        assert len(trends) == 3


class TestIntegration:
    """Integration tests for earnings topic analysis."""

    def test_full_analysis_pipeline(
        self,
        earnings_transcripts,
        earnings_dates,
        sentiment_scores
    ):
        """Test complete analysis pipeline."""
        # Create analyzer
        analyzer = EarningsTopicAnalyzer(n_topics=3, model_type='lsi')

        # Analyze transcripts
        results = analyzer.analyze(earnings_transcripts, dates=earnings_dates)

        assert results['topics'] is not None
        assert results['distributions'] is not None
        assert results['temporal_trends'] is not None

        # Detect shifts
        shifts = analyzer.detect_topic_shifts(
            earnings_transcripts,
            earnings_dates,
            window_size=3
        )

        assert isinstance(shifts, list)

        # Analyze sentiment
        sentiment_map = analyzer.topic_sentiment(
            earnings_transcripts,
            sentiment_scores
        )

        assert len(sentiment_map) == 3

        # Label topics
        analyzer.label_topics({
            0: "Financial Performance",
            1: "Strategic Initiatives",
            2: "Market Environment"
        })

        # Get summaries
        for topic_id in range(3):
            summary = analyzer.get_topic_summary(topic_id)
            assert 'label' in summary
            assert summary['label'] != f"Topic {topic_id}"

    def test_lda_vs_lsi_comparison(self, earnings_transcripts):
        """Test comparing LDA and LSI models."""
        # Fit LSI model
        lsi_analyzer = EarningsTopicAnalyzer(
            n_topics=3,
            model_type='lsi'
        )
        lsi_results = lsi_analyzer.analyze(earnings_transcripts)

        # Fit LDA model
        lda_analyzer = EarningsTopicAnalyzer(
            n_topics=3,
            model_type='lda',
            use_gensim=False
        )
        lda_results = lda_analyzer.analyze(earnings_transcripts)

        # Both should produce valid results
        assert len(lsi_results['topics']) == 3
        assert len(lda_results['topics']) == 3

        # Distributions should have correct shape
        assert lsi_results['distributions'].shape == (len(earnings_transcripts), 3)
        assert lda_results['distributions'].shape == (len(earnings_transcripts), 3)

    def test_multiple_time_periods(self):
        """Test analyzing different time periods separately."""
        # Early period transcripts
        early_transcripts = [
            "traditional business model retail stores physical presence",
            "established market leader competitive advantage scale",
            "steady growth stable margins predictable performance"
        ]

        # Later period transcripts
        later_transcripts = [
            "digital transformation e-commerce platform online growth",
            "technology innovation cloud computing data analytics",
            "disruption agile strategy new business models"
        ]

        # Analyze each period
        early_analyzer = EarningsTopicAnalyzer(n_topics=2, model_type='lsi')
        early_analyzer.analyze(early_transcripts)

        later_analyzer = EarningsTopicAnalyzer(n_topics=2, model_type='lsi')
        later_analyzer.analyze(later_transcripts)

        # Compare topics between periods
        similarity = compare_earnings_topics(early_analyzer, later_analyzer)

        assert similarity.shape == (2, 2)
        # Should show differences in topics between periods
        assert np.all(np.isfinite(similarity))


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_transcript(self):
        """Test with single transcript."""
        analyzer = EarningsTopicAnalyzer(n_topics=2)

        # Should handle gracefully or raise appropriate error
        with pytest.raises(Exception):
            analyzer.analyze(["single transcript here"])

    def test_very_short_transcripts(self):
        """Test with very short transcripts."""
        short_transcripts = ["word", "another", "text"]

        analyzer = EarningsTopicAnalyzer(n_topics=2)

        # Should handle gracefully
        with pytest.raises(Exception):
            analyzer.analyze(short_transcripts)

    def test_topic_shift_small_window(self, earnings_transcripts, earnings_dates):
        """Test shift detection with minimal window."""
        analyzer = EarningsTopicAnalyzer(n_topics=3)
        analyzer.analyze(earnings_transcripts)

        shifts = analyzer.detect_topic_shifts(
            earnings_transcripts,
            earnings_dates,
            window_size=2,
            threshold=0.1
        )

        # Should still work with small window
        assert isinstance(shifts, list)
