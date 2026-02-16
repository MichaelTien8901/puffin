"""
Earnings call topic analysis.

Analyzes topics in earnings call transcripts to identify key themes,
track topic evolution, and detect strategic shifts.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import warnings

from .topic_models import LDAModel, LSIModel

try:
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EarningsTopicAnalyzer:
    """
    Analyzer for discovering and tracking topics in earnings calls.

    Identifies key themes discussed in earnings calls and tracks how
    these themes evolve over time.

    Example:
        >>> analyzer = EarningsTopicAnalyzer(n_topics=5)
        >>> results = analyzer.analyze(transcripts)
        >>> shifts = analyzer.detect_topic_shifts(transcripts, dates)
        >>> sentiment_map = analyzer.topic_sentiment(transcripts, sentiment_scores)
    """

    def __init__(
        self,
        n_topics: int = 10,
        model_type: str = 'lda',
        use_gensim: bool = True
    ):
        """
        Initialize earnings topic analyzer.

        Args:
            n_topics: Number of topics to extract
            model_type: 'lda' or 'lsi'
            use_gensim: Whether to use Gensim for LDA (if available)
        """
        self.n_topics = n_topics
        self.model_type = model_type
        self.use_gensim = use_gensim
        self.model = None
        self.topic_labels = {}

    def analyze(
        self,
        transcripts: List[str],
        dates: Optional[List[Union[str, datetime]]] = None
    ) -> Dict:
        """
        Analyze topics in earnings call transcripts.

        Args:
            transcripts: List of earnings call transcripts
            dates: Optional list of dates for temporal analysis

        Returns:
            Dictionary containing:
                - topics: List of (topic_id, [(word, weight), ...])
                - distributions: Topic distribution matrix
                - dominant_topics: List of dominant topic per transcript
                - temporal_trends: Topic weights over time (if dates provided)
        """
        if not transcripts:
            raise ValueError("No transcripts provided")

        # Fit topic model
        if self.model_type == 'lda':
            self.model = LDAModel(use_gensim=self.use_gensim)
            self.model.fit(transcripts, n_topics=self.n_topics, passes=15)
        else:
            self.model = LSIModel()
            self.model.fit(transcripts, n_topics=self.n_topics)

        # Extract topics
        topics = self.model.get_topics(n_words=10)

        # Get distributions
        distributions = self.model.transform(transcripts)

        # Find dominant topic for each transcript
        dominant_topics = [int(np.argmax(dist)) for dist in distributions]

        results = {
            'topics': topics,
            'distributions': distributions,
            'dominant_topics': dominant_topics,
            'model': self.model
        }

        # Add temporal trends if dates provided
        if dates:
            results['temporal_trends'] = self._compute_temporal_trends(
                distributions, dates
            )

        # Add coherence score for LDA
        if self.model_type == 'lda' and hasattr(self.model, 'coherence_score'):
            try:
                results['coherence_score'] = self.model.coherence_score()
            except:
                results['coherence_score'] = None

        return results

    def detect_topic_shifts(
        self,
        transcripts: List[str],
        dates: List[Union[str, datetime]],
        window_size: int = 4,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        Detect significant topic shifts over time.

        Args:
            transcripts: List of earnings call transcripts
            dates: List of dates corresponding to transcripts
            window_size: Number of consecutive calls to compare
            threshold: Minimum change in topic dominance to flag as shift

        Returns:
            List of shift events with:
                - date: Date of shift
                - old_topic: Previous dominant topic
                - new_topic: New dominant topic
                - change_magnitude: Magnitude of change
        """
        if len(transcripts) != len(dates):
            raise ValueError("Number of transcripts must match number of dates")

        if self.model is None:
            # Fit model if not already fitted
            self.analyze(transcripts, dates)

        # Convert dates if needed
        if dates and isinstance(dates[0], str):
            dates = [datetime.fromisoformat(d.replace('Z', '+00:00')) for d in dates]

        # Get topic distributions
        distributions = self.model.transform(transcripts)

        shifts = []

        for i in range(window_size, len(transcripts)):
            # Compare current window to previous window
            prev_window = distributions[i - window_size:i]
            curr_window = distributions[i - window_size + 1:i + 1]

            # Average distributions in each window
            prev_avg = np.mean(prev_window, axis=0)
            curr_avg = np.mean(curr_window, axis=0)

            # Find dominant topics
            prev_topic = int(np.argmax(prev_avg))
            curr_topic = int(np.argmax(curr_avg))

            # Calculate change magnitude
            change = curr_avg - prev_avg
            magnitude = float(np.linalg.norm(change))

            # Check if there's a significant shift
            if prev_topic != curr_topic and magnitude > threshold:
                shifts.append({
                    'date': dates[i],
                    'old_topic': prev_topic,
                    'new_topic': curr_topic,
                    'change_magnitude': magnitude,
                    'old_weight': float(prev_avg[prev_topic]),
                    'new_weight': float(curr_avg[curr_topic])
                })

        return shifts

    def topic_sentiment(
        self,
        transcripts: List[str],
        sentiment_scores: Optional[List[float]] = None
    ) -> Dict[int, Dict]:
        """
        Analyze sentiment associated with each topic.

        Args:
            transcripts: List of earnings call transcripts
            sentiment_scores: Optional list of sentiment scores per transcript
                            (if None, uses simple polarity analysis)

        Returns:
            Dictionary mapping topic_id to:
                - avg_sentiment: Average sentiment for the topic
                - sentiment_variance: Variance in sentiment
                - document_count: Number of documents where topic is dominant
        """
        if self.model is None:
            self.analyze(transcripts)

        # Get topic distributions
        distributions = self.model.transform(transcripts)

        # Use provided sentiment or compute simple polarity
        if sentiment_scores is None:
            sentiment_scores = self._compute_simple_sentiment(transcripts)

        if len(sentiment_scores) != len(transcripts):
            raise ValueError("Sentiment scores must match transcripts")

        # Map topics to sentiment
        topic_sentiment_map = {}

        for topic_id in range(self.n_topics):
            # Find documents where this topic is dominant
            topic_mask = np.argmax(distributions, axis=1) == topic_id

            if np.sum(topic_mask) == 0:
                # No documents for this topic
                topic_sentiment_map[topic_id] = {
                    'avg_sentiment': 0.0,
                    'sentiment_variance': 0.0,
                    'document_count': 0,
                    'weighted_sentiment': 0.0
                }
                continue

            # Get sentiment scores for this topic
            topic_sentiments = np.array(sentiment_scores)[topic_mask]
            topic_weights = distributions[topic_mask, topic_id]

            # Weighted sentiment
            weighted_sentiment = np.average(topic_sentiments, weights=topic_weights)

            topic_sentiment_map[topic_id] = {
                'avg_sentiment': float(np.mean(topic_sentiments)),
                'sentiment_variance': float(np.var(topic_sentiments)),
                'document_count': int(np.sum(topic_mask)),
                'weighted_sentiment': float(weighted_sentiment)
            }

        return topic_sentiment_map

    def label_topics(
        self,
        labels: Dict[int, str]
    ) -> None:
        """
        Assign human-readable labels to topics.

        Args:
            labels: Dictionary mapping topic_id to label string
        """
        self.topic_labels = labels

    def get_topic_summary(self, topic_id: int, n_words: int = 10) -> Dict:
        """
        Get comprehensive summary of a topic.

        Args:
            topic_id: Topic ID
            n_words: Number of top words to include

        Returns:
            Dictionary with topic summary
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        topics = self.model.get_topics(n_words=n_words)

        # Find the topic
        topic_words = None
        for tid, words in topics:
            if tid == topic_id:
                topic_words = words
                break

        if topic_words is None:
            raise ValueError(f"Topic {topic_id} not found")

        summary = {
            'topic_id': topic_id,
            'label': self.topic_labels.get(topic_id, f'Topic {topic_id}'),
            'top_words': topic_words
        }

        return summary

    def _compute_temporal_trends(
        self,
        distributions: np.ndarray,
        dates: List[Union[str, datetime]]
    ) -> Dict:
        """
        Compute how topic weights evolve over time.

        Args:
            distributions: Topic distribution matrix
            dates: List of dates

        Returns:
            Dictionary with temporal trend data
        """
        # Convert dates if needed
        if dates and isinstance(dates[0], str):
            dates = [datetime.fromisoformat(d.replace('Z', '+00:00')) for d in dates]

        # Sort by date
        sorted_indices = np.argsort(dates)
        sorted_dates = [dates[i] for i in sorted_indices]
        sorted_distributions = distributions[sorted_indices]

        # Compute trends for each topic
        trends = {}

        for topic_id in range(self.n_topics):
            topic_weights = sorted_distributions[:, topic_id]

            # Simple linear trend
            x = np.arange(len(topic_weights))
            if len(x) > 1 and SKLEARN_AVAILABLE:
                # Compute trend direction
                correlation = np.corrcoef(x, topic_weights)[0, 1]
                trend_direction = 'increasing' if correlation > 0.1 else (
                    'decreasing' if correlation < -0.1 else 'stable'
                )
            else:
                trend_direction = 'unknown'

            trends[topic_id] = {
                'weights': topic_weights.tolist(),
                'dates': [d.isoformat() for d in sorted_dates],
                'trend_direction': trend_direction,
                'avg_weight': float(np.mean(topic_weights)),
                'max_weight': float(np.max(topic_weights)),
                'min_weight': float(np.min(topic_weights))
            }

        return trends

    def _compute_simple_sentiment(self, transcripts: List[str]) -> List[float]:
        """
        Compute simple sentiment scores based on positive/negative word counts.

        Args:
            transcripts: List of transcripts

        Returns:
            List of sentiment scores (-1 to 1)
        """
        # Simple positive/negative word lists
        positive_words = {
            'growth', 'strong', 'improved', 'excellent', 'positive',
            'increase', 'gain', 'success', 'opportunity', 'momentum',
            'outperform', 'exceed', 'beat', 'profit', 'revenue'
        }

        negative_words = {
            'decline', 'weak', 'decreased', 'negative', 'loss',
            'challenge', 'risk', 'concern', 'miss', 'below',
            'underperform', 'difficult', 'pressure', 'headwind'
        }

        sentiments = []

        for transcript in transcripts:
            words = transcript.lower().split()

            pos_count = sum(1 for w in words if w in positive_words)
            neg_count = sum(1 for w in words if w in negative_words)

            total = pos_count + neg_count
            if total == 0:
                sentiment = 0.0
            else:
                sentiment = (pos_count - neg_count) / total

            sentiments.append(sentiment)

        return sentiments


def compare_earnings_topics(
    analyzer1: EarningsTopicAnalyzer,
    analyzer2: EarningsTopicAnalyzer,
    method: str = 'cosine'
) -> np.ndarray:
    """
    Compare topic models from different time periods or companies.

    Args:
        analyzer1: First analyzer
        analyzer2: Second analyzer
        method: Comparison method ('cosine', 'euclidean')

    Returns:
        Similarity matrix between topics of the two models
    """
    if analyzer1.model is None or analyzer2.model is None:
        raise ValueError("Both analyzers must have fitted models")

    # Get topic-word distributions
    topics1 = analyzer1.model.get_topics(n_words=100)
    topics2 = analyzer2.model.get_topics(n_words=100)

    # Create word-weight dictionaries
    def topics_to_dict(topics):
        return {tid: {word: weight for word, weight in words}
                for tid, words in topics}

    dict1 = topics_to_dict(topics1)
    dict2 = topics_to_dict(topics2)

    # Get all unique words
    all_words = set()
    for words_dict in dict1.values():
        all_words.update(words_dict.keys())
    for words_dict in dict2.values():
        all_words.update(words_dict.keys())

    word_list = sorted(all_words)
    word_to_idx = {word: idx for idx, word in enumerate(word_list)}

    # Create topic vectors
    def create_vectors(topic_dict, word_to_idx):
        vectors = []
        for topic_id in sorted(topic_dict.keys()):
            vector = np.zeros(len(word_to_idx))
            for word, weight in topic_dict[topic_id].items():
                if word in word_to_idx:
                    vector[word_to_idx[word]] = weight
            vectors.append(vector)
        return np.array(vectors)

    vectors1 = create_vectors(dict1, word_to_idx)
    vectors2 = create_vectors(dict2, word_to_idx)

    # Normalize vectors
    if SKLEARN_AVAILABLE:
        vectors1 = normalize(vectors1, norm='l2')
        vectors2 = normalize(vectors2, norm='l2')

    # Compute similarity matrix
    if method == 'cosine':
        similarity = np.dot(vectors1, vectors2.T)
    elif method == 'euclidean':
        similarity = -np.linalg.norm(
            vectors1[:, np.newaxis] - vectors2[np.newaxis, :],
            axis=2
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return similarity
