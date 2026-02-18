"""
Topic modeling for financial text analysis.

Provides LSI and LDA topic models for discovering latent themes in
earnings calls, financial news, and other text data.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import warnings

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaMulticore, LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("Gensim not available. Install with: pip install gensim")


class LSIModel:
    """
    Latent Semantic Indexing (LSI) topic model using TruncatedSVD.

    LSI discovers latent topics by performing singular value decomposition
    on a TF-IDF matrix of documents.

    Example:
        >>> documents = ["earnings improved significantly",
        ...              "revenue growth exceeded expectations",
        ...              "market volatility increased"]
        >>> model = LSIModel()
        >>> model.fit(documents, n_topics=2)
        >>> topics = model.get_topics(n_words=5)
        >>> weights = model.transform(["strong revenue growth"])
    """

    def __init__(self):
        """Initialize LSI model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for LSIModel")

        self.vectorizer = None
        self.svd = None
        self.n_topics = None
        self.feature_names = None

    def fit(self, documents: List[str], n_topics: int = 10) -> 'LSIModel':
        """
        Fit LSI model to documents.

        Args:
            documents: List of text documents
            n_topics: Number of topics to extract

        Returns:
            self: Fitted model
        """
        self.n_topics = n_topics

        # Scale min_df with corpus size to avoid empty vocabulary on small corpora
        min_df = min(2, max(1, len(documents) // 3))

        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            min_df=min_df,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Perform SVD (cap n_topics at available features)
        n_components = min(n_topics, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        n_components = max(1, n_components)
        self.n_topics = n_components
        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )
        self.svd.fit(tfidf_matrix)

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to topic weights.

        Args:
            documents: List of text documents

        Returns:
            Topic weights matrix (n_documents x n_topics)
        """
        if self.svd is None:
            raise ValueError("Model must be fitted before transform")

        tfidf_matrix = self.vectorizer.transform(documents)
        topic_weights = self.svd.transform(tfidf_matrix)

        return topic_weights

    def get_topics(self, n_words: int = 10) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """
        Get top words for each topic.

        Args:
            n_words: Number of top words per topic

        Returns:
            List of (topic_id, [(word, weight), ...])
        """
        if self.svd is None:
            raise ValueError("Model must be fitted before getting topics")

        topics = []

        for topic_idx, component in enumerate(self.svd.components_):
            # Get top word indices (cap at available features)
            actual_n_words = min(n_words, len(self.feature_names))
            top_indices = np.argsort(np.abs(component))[::-1][:actual_n_words]

            # Get words and weights
            top_words = [
                (self.feature_names[idx], float(component[idx]))
                for idx in top_indices
            ]

            topics.append((topic_idx, top_words))

        return topics

    def explained_variance_ratio(self) -> np.ndarray:
        """
        Get explained variance ratio for each topic, sorted in descending order.

        Returns:
            Array of explained variance ratios (descending)
        """
        if self.svd is None:
            raise ValueError("Model must be fitted")

        # Sort in descending order to ensure consistency
        return np.sort(self.svd.explained_variance_ratio_)[::-1]


class LDAModel:
    """
    Latent Dirichlet Allocation (LDA) topic model.

    Uses Gensim's LdaMulticore for efficient parallel processing,
    with graceful fallback to sklearn's LatentDirichletAllocation.

    Example:
        >>> documents = ["earnings improved significantly",
        ...              "revenue growth exceeded expectations",
        ...              "market volatility increased"]
        >>> model = LDAModel()
        >>> model.fit(documents, n_topics=2)
        >>> topics = model.get_topics(n_words=5)
        >>> distributions = model.transform(["strong revenue growth"])
        >>> coherence = model.coherence_score()
    """

    def __init__(self, use_gensim: bool = True):
        """
        Initialize LDA model.

        Args:
            use_gensim: Whether to use Gensim (True) or sklearn (False)
        """
        self.use_gensim = use_gensim and GENSIM_AVAILABLE

        if use_gensim and not GENSIM_AVAILABLE:
            warnings.warn("Gensim not available, falling back to sklearn")
            self.use_gensim = False

        if not self.use_gensim and not SKLEARN_AVAILABLE:
            raise ImportError("Either gensim or scikit-learn is required")

        self.model = None
        self.dictionary = None
        self.vectorizer = None
        self.n_topics = None
        self.documents = None
        self.feature_names = None

    def fit(self, documents: List[str], n_topics: int = 10, passes: int = 15) -> 'LDAModel':
        """
        Fit LDA model to documents.

        Args:
            documents: List of text documents
            n_topics: Number of topics to extract
            passes: Number of passes through the corpus (Gensim only)

        Returns:
            self: Fitted model
        """
        self.n_topics = n_topics
        self.documents = documents

        if self.use_gensim:
            # Tokenize documents
            texts = [doc.lower().split() for doc in documents]

            # Create dictionary and corpus
            self.dictionary = corpora.Dictionary(texts)

            # Filter extremes (scale threshold with corpus size)
            n_docs = len(texts)
            no_below = min(2, max(1, n_docs // 5))
            self.dictionary.filter_extremes(no_below=no_below, no_above=0.8)

            corpus = [self.dictionary.doc2bow(text) for text in texts]

            # Fit LDA model
            try:
                self.model = LdaMulticore(
                    corpus=corpus,
                    id2word=self.dictionary,
                    num_topics=n_topics,
                    passes=passes,
                    workers=2,
                    random_state=42,
                    alpha='asymmetric',
                    per_word_topics=True
                )
            except:
                # Fallback to single-core LDA
                self.model = LdaModel(
                    corpus=corpus,
                    id2word=self.dictionary,
                    num_topics=n_topics,
                    passes=passes,
                    random_state=42,
                    alpha='asymmetric',
                    per_word_topics=True
                )
        else:
            # Use sklearn
            self.vectorizer = CountVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.8,
                stop_words='english',
                ngram_range=(1, 1)
            )

            count_matrix = self.vectorizer.fit_transform(documents)
            self.feature_names = self.vectorizer.get_feature_names_out()

            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
                learning_method='batch',
                n_jobs=-1
            )

            self.model.fit(count_matrix)

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to topic distributions.

        Args:
            documents: List of text documents

        Returns:
            Topic distribution matrix (n_documents x n_topics)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before transform")

        if self.use_gensim:
            texts = [doc.lower().split() for doc in documents]
            corpus = [self.dictionary.doc2bow(text) for text in texts]

            # Get topic distributions
            distributions = []
            for doc_bow in corpus:
                topic_dist = self.model.get_document_topics(
                    doc_bow,
                    minimum_probability=0.0
                )
                dist_array = np.zeros(self.n_topics)
                for topic_id, prob in topic_dist:
                    dist_array[topic_id] = prob
                distributions.append(dist_array)

            return np.array(distributions)
        else:
            count_matrix = self.vectorizer.transform(documents)
            return self.model.transform(count_matrix)

    def get_topics(self, n_words: int = 10) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """
        Get top words for each topic.

        Args:
            n_words: Number of top words per topic

        Returns:
            List of (topic_id, [(word, weight), ...])
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting topics")

        topics = []

        if self.use_gensim:
            for topic_id in range(self.n_topics):
                topic_words = self.model.show_topic(topic_id, topn=n_words)
                topics.append((topic_id, topic_words))
        else:
            for topic_idx, topic in enumerate(self.model.components_):
                top_indices = np.argsort(topic)[::-1][:n_words]
                top_words = [
                    (self.feature_names[idx], float(topic[idx]))
                    for idx in top_indices
                ]
                topics.append((topic_idx, top_words))

        return topics

    def coherence_score(self, coherence_type: str = 'c_v') -> float:
        """
        Calculate topic coherence score.

        Args:
            coherence_type: Type of coherence ('c_v', 'u_mass', 'c_uci', 'c_npmi')

        Returns:
            Coherence score (higher is better)
        """
        if self.model is None:
            raise ValueError("Model must be fitted")

        if not self.use_gensim:
            warnings.warn("Coherence score only available with Gensim, returning 0.0")
            return 0.0

        texts = [doc.lower().split() for doc in self.documents]
        corpus = [self.dictionary.doc2bow(text) for text in texts]

        coherence_model = CoherenceModel(
            model=self.model,
            texts=texts,
            dictionary=self.dictionary,
            coherence=coherence_type
        )

        return coherence_model.get_coherence()

    def perplexity(self) -> float:
        """
        Calculate perplexity on training corpus.

        Returns:
            Perplexity score (lower is better)
        """
        if self.model is None:
            raise ValueError("Model must be fitted")

        if self.use_gensim:
            texts = [doc.lower().split() for doc in self.documents]
            corpus = [self.dictionary.doc2bow(text) for text in texts]
            return self.model.log_perplexity(corpus)
        else:
            count_matrix = self.vectorizer.transform(self.documents)
            return self.model.perplexity(count_matrix)


def find_optimal_topics(
    documents: List[str],
    min_topics: int = 2,
    max_topics: int = 20,
    step: int = 2,
    method: str = 'lda'
) -> Tuple[int, List[Tuple[int, float]]]:
    """
    Find optimal number of topics using coherence score.

    Args:
        documents: List of text documents
        min_topics: Minimum number of topics to try
        max_topics: Maximum number of topics to try
        step: Step size for topic range
        method: 'lda' or 'lsi'

    Returns:
        Tuple of (optimal_n_topics, [(n_topics, coherence_score), ...])
    """
    coherence_scores = []

    for n_topics in range(min_topics, max_topics + 1, step):
        if method == 'lda':
            model = LDAModel(use_gensim=True)
            model.fit(documents, n_topics=n_topics, passes=10)
            if model.use_gensim:
                score = model.coherence_score()
            else:
                score = 0.0
        else:
            model = LSIModel()
            model.fit(documents, n_topics=n_topics)
            # Use explained variance as proxy
            score = float(np.sum(model.explained_variance_ratio()))

        coherence_scores.append((n_topics, score))

    # Find optimal
    optimal_idx = np.argmax([score for _, score in coherence_scores])
    optimal_n_topics = coherence_scores[optimal_idx][0]

    return optimal_n_topics, coherence_scores
