"""NLP for trading tasks."""

from puffin.nlp.classifier import NewsClassifier
from puffin.nlp.pipeline import NLPPipeline, ProcessedDoc
from puffin.nlp.sentiment import LexiconSentiment, RuleSentiment
from puffin.nlp.vectorizers import (
    DocumentTermMatrix,
    build_bow,
    build_tfidf,
)
from puffin.nlp.topic_models import LSIModel, LDAModel, find_optimal_topics
from puffin.nlp.topic_viz import (
    plot_topic_distribution,
    plot_topic_evolution,
    plot_topic_heatmap,
    plot_topic_words,
    plot_coherence_scores,
    prepare_pyldavis,
    save_pyldavis_html,
)
from puffin.nlp.earnings_topics import EarningsTopicAnalyzer, compare_earnings_topics

# Optional imports (require additional dependencies)
try:
    from puffin.nlp.embeddings import Word2VecTrainer, GloVeLoader, Doc2VecTrainer
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

try:
    from puffin.nlp.sec_analysis import SECFilingAnalyzer
    _SEC_ANALYSIS_AVAILABLE = True
except ImportError:
    _SEC_ANALYSIS_AVAILABLE = False

try:
    from puffin.nlp.transformer_embeddings import TransformerEmbedder, SentenceTransformerEmbedder
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

__all__ = [
    "NLPPipeline",
    "ProcessedDoc",
    "build_bow",
    "build_tfidf",
    "DocumentTermMatrix",
    "NewsClassifier",
    "RuleSentiment",
    "LexiconSentiment",
    "LSIModel",
    "LDAModel",
    "find_optimal_topics",
    "plot_topic_distribution",
    "plot_topic_evolution",
    "plot_topic_heatmap",
    "plot_topic_words",
    "plot_coherence_scores",
    "prepare_pyldavis",
    "save_pyldavis_html",
    "EarningsTopicAnalyzer",
    "compare_earnings_topics",
]

# Add optional exports if available
if _EMBEDDINGS_AVAILABLE:
    __all__.extend(["Word2VecTrainer", "GloVeLoader", "Doc2VecTrainer"])

if _SEC_ANALYSIS_AVAILABLE:
    __all__.extend(["SECFilingAnalyzer"])

if _TRANSFORMERS_AVAILABLE:
    __all__.extend(["TransformerEmbedder", "SentenceTransformerEmbedder"])
