"""
Topic model visualization tools.

Provides visualization functions for topic distributions, evolution over time,
and interactive exploration with pyLDAvis.
"""

import numpy as np
from typing import List, Optional, Union, Any
import warnings
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Install with: pip install matplotlib")

try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False
    warnings.warn("pyLDAvis not available. Install with: pip install pyldavis")

from .topic_models import LSIModel, LDAModel


def plot_topic_distribution(
    model: Union[LSIModel, LDAModel],
    document: str,
    figsize: tuple = (10, 6),
    title: Optional[str] = None
) -> Any:
    """
    Plot topic distribution for a single document as a bar chart.

    Args:
        model: Fitted topic model (LSI or LDA)
        document: Text document to analyze
        figsize: Figure size (width, height)
        title: Optional plot title

    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    # Get topic weights/distribution
    topic_weights = model.transform([document])[0]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    topics = list(range(len(topic_weights)))
    ax.bar(topics, topic_weights, color='steelblue', alpha=0.7)

    ax.set_xlabel('Topic ID', fontsize=12)
    ax.set_ylabel('Topic Weight/Probability', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Topic Distribution', fontsize=14, fontweight='bold')

    ax.set_xticks(topics)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    return fig


def plot_topic_evolution(
    model: Union[LSIModel, LDAModel],
    documents: List[str],
    dates: List[Union[str, datetime]],
    topic_ids: Optional[List[int]] = None,
    figsize: tuple = (12, 6),
    title: Optional[str] = None,
    rolling_window: Optional[int] = None
) -> Any:
    """
    Plot topic weights over time.

    Args:
        model: Fitted topic model
        documents: List of text documents
        dates: List of dates corresponding to documents
        topic_ids: Optional list of specific topic IDs to plot (default: all)
        figsize: Figure size (width, height)
        title: Optional plot title
        rolling_window: Optional window size for rolling average smoothing

    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    # Convert dates if needed
    if dates and isinstance(dates[0], str):
        dates = [datetime.fromisoformat(d.replace('Z', '+00:00')) for d in dates]

    # Get topic distributions
    topic_distributions = model.transform(documents)

    # Select topics to plot
    if topic_ids is None:
        topic_ids = list(range(topic_distributions.shape[1]))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each topic
    colors = plt.cm.tab10(np.linspace(0, 1, len(topic_ids)))

    for idx, topic_id in enumerate(topic_ids):
        topic_weights = topic_distributions[:, topic_id]

        # Apply rolling average if specified
        if rolling_window and rolling_window > 1:
            topic_weights = np.convolve(
                topic_weights,
                np.ones(rolling_window) / rolling_window,
                mode='valid'
            )
            plot_dates = dates[rolling_window - 1:]
        else:
            plot_dates = dates

        ax.plot(
            plot_dates,
            topic_weights,
            label=f'Topic {topic_id}',
            color=colors[idx],
            linewidth=2,
            alpha=0.7
        )

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Topic Weight/Probability', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Topic Evolution Over Time', fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    return fig


def plot_topic_heatmap(
    model: Union[LSIModel, LDAModel],
    documents: List[str],
    labels: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
    title: Optional[str] = None,
    cmap: str = 'YlOrRd'
) -> Any:
    """
    Plot heatmap of document-topic distributions.

    Args:
        model: Fitted topic model
        documents: List of text documents
        labels: Optional list of document labels
        figsize: Figure size (width, height)
        title: Optional plot title
        cmap: Colormap name

    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    # Get topic distributions
    topic_distributions = model.transform(documents)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(topic_distributions.T, aspect='auto', cmap=cmap)

    # Set labels
    ax.set_xlabel('Document', fontsize=12)
    ax.set_ylabel('Topic ID', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Document-Topic Distribution Heatmap', fontsize=14, fontweight='bold')

    # Set y-axis ticks
    ax.set_yticks(range(topic_distributions.shape[1]))

    # Set x-axis labels if provided
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, ha='right')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Topic Weight/Probability', fontsize=10)

    plt.tight_layout()

    return fig


def plot_topic_words(
    model: Union[LSIModel, LDAModel],
    topic_id: int,
    n_words: int = 15,
    figsize: tuple = (10, 6),
    title: Optional[str] = None
) -> Any:
    """
    Plot top words for a specific topic.

    Args:
        model: Fitted topic model
        topic_id: Topic ID to visualize
        n_words: Number of top words to show
        figsize: Figure size (width, height)
        title: Optional plot title

    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    # Get topics
    topics = model.get_topics(n_words=n_words)

    # Find the specified topic
    topic_words = None
    for tid, words in topics:
        if tid == topic_id:
            topic_words = words
            break

    if topic_words is None:
        raise ValueError(f"Topic {topic_id} not found")

    # Extract words and weights
    words = [word for word, _ in topic_words]
    weights = [weight for _, weight in topic_words]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(words))
    ax.barh(y_pos, weights, color='steelblue', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()  # Top word at top

    ax.set_xlabel('Weight', fontsize=12)
    ax.set_ylabel('Word', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Top Words for Topic {topic_id}', fontsize=14, fontweight='bold')

    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    return fig


def prepare_pyldavis(model: LDAModel) -> Optional[Any]:
    """
    Prepare pyLDAvis visualization for interactive exploration.

    Args:
        model: Fitted LDA model (must use Gensim backend)

    Returns:
        pyLDAvis prepared data, or None if not available

    Example:
        >>> model = LDAModel()
        >>> model.fit(documents, n_topics=5)
        >>> vis_data = prepare_pyldavis(model)
        >>> if vis_data:
        ...     pyLDAvis.display(vis_data)
    """
    if not PYLDAVIS_AVAILABLE:
        warnings.warn("pyLDAvis not available. Install with: pip install pyldavis")
        return None

    if not isinstance(model, LDAModel):
        warnings.warn("pyLDAvis only supports LDA models")
        return None

    if not model.use_gensim:
        warnings.warn("pyLDAvis requires Gensim-based LDA model")
        return None

    if model.model is None:
        raise ValueError("Model must be fitted before visualization")

    try:
        # Prepare corpus
        texts = [doc.lower().split() for doc in model.documents]
        corpus = [model.dictionary.doc2bow(text) for text in texts]

        # Prepare visualization
        vis_data = gensimvis.prepare(
            model.model,
            corpus,
            model.dictionary,
            sort_topics=False
        )

        return vis_data

    except Exception as e:
        warnings.warn(f"Failed to prepare pyLDAvis: {e}")
        return None


def save_pyldavis_html(
    vis_data: Any,
    output_path: str
) -> None:
    """
    Save pyLDAvis visualization to HTML file.

    Args:
        vis_data: Prepared pyLDAvis data
        output_path: Path to save HTML file
    """
    if not PYLDAVIS_AVAILABLE:
        raise ImportError("pyLDAvis is required")

    pyLDAvis.save_html(vis_data, output_path)


def plot_coherence_scores(
    coherence_results: List[tuple],
    figsize: tuple = (10, 6),
    title: Optional[str] = None
) -> Any:
    """
    Plot coherence scores for different numbers of topics.

    Args:
        coherence_results: List of (n_topics, coherence_score) tuples
        figsize: Figure size (width, height)
        title: Optional plot title

    Returns:
        Matplotlib figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization")

    n_topics_list = [n for n, _ in coherence_results]
    scores = [score for _, score in coherence_results]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(n_topics_list, scores, marker='o', linewidth=2,
            markersize=8, color='steelblue')

    # Mark optimal
    optimal_idx = np.argmax(scores)
    ax.scatter(
        n_topics_list[optimal_idx],
        scores[optimal_idx],
        color='red',
        s=200,
        marker='*',
        label=f'Optimal: {n_topics_list[optimal_idx]} topics',
        zorder=5
    )

    ax.set_xlabel('Number of Topics', fontsize=12)
    ax.set_ylabel('Coherence Score', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Topic Coherence vs Number of Topics', fontsize=14, fontweight='bold')

    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig
