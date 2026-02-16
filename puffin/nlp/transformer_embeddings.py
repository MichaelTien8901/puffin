"""
Transformer-based embeddings for financial text analysis.

This module provides BERT and FinBERT embeddings using HuggingFace transformers
for state-of-the-art text representations.
"""

import numpy as np
import warnings
from typing import List, Union, Optional
from scipy.spatial.distance import cosine

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "Transformers not available. Install with: pip install transformers torch"
    )


class TransformerEmbedder:
    """
    Generate embeddings using transformer models (BERT, FinBERT, etc.).

    This class provides a unified interface for generating embeddings using
    various transformer models, with special support for financial text.

    Attributes:
        model_name: Name of the loaded model
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        device: Device for computation (cpu/cuda)

    Example:
        >>> embedder = TransformerEmbedder()
        >>> embeddings = embedder.encode(['Market volatility increased.'])
        >>> similarity = embedder.similarity('bull market', 'bear market')
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize TransformerEmbedder.

        Args:
            device: Device for computation ('cpu', 'cuda', or None for auto-detect)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers library is required. "
                "Install with: pip install transformers torch"
            )

        self.model_name: Optional[str] = None
        self.tokenizer = None
        self.model = None

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

    def _load_model(self, model_name: str) -> None:
        """
        Load a transformer model and tokenizer.

        Args:
            model_name: HuggingFace model identifier
        """
        if self.model_name == model_name and self.model is not None:
            return  # Model already loaded

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.model_name = model_name
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

    def _mean_pooling(
        self,
        model_output,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform mean pooling on model output.

        Args:
            model_output: Model output (last_hidden_state)
            attention_mask: Attention mask

        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]  # First element is last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(
        self,
        texts: Union[str, List[str]],
        model_name: str = 'distilbert-base-uncased',
        max_length: int = 512,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Text or list of texts to encode
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            batch_size: Batch size for processing

        Returns:
            Array of embeddings (num_texts, embedding_dim)

        Example:
            >>> texts = ['Market is bullish.', 'Prices are falling.']
            >>> embeddings = embedder.encode(texts)
            >>> print(embeddings.shape)
            (2, 768)
        """
        # Load model if needed
        self._load_model(model_name)

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            # Mean pooling
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

        # Concatenate all batches
        return np.vstack(all_embeddings)

    def encode_financial(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode financial texts using FinBERT or fallback to DistilBERT.

        FinBERT is a BERT model pre-trained on financial text, providing
        better representations for financial documents.

        Args:
            texts: Text or list of texts to encode
            max_length: Maximum sequence length
            batch_size: Batch size for processing

        Returns:
            Array of embeddings

        Example:
            >>> text = "Company reported strong quarterly earnings."
            >>> embedding = embedder.encode_financial(text)
        """
        # Try FinBERT first, fallback to DistilBERT
        finbert_models = [
            'ProsusAI/finbert',
            'yiyanghkust/finbert-tone',
            'distilbert-base-uncased'
        ]

        for model_name in finbert_models:
            try:
                return self.encode(
                    texts,
                    model_name=model_name,
                    max_length=max_length,
                    batch_size=batch_size
                )
            except Exception as e:
                if model_name == finbert_models[-1]:
                    # Last model failed
                    raise RuntimeError(f"All models failed. Last error: {e}")
                continue

    def similarity(
        self,
        text1: str,
        text2: str,
        model_name: str = 'distilbert-base-uncased'
    ) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            model_name: HuggingFace model name

        Returns:
            Cosine similarity score (-1 to 1)

        Example:
            >>> sim = embedder.similarity('bull market', 'rising prices')
            >>> print(f"Similarity: {sim:.3f}")
        """
        embeddings = self.encode([text1, text2], model_name=model_name)
        return float(1 - cosine(embeddings[0], embeddings[1]))

    def batch_similarity(
        self,
        texts1: List[str],
        texts2: List[str],
        model_name: str = 'distilbert-base-uncased'
    ) -> np.ndarray:
        """
        Calculate pairwise cosine similarities between two lists of texts.

        Args:
            texts1: First list of texts
            texts2: Second list of texts
            model_name: HuggingFace model name

        Returns:
            Similarity matrix (len(texts1), len(texts2))

        Example:
            >>> texts1 = ['bull market', 'bear market']
            >>> texts2 = ['rising prices', 'falling prices']
            >>> sims = embedder.batch_similarity(texts1, texts2)
        """
        embeddings1 = self.encode(texts1, model_name=model_name)
        embeddings2 = self.encode(texts2, model_name=model_name)

        # Compute cosine similarity matrix
        similarities = np.dot(embeddings1, embeddings2.T)
        return similarities

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        model_name: str = 'distilbert-base-uncased',
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar texts to a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            model_name: HuggingFace model name
            top_k: Number of results to return

        Returns:
            List of (text, similarity_score, index) tuples

        Example:
            >>> query = "increased market volatility"
            >>> candidates = ["volatile market", "stable prices", "uncertain conditions"]
            >>> results = embedder.find_similar(query, candidates, top_k=2)
        """
        # Encode query and candidates
        query_emb = self.encode(query, model_name=model_name)
        candidate_embs = self.encode(candidates, model_name=model_name)

        # Calculate similarities
        similarities = np.dot(candidate_embs, query_emb.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results
        results = [
            (candidates[idx], float(similarities[idx]), int(idx))
            for idx in top_indices
        ]
        return results

    def semantic_search(
        self,
        query: str,
        corpus: List[str],
        model_name: str = 'distilbert-base-uncased',
        top_k: int = 10
    ) -> List[dict]:
        """
        Perform semantic search over a corpus of texts.

        Args:
            query: Search query
            corpus: List of texts to search
            model_name: HuggingFace model name
            top_k: Number of results to return

        Returns:
            List of dictionaries with 'text', 'score', and 'index'

        Example:
            >>> corpus = ["Market volatility increased...", "Prices remained stable..."]
            >>> results = embedder.semantic_search("volatility", corpus, top_k=5)
        """
        results = self.find_similar(query, corpus, model_name, top_k)

        return [
            {'text': text, 'score': score, 'index': idx}
            for text, score, idx in results
        ]

    def cluster_texts(
        self,
        texts: List[str],
        n_clusters: int = 5,
        model_name: str = 'distilbert-base-uncased'
    ) -> np.ndarray:
        """
        Cluster texts using their embeddings.

        Args:
            texts: List of texts to cluster
            n_clusters: Number of clusters
            model_name: HuggingFace model name

        Returns:
            Array of cluster labels

        Example:
            >>> texts = ["bull market", "bear market", "rising prices", "falling prices"]
            >>> labels = embedder.cluster_texts(texts, n_clusters=2)
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("scikit-learn is required for clustering. Install with: pip install scikit-learn")

        # Get embeddings
        embeddings = self.encode(texts, model_name=model_name)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        return labels


class SentenceTransformerEmbedder:
    """
    Simplified interface using sentence-transformers library.

    This class provides a more efficient interface for sentence embeddings
    using the sentence-transformers library, which is optimized for
    semantic similarity tasks.

    Example:
        >>> embedder = SentenceTransformerEmbedder()
        >>> embeddings = embedder.encode(['Market is volatile.'])
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize SentenceTransformerEmbedder.

        Args:
            model_name: Sentence transformer model name
            device: Device for computation
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'

        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Text or list of texts
            batch_size: Batch size for processing
            show_progress_bar: Show progress bar

        Returns:
            Array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self.encode([text1, text2])
        return float(1 - cosine(embeddings[0], embeddings[1]))
