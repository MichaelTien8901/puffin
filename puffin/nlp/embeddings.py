"""
Word embeddings for financial text analysis.

This module provides implementations for training and using word2vec, GloVe, and doc2vec
embeddings for financial text analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import warnings

try:
    from gensim.models import Word2Vec, Doc2Vec
    from gensim.models.doc2vec import TaggedDocument
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    warnings.warn("Gensim not available. Install with: pip install gensim")


class Word2VecTrainer:
    """
    Train and use Word2Vec embeddings with Gensim.

    Word2Vec learns distributed representations of words by training a shallow neural network
    to predict words from their context (CBOW) or context from words (Skip-gram).

    Attributes:
        model: Trained Gensim Word2Vec model

    Example:
        >>> documents = [['market', 'volatility', 'increased'], ['stock', 'price', 'dropped']]
        >>> trainer = Word2VecTrainer()
        >>> model = trainer.train(documents, vector_size=100, window=5, sg=1)
        >>> vector = trainer.word_vector('market')
        >>> similar = trainer.similar_words('market', topn=5)
    """

    def __init__(self):
        """Initialize Word2VecTrainer."""
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required for Word2VecTrainer. Install with: pip install gensim")
        self.model = None

    def train(
        self,
        documents: List[List[str]],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        sg: int = 1,
        workers: int = 4,
        epochs: int = 10,
        **kwargs
    ) -> Word2Vec:
        """
        Train Word2Vec model on documents.

        Args:
            documents: List of tokenized documents (list of lists of words)
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Ignores words with frequency lower than this
            sg: Training algorithm (1=skip-gram, 0=CBOW)
            workers: Number of worker threads
            epochs: Number of training epochs
            **kwargs: Additional arguments passed to Word2Vec

        Returns:
            Trained Word2Vec model

        Example:
            >>> documents = [['the', 'stock', 'market'], ['trading', 'volume']]
            >>> trainer = Word2VecTrainer()
            >>> model = trainer.train(documents, vector_size=50, sg=1)
        """
        self.model = Word2Vec(
            sentences=documents,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=workers,
            epochs=epochs,
            **kwargs
        )
        return self.model

    def load(self, path: Union[str, Path]) -> Word2Vec:
        """
        Load a trained Word2Vec model from disk.

        Args:
            path: Path to saved model file

        Returns:
            Loaded Word2Vec model

        Example:
            >>> trainer = Word2VecTrainer()
            >>> model = trainer.load('word2vec.model')
        """
        self.model = Word2Vec.load(str(path))
        return self.model

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model

        Example:
            >>> trainer.save('word2vec.model')
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        self.model.save(str(path))

    def similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words to a given word.

        Args:
            word: Word to find similar words for
            topn: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples

        Example:
            >>> similar = trainer.similar_words('market', topn=5)
            >>> print(similar)
            [('markets', 0.85), ('trading', 0.78), ...]
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            return []

    def word_vector(self, word: str) -> np.ndarray:
        """
        Get the vector representation of a word.

        Args:
            word: Word to get vector for

        Returns:
            Vector representation of the word

        Raises:
            KeyError: If word not in vocabulary

        Example:
            >>> vector = trainer.word_vector('market')
            >>> print(vector.shape)
            (100,)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        return self.model.wv[word]

    def document_vector(self, doc: List[str]) -> np.ndarray:
        """
        Get document vector by averaging word vectors.

        Args:
            doc: Tokenized document (list of words)

        Returns:
            Document vector (average of word vectors)

        Example:
            >>> doc = ['market', 'volatility', 'increased']
            >>> doc_vec = trainer.document_vector(doc)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        vectors = []
        for word in doc:
            try:
                vectors.append(self.model.wv[word])
            except KeyError:
                continue

        if not vectors:
            # Return zero vector if no words in vocabulary
            return np.zeros(self.model.vector_size)

        return np.mean(vectors, axis=0)

    def analogy(self, positive: List[str], negative: List[str], topn: int = 1) -> List[Tuple[str, float]]:
        """
        Solve word analogies (e.g., king - man + woman = queen).

        Args:
            positive: Words to add
            negative: Words to subtract
            topn: Number of results to return

        Returns:
            List of (word, score) tuples

        Example:
            >>> # bull market - bull + bear = bear market
            >>> result = trainer.analogy(['bull', 'market'], ['bull'])
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        try:
            return self.model.wv.most_similar(positive=positive, negative=negative, topn=topn)
        except KeyError:
            return []


class GloVeLoader:
    """
    Load and use pretrained GloVe embeddings.

    GloVe (Global Vectors for Word Representation) learns word embeddings by factorizing
    the word co-occurrence matrix.

    Attributes:
        vectors: Dictionary mapping words to their vector representations
        vector_size: Dimensionality of word vectors

    Example:
        >>> loader = GloVeLoader()
        >>> loader.load('glove.6B.100d.txt')
        >>> vector = loader.word_vector('market')
        >>> similar = loader.similar_words('market', topn=5)
    """

    def __init__(self):
        """Initialize GloVeLoader."""
        self.vectors: Dict[str, np.ndarray] = {}
        self.vector_size: int = 0

    def load(self, path: Union[str, Path]) -> 'GloVeLoader':
        """
        Load pretrained GloVe vectors from file.

        Expected format: word vec1 vec2 vec3 ... (space-separated)

        Args:
            path: Path to GloVe vectors file

        Returns:
            Self for method chaining

        Example:
            >>> loader = GloVeLoader()
            >>> loader.load('glove.6B.100d.txt')
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GloVe file not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                word = parts[0]
                try:
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    self.vectors[word] = vector
                    if self.vector_size == 0:
                        self.vector_size = len(vector)
                except ValueError:
                    continue

        return self

    def word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the vector representation of a word.

        Args:
            word: Word to get vector for

        Returns:
            Vector representation, or None if word not found

        Example:
            >>> vector = loader.word_vector('market')
            >>> if vector is not None:
            ...     print(vector.shape)
        """
        return self.vectors.get(word)

    def similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar words using cosine similarity.

        Args:
            word: Word to find similar words for
            topn: Number of similar words to return

        Returns:
            List of (word, similarity_score) tuples

        Example:
            >>> similar = loader.similar_words('market', topn=5)
        """
        if word not in self.vectors:
            return []

        word_vec = self.vectors[word]

        # Compute cosine similarities
        similarities = []
        for other_word, other_vec in self.vectors.items():
            if other_word == word:
                continue

            # Cosine similarity
            sim = np.dot(word_vec, other_vec) / (np.linalg.norm(word_vec) * np.linalg.norm(other_vec))
            similarities.append((other_word, float(sim)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def document_vector(self, doc: List[str]) -> np.ndarray:
        """
        Get document vector by averaging word vectors.

        Args:
            doc: Tokenized document (list of words)

        Returns:
            Document vector (average of word vectors)

        Example:
            >>> doc = ['market', 'volatility', 'increased']
            >>> doc_vec = loader.document_vector(doc)
        """
        vectors = []
        for word in doc:
            vec = self.word_vector(word)
            if vec is not None:
                vectors.append(vec)

        if not vectors:
            # Return zero vector if no words found
            return np.zeros(self.vector_size)

        return np.mean(vectors, axis=0)

    def __len__(self) -> int:
        """Return number of words in vocabulary."""
        return len(self.vectors)


class Doc2VecTrainer:
    """
    Train and use Doc2Vec embeddings with Gensim.

    Doc2Vec (Paragraph Vector) extends Word2Vec to learn document-level embeddings
    in addition to word embeddings.

    Attributes:
        model: Trained Gensim Doc2Vec model

    Example:
        >>> documents = [['market', 'volatility'], ['stock', 'price']]
        >>> trainer = Doc2VecTrainer()
        >>> model = trainer.train(documents, vector_size=100)
        >>> doc_vec = trainer.infer_vector(['new', 'document'])
    """

    def __init__(self):
        """Initialize Doc2VecTrainer."""
        if not GENSIM_AVAILABLE:
            raise ImportError("Gensim is required for Doc2VecTrainer. Install with: pip install gensim")
        self.model = None

    def train(
        self,
        documents: List[List[str]],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        workers: int = 4,
        epochs: int = 10,
        dm: int = 1,
        **kwargs
    ) -> Doc2Vec:
        """
        Train Doc2Vec model on documents.

        Args:
            documents: List of tokenized documents
            vector_size: Dimensionality of document/word vectors
            window: Maximum distance between current and predicted word
            min_count: Ignores words with frequency lower than this
            workers: Number of worker threads
            epochs: Number of training epochs
            dm: Training algorithm (1=PV-DM, 0=PV-DBOW)
            **kwargs: Additional arguments passed to Doc2Vec

        Returns:
            Trained Doc2Vec model

        Example:
            >>> documents = [['the', 'stock', 'market'], ['trading', 'volume']]
            >>> trainer = Doc2VecTrainer()
            >>> model = trainer.train(documents, vector_size=50)
        """
        # Tag documents with unique identifiers
        tagged_docs = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(documents)]

        self.model = Doc2Vec(
            documents=tagged_docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            dm=dm,
            **kwargs
        )
        return self.model

    def load(self, path: Union[str, Path]) -> Doc2Vec:
        """
        Load a trained Doc2Vec model from disk.

        Args:
            path: Path to saved model file

        Returns:
            Loaded Doc2Vec model

        Example:
            >>> trainer = Doc2VecTrainer()
            >>> model = trainer.load('doc2vec.model')
        """
        self.model = Doc2Vec.load(str(path))
        return self.model

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model

        Example:
            >>> trainer.save('doc2vec.model')
        """
        if self.model is None:
            raise ValueError("No model to save. Train or load a model first.")
        self.model.save(str(path))

    def infer_vector(self, doc: List[str], steps: int = 20, alpha: float = 0.025) -> np.ndarray:
        """
        Infer vector for a new document.

        Args:
            doc: Tokenized document
            steps: Number of inference steps
            alpha: Learning rate

        Returns:
            Document vector

        Example:
            >>> doc = ['market', 'volatility', 'increased']
            >>> doc_vec = trainer.infer_vector(doc)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        return self.model.infer_vector(doc, epochs=steps, alpha=alpha)

    def similar_documents(self, doc: List[str], topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar documents to a given document.

        Args:
            doc: Tokenized document
            topn: Number of similar documents to return

        Returns:
            List of (doc_id, similarity_score) tuples

        Example:
            >>> similar = trainer.similar_documents(['market', 'volatility'], topn=5)
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        # Infer vector for input document
        doc_vec = self.infer_vector(doc)

        # Find most similar document vectors
        return self.model.dv.most_similar([doc_vec], topn=topn)

    def document_vector(self, doc_id: str) -> np.ndarray:
        """
        Get the learned vector for a document by its ID.

        Args:
            doc_id: Document identifier (tag used during training)

        Returns:
            Document vector

        Example:
            >>> vec = trainer.document_vector('0')
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        return self.model.dv[doc_id]

    def word_vector(self, word: str) -> np.ndarray:
        """
        Get the vector representation of a word.

        Args:
            word: Word to get vector for

        Returns:
            Word vector

        Raises:
            KeyError: If word not in vocabulary

        Example:
            >>> vec = trainer.word_vector('market')
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        return self.model.wv[word]
