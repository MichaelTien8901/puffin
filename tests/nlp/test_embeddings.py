"""
Tests for word embeddings module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

try:
    from puffin.nlp.embeddings import Word2VecTrainer, GloVeLoader, Doc2VecTrainer
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


# Sample documents for testing
SAMPLE_DOCS = [
    ['the', 'market', 'is', 'very', 'volatile', 'today'],
    ['stock', 'prices', 'are', 'rising', 'rapidly'],
    ['bond', 'yields', 'increased', 'significantly'],
    ['volatility', 'in', 'the', 'market', 'continues'],
    ['stock', 'market', 'closed', 'higher', 'today'],
    ['oil', 'prices', 'dropped', 'sharply'],
    ['interest', 'rates', 'remain', 'low'],
    ['market', 'sentiment', 'is', 'positive'],
    ['stock', 'volatility', 'increased'],
    ['bond', 'market', 'is', 'stable']
]


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
class TestWord2VecTrainer:
    """Test Word2VecTrainer class."""

    def test_train(self):
        """Test training Word2Vec model."""
        trainer = Word2VecTrainer()
        model = trainer.train(
            SAMPLE_DOCS,
            vector_size=50,
            window=3,
            min_count=1,
            sg=1,
            epochs=10
        )

        assert model is not None
        assert trainer.model is not None
        assert model.vector_size == 50

    def test_word_vector(self):
        """Test getting word vectors."""
        trainer = Word2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        vector = trainer.word_vector('market')
        assert isinstance(vector, np.ndarray)
        assert vector.shape == (50,)

    def test_word_vector_missing(self):
        """Test getting vector for word not in vocabulary."""
        trainer = Word2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=5, epochs=10)

        with pytest.raises(KeyError):
            trainer.word_vector('nonexistent_word_xyz')

    def test_similar_words(self):
        """Test finding similar words."""
        trainer = Word2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        similar = trainer.similar_words('market', topn=3)
        assert isinstance(similar, list)
        assert len(similar) <= 3
        if similar:
            assert isinstance(similar[0], tuple)
            assert len(similar[0]) == 2
            assert isinstance(similar[0][0], str)
            assert isinstance(similar[0][1], float)

    def test_similar_words_missing(self):
        """Test finding similar words for word not in vocabulary."""
        trainer = Word2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        similar = trainer.similar_words('nonexistent_word_xyz', topn=3)
        assert similar == []

    def test_document_vector(self):
        """Test getting document vectors."""
        trainer = Word2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        doc_vec = trainer.document_vector(['market', 'volatility'])
        assert isinstance(doc_vec, np.ndarray)
        assert doc_vec.shape == (50,)

    def test_document_vector_empty(self):
        """Test document vector for document with no known words."""
        trainer = Word2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=5, epochs=10)

        doc_vec = trainer.document_vector(['xyz', 'abc'])
        assert isinstance(doc_vec, np.ndarray)
        assert doc_vec.shape == (50,)
        assert np.all(doc_vec == 0)

    def test_save_load(self):
        """Test saving and loading model."""
        trainer = Word2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model.w2v'
            trainer.save(model_path)

            # Load in new trainer
            new_trainer = Word2VecTrainer()
            new_trainer.load(model_path)

            # Check vectors match
            vec1 = trainer.word_vector('market')
            vec2 = new_trainer.word_vector('market')
            np.testing.assert_array_almost_equal(vec1, vec2)

    def test_analogy(self):
        """Test word analogies."""
        trainer = Word2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        # Test analogy (may not work well on small dataset)
        result = trainer.analogy(['market', 'stock'], ['market'])
        assert isinstance(result, list)


class TestGloVeLoader:
    """Test GloVeLoader class."""

    def test_init(self):
        """Test GloVeLoader initialization."""
        loader = GloVeLoader()
        assert len(loader.vectors) == 0
        assert loader.vector_size == 0

    def test_load_from_file(self):
        """Test loading GloVe vectors from file."""
        # Create temporary GloVe file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('market 0.1 0.2 0.3 0.4 0.5\n')
            f.write('stock 0.2 0.3 0.4 0.5 0.6\n')
            f.write('price 0.3 0.4 0.5 0.6 0.7\n')
            temp_path = f.name

        try:
            loader = GloVeLoader()
            loader.load(temp_path)

            assert len(loader) == 3
            assert loader.vector_size == 5
            assert 'market' in loader.vectors
            assert 'stock' in loader.vectors
            assert 'price' in loader.vectors
        finally:
            Path(temp_path).unlink()

    def test_word_vector(self):
        """Test getting word vectors."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('market 0.1 0.2 0.3\n')
            f.write('stock 0.4 0.5 0.6\n')
            temp_path = f.name

        try:
            loader = GloVeLoader()
            loader.load(temp_path)

            vec = loader.word_vector('market')
            assert vec is not None
            assert isinstance(vec, np.ndarray)
            assert vec.shape == (3,)
            np.testing.assert_array_almost_equal(vec, [0.1, 0.2, 0.3])
        finally:
            Path(temp_path).unlink()

    def test_word_vector_missing(self):
        """Test getting vector for word not in vocabulary."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('market 0.1 0.2 0.3\n')
            temp_path = f.name

        try:
            loader = GloVeLoader()
            loader.load(temp_path)

            vec = loader.word_vector('nonexistent')
            assert vec is None
        finally:
            Path(temp_path).unlink()

    def test_document_vector(self):
        """Test getting document vectors."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('market 1.0 0.0 0.0\n')
            f.write('stock 0.0 1.0 0.0\n')
            temp_path = f.name

        try:
            loader = GloVeLoader()
            loader.load(temp_path)

            doc_vec = loader.document_vector(['market', 'stock'])
            assert isinstance(doc_vec, np.ndarray)
            assert doc_vec.shape == (3,)
            np.testing.assert_array_almost_equal(doc_vec, [0.5, 0.5, 0.0])
        finally:
            Path(temp_path).unlink()

    def test_document_vector_empty(self):
        """Test document vector for document with no known words."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('market 1.0 0.0 0.0\n')
            temp_path = f.name

        try:
            loader = GloVeLoader()
            loader.load(temp_path)

            doc_vec = loader.document_vector(['xyz', 'abc'])
            assert isinstance(doc_vec, np.ndarray)
            assert np.all(doc_vec == 0)
        finally:
            Path(temp_path).unlink()


@pytest.mark.skipif(not GENSIM_AVAILABLE, reason="Gensim not available")
class TestDoc2VecTrainer:
    """Test Doc2VecTrainer class."""

    def test_train(self):
        """Test training Doc2Vec model."""
        trainer = Doc2VecTrainer()
        model = trainer.train(
            SAMPLE_DOCS,
            vector_size=50,
            window=3,
            min_count=1,
            epochs=10
        )

        assert model is not None
        assert trainer.model is not None
        assert model.vector_size == 50

    def test_infer_vector(self):
        """Test inferring vector for new document."""
        trainer = Doc2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        doc_vec = trainer.infer_vector(['market', 'volatility', 'increased'])
        assert isinstance(doc_vec, np.ndarray)
        assert doc_vec.shape == (50,)

    def test_document_vector(self):
        """Test getting trained document vector."""
        trainer = Doc2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        # Get vector for first training document
        doc_vec = trainer.document_vector('0')
        assert isinstance(doc_vec, np.ndarray)
        assert doc_vec.shape == (50,)

    def test_similar_documents(self):
        """Test finding similar documents."""
        trainer = Doc2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        similar = trainer.similar_documents(['market', 'volatility'], topn=3)
        assert isinstance(similar, list)
        assert len(similar) <= 3
        if similar:
            assert isinstance(similar[0], tuple)
            assert len(similar[0]) == 2

    def test_word_vector(self):
        """Test getting word vectors from Doc2Vec."""
        trainer = Doc2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        word_vec = trainer.word_vector('market')
        assert isinstance(word_vec, np.ndarray)
        assert word_vec.shape == (50,)

    def test_save_load(self):
        """Test saving and loading model."""
        trainer = Doc2VecTrainer()
        trainer.train(SAMPLE_DOCS, vector_size=50, min_count=1, epochs=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model.d2v'
            trainer.save(model_path)

            # Load in new trainer
            new_trainer = Doc2VecTrainer()
            new_trainer.load(model_path)

            # Check vectors match
            vec1 = trainer.document_vector('0')
            vec2 = new_trainer.document_vector('0')
            np.testing.assert_array_almost_equal(vec1, vec2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
