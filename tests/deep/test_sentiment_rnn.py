"""
Tests for sentiment analysis RNN models.
"""

import pytest
import numpy as np
import torch

from puffin.deep.sentiment_rnn import (
    SentimentLSTM,
    SentimentClassifier
)


class TestSentimentLSTM:
    """Tests for SentimentLSTM module."""

    def test_forward_pass(self):
        """Test forward pass through sentiment LSTM."""
        vocab_size = 1000
        model = SentimentLSTM(
            vocab_size=vocab_size,
            embed_dim=50,
            hidden_dim=64,
            output_dim=3
        )

        # Create dummy input (batch_size=16, seq_len=20)
        x = torch.randint(0, vocab_size, (16, 20))
        output = model(x)

        assert output.shape == (16, 3)

    def test_with_pretrained_embeddings(self):
        """Test with pretrained embeddings."""
        vocab_size = 1000
        embed_dim = 100

        # Create dummy pretrained embeddings
        pretrained = np.random.randn(vocab_size, embed_dim).astype(np.float32)

        model = SentimentLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=64,
            output_dim=3,
            pretrained_embeddings=pretrained
        )

        x = torch.randint(0, vocab_size, (8, 15))
        output = model(x)

        assert output.shape == (8, 3)

        # Check embeddings were loaded
        np.testing.assert_array_almost_equal(
            model.embedding.weight.data.numpy(),
            pretrained,
            decimal=5
        )

    def test_different_output_dims(self):
        """Test with different numbers of output classes."""
        vocab_size = 500

        # Binary classification
        model_binary = SentimentLSTM(vocab_size=vocab_size, output_dim=2)
        x = torch.randint(0, vocab_size, (4, 10))
        out_binary = model_binary(x)
        assert out_binary.shape == (4, 2)

        # 5-class classification
        model_multi = SentimentLSTM(vocab_size=vocab_size, output_dim=5)
        out_multi = model_multi(x)
        assert out_multi.shape == (4, 5)


class TestSentimentClassifier:
    """Tests for SentimentClassifier class."""

    def test_tokenize(self):
        """Test text tokenization."""
        classifier = SentimentClassifier()

        text = "This is a TEST! With punctuation, and numbers123."
        tokens = classifier._tokenize(text)

        assert 'this' in tokens
        assert 'test' in tokens
        assert 'punctuation' in tokens
        # Should handle mixed case
        assert all(token.islower() or token.isdigit() for token in tokens)

    def test_build_vocab(self):
        """Test vocabulary building."""
        texts = [
            "the stock market is up today",
            "the market crashed yesterday",
            "stock prices are rising",
            "prices fell sharply"
        ]

        classifier = SentimentClassifier()
        word2idx = classifier.build_vocab(texts, max_vocab=100, min_freq=1)

        # Should have PAD and UNK
        assert '<PAD>' in word2idx
        assert '<UNK>' in word2idx
        assert word2idx['<PAD>'] == 0
        assert word2idx['<UNK>'] == 1

        # Common words should be in vocab
        assert 'the' in word2idx
        assert 'stock' in word2idx
        assert 'market' in word2idx

    def test_build_vocab_min_freq(self):
        """Test vocabulary filtering by minimum frequency."""
        texts = [
            "common common common",
            "rare",
            "common word"
        ]

        classifier = SentimentClassifier()
        word2idx = classifier.build_vocab(texts, min_freq=2)

        # 'common' appears 3 times, should be included
        assert 'common' in word2idx

        # 'rare' appears once, should be excluded
        assert 'rare' not in word2idx

    def test_text_to_sequence(self):
        """Test conversion of text to sequence."""
        texts = ["the stock market", "prices rising"]

        classifier = SentimentClassifier()
        classifier.build_vocab(texts)

        seq = classifier._text_to_sequence("the stock", max_len=5)

        assert len(seq) == 5
        assert seq[0] == classifier.word2idx['the']
        assert seq[1] == classifier.word2idx['stock']
        # Remaining should be padding
        assert seq[2] == classifier.word2idx['<PAD>']

    def test_text_to_sequence_unknown_words(self):
        """Test handling of unknown words."""
        texts = ["known words here"]

        classifier = SentimentClassifier()
        classifier.build_vocab(texts)

        seq = classifier._text_to_sequence("unknown word", max_len=3)

        # Unknown words should map to UNK
        assert seq[0] == classifier.word2idx['<UNK>']
        assert seq[1] == classifier.word2idx['<UNK>']

    def test_text_to_sequence_truncation(self):
        """Test sequence truncation."""
        texts = ["word1 word2 word3"]

        classifier = SentimentClassifier()
        classifier.build_vocab(texts)

        long_text = "word1 word2 word3 word1 word2"
        seq = classifier._text_to_sequence(long_text, max_len=3)

        assert len(seq) == 3
        # Should keep first 3 tokens
        assert seq[0] == classifier.word2idx['word1']
        assert seq[1] == classifier.word2idx['word2']
        assert seq[2] == classifier.word2idx['word3']

    def test_fit_basic(self):
        """Test basic training."""
        # Create simple training data
        texts = [
            "good stock buy now",
            "excellent profit rising",
            "great investment opportunity",
            "bad stock falling",
            "terrible losses declining",
            "poor performance dropping",
            "neutral market stable",
            "sideways trading flat"
        ]

        # 0=negative, 1=neutral, 2=positive
        labels = [2, 2, 2, 0, 0, 0, 1, 1]

        classifier = SentimentClassifier()
        history = classifier.fit(
            texts,
            labels,
            epochs=5,
            batch_size=4,
            max_len=10,
            embed_dim=20,
            hidden_dim=32
        )

        assert 'train_loss' in history
        assert 'train_acc' in history
        assert 'val_loss' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) == 5

    def test_predict(self):
        """Test prediction."""
        texts = [
            "stock rising good",
            "stock falling bad",
            "market stable neutral"
        ] * 10  # Repeat to have enough data

        labels = [2, 0, 1] * 10  # positive, negative, neutral

        classifier = SentimentClassifier()
        classifier.fit(
            texts,
            labels,
            epochs=10,
            batch_size=8,
            max_len=10
        )

        # Predict on new texts
        test_texts = ["stock rising good", "stock falling bad"]
        predictions = classifier.predict(test_texts)

        assert predictions.shape == (2,)
        assert all(0 <= p <= 2 for p in predictions)

    def test_predict_proba(self):
        """Test probability prediction."""
        texts = ["good stock", "bad stock", "neutral"] * 10
        labels = [2, 0, 1] * 10

        classifier = SentimentClassifier()
        classifier.fit(
            texts,
            labels,
            epochs=5,
            batch_size=8,
            max_len=5
        )

        test_texts = ["good stock", "bad stock"]
        probas = classifier.predict_proba(test_texts)

        # Should have shape (n_samples, n_classes)
        assert probas.shape == (2, 3)

        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(
            probas.sum(axis=1),
            np.ones(2),
            decimal=5
        )

        # Probabilities should be between 0 and 1
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)

    def test_predict_before_fit_raises(self):
        """Test that predict raises error before fit."""
        classifier = SentimentClassifier()

        with pytest.raises(ValueError, match="Model must be trained"):
            classifier.predict(["some text"])

    def test_predict_proba_before_fit_raises(self):
        """Test that predict_proba raises error before fit."""
        classifier = SentimentClassifier()

        with pytest.raises(ValueError, match="Model must be trained"):
            classifier.predict_proba(["some text"])

    def test_convergence_on_simple_data(self):
        """Test that model can learn simple sentiment patterns."""
        # Create data with clear patterns
        positive_texts = [
            "excellent great wonderful amazing",
            "fantastic superb outstanding brilliant",
            "good positive happy delighted"
        ] * 15

        negative_texts = [
            "terrible horrible awful dreadful",
            "bad negative sad disappointed",
            "poor weak failing declining"
        ] * 15

        texts = positive_texts + negative_texts
        labels = [1] * len(positive_texts) + [0] * len(negative_texts)

        # Shuffle
        indices = np.random.permutation(len(texts))
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

        classifier = SentimentClassifier()
        history = classifier.fit(
            texts,
            labels,
            epochs=20,
            batch_size=16,
            max_len=10
        )

        # Training accuracy should improve
        assert history['train_acc'][-1] > history['train_acc'][0]

        # Final accuracy should be reasonable
        assert history['train_acc'][-1] > 0.6


class TestDeviceHandling:
    """Tests for device handling."""

    def test_sentiment_classifier_uses_correct_device(self):
        """Test that SentimentClassifier uses available device."""
        classifier = SentimentClassifier()
        expected_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert classifier.device.type == expected_device


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
