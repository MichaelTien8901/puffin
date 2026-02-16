"""
LSTM-based sentiment analysis for trading applications.

This module provides sentiment classification using LSTM networks with
word embeddings, useful for analyzing news, tweets, and other text data
that may influence trading decisions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
from collections import Counter
import re


class SentimentLSTM(nn.Module):
    """
    LSTM network for sentiment classification.

    Uses word embeddings (optionally pretrained) followed by LSTM layers
    for text classification.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary
    embed_dim : int, default=100
        Dimension of word embeddings
    hidden_dim : int, default=128
        Number of hidden units in LSTM
    output_dim : int, default=3
        Number of output classes (e.g., 3 for negative/neutral/positive)
    pretrained_embeddings : np.ndarray, optional
        Pretrained embedding matrix of shape (vocab_size, embed_dim)
    dropout : float, default=0.3
        Dropout probability
    num_layers : int, default=2
        Number of LSTM layers
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        output_dim: int = 3,
        pretrained_embeddings: Optional[np.ndarray] = None,
        dropout: float = 0.3,
        num_layers: int = 2
    ):
        super(SentimentLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Optionally freeze embeddings
            # self.embedding.weight.requires_grad = False

        # LSTM layer
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len) containing token indices

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, output_dim)
        """
        # Embedding: (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)

        # LSTM: (batch_size, seq_len, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use the last hidden state from the last layer
        # hidden shape: (num_layers, batch_size, hidden_dim)
        last_hidden = hidden[-1]

        # Dropout
        out = self.dropout(last_hidden)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class SentimentClassifier:
    """
    High-level sentiment classifier using LSTM.

    Handles vocabulary building, text preprocessing, training, and prediction.
    """

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self.output_dim = None

    def build_vocab(
        self,
        texts: List[str],
        max_vocab: int = 10000,
        min_freq: int = 2
    ) -> Dict[str, int]:
        """
        Build vocabulary from text corpus.

        Parameters
        ----------
        texts : list of str
            List of text documents
        max_vocab : int, default=10000
            Maximum vocabulary size
        min_freq : int, default=2
            Minimum frequency for a word to be included

        Returns
        -------
        word2idx : dict
            Mapping from words to indices
        """
        # Tokenize and count words
        word_counts = Counter()
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)

        # Filter by frequency and limit vocabulary size
        vocab_items = [
            (word, count) for word, count in word_counts.items()
            if count >= min_freq
        ]
        vocab_items.sort(key=lambda x: x[1], reverse=True)
        vocab_items = vocab_items[:max_vocab-2]  # Reserve space for PAD and UNK

        # Create mappings
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in vocab_items:
            self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab = set(self.word2idx.keys())

        return self.word2idx

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization by lowercasing and splitting on non-alphanumeric.

        Parameters
        ----------
        text : str
            Input text

        Returns
        -------
        tokens : list of str
            List of tokens
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _text_to_sequence(self, text: str, max_len: int) -> np.ndarray:
        """
        Convert text to sequence of token indices.

        Parameters
        ----------
        text : str
            Input text
        max_len : int
            Maximum sequence length

        Returns
        -------
        sequence : np.ndarray
            Array of token indices, padded/truncated to max_len
        """
        tokens = self._tokenize(text)
        indices = [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokens
        ]

        # Pad or truncate
        if len(indices) < max_len:
            indices = indices + [self.word2idx['<PAD>']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return np.array(indices)

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        validation_split: float = 0.2,
        max_len: int = 100,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        pretrained_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """
        Train the sentiment classifier.

        Parameters
        ----------
        texts : list of str
            Training texts
        labels : list of int
            Training labels (integer class indices)
        epochs : int, default=10
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        lr : float, default=0.001
            Learning rate
        validation_split : float, default=0.2
            Fraction of data for validation
        max_len : int, default=100
            Maximum sequence length
        embed_dim : int, default=100
            Embedding dimension
        hidden_dim : int, default=128
            LSTM hidden dimension
        pretrained_embeddings : np.ndarray, optional
            Pretrained embedding matrix

        Returns
        -------
        history : dict
            Training history with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        """
        self.max_len = max_len
        self.output_dim = len(set(labels))

        # Build vocabulary if not already built
        if self.vocab is None:
            self.build_vocab(texts)

        # Convert texts to sequences
        sequences = np.array([
            self._text_to_sequence(text, max_len)
            for text in texts
        ])
        labels_array = np.array(labels)

        # Split into train and validation
        split_idx = int(len(sequences) * (1 - validation_split))
        indices = np.random.permutation(len(sequences))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        X_train = sequences[train_indices]
        y_train = labels_array[train_indices]
        X_val = sequences[val_indices]
        y_val = labels_array[val_indices]

        # Convert to tensors
        X_train = torch.LongTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.LongTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)

        # Initialize model
        vocab_size = len(self.word2idx)
        self.model = SentimentLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim,
            pretrained_embeddings=pretrained_embeddings
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        for epoch in range(epochs):
            self.model.train()

            # Mini-batch training
            train_loss = 0
            train_correct = 0
            train_total = 0

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss /= (len(X_train) / batch_size)
            train_acc = train_correct / train_total

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i+batch_size]
                    batch_y = y_val[i:i+batch_size]

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_loss /= (len(X_val) / batch_size)
            val_acc = val_correct / val_total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        return history

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict sentiment labels for texts.

        Parameters
        ----------
        texts : list of str
            Texts to classify

        Returns
        -------
        predictions : np.ndarray
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        # Convert texts to sequences
        sequences = np.array([
            self._text_to_sequence(text, self.max_len)
            for text in texts
        ])

        X = torch.LongTensor(sequences).to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for texts.

        Parameters
        ----------
        texts : list of str
            Texts to classify

        Returns
        -------
        probabilities : np.ndarray
            Class probabilities of shape (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        self.model.eval()

        # Convert texts to sequences
        sequences = np.array([
            self._text_to_sequence(text, self.max_len)
            for text in texts
        ])

        X = torch.LongTensor(sequences).to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            probabilities = torch.softmax(outputs, dim=1)

        return probabilities.cpu().numpy()
