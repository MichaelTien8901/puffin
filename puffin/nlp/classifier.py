"""News classification using Naive Bayes and TF-IDF."""

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class NewsClassifier:
    """
    News classifier using TF-IDF + Multinomial Naive Bayes.

    Designed for financial news classification into categories like:
    bullish, bearish, neutral
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        alpha: float = 1.0,
        **kwargs: Any
    ):
        """
        Initialize news classifier.

        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for feature extraction
            alpha: Smoothing parameter for Naive Bayes
            **kwargs: Additional arguments for TfidfVectorizer
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.kwargs = kwargs

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                **kwargs
            )),
            ('clf', MultinomialNB(alpha=alpha))
        ])

        self.classes_: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, texts: list[str], labels: list[str]) -> "NewsClassifier":
        """
        Fit classifier on training data.

        Args:
            texts: List of text documents
            labels: List of corresponding labels (e.g., "bullish", "bearish", "neutral")

        Returns:
            Self for method chaining
        """
        self.pipeline.fit(texts, labels)
        self.classes_ = self.pipeline.named_steps['clf'].classes_
        self.is_fitted_ = True
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """
        Predict labels for texts.

        Args:
            texts: List of text documents

        Returns:
            Array of predicted labels
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before prediction.")

        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Predict class probabilities for texts.

        Args:
            texts: List of text documents

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before prediction.")

        return self.pipeline.predict_proba(texts)

    def evaluate(
        self,
        texts: list[str],
        labels: list[str],
        average: str = "weighted"
    ) -> dict[str, Any]:
        """
        Evaluate classifier on test data.

        Args:
            texts: List of text documents
            labels: List of true labels
            average: Averaging method for multi-class metrics

        Returns:
            Dictionary with evaluation metrics:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1 score
            - confusion_matrix: Confusion matrix
            - classification_report: Detailed classification report
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before evaluation.")

        predictions = self.predict(texts)

        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, average=average, zero_division=0),
            "recall": recall_score(labels, predictions, average=average, zero_division=0),
            "f1": f1_score(labels, predictions, average=average, zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
            "classification_report": classification_report(labels, predictions),
        }

    def get_feature_importance(self, n: int = 20) -> dict[str, list[tuple[str, float]]]:
        """
        Get top features for each class.

        Args:
            n: Number of top features per class

        Returns:
            Dictionary mapping class labels to list of (feature, importance) tuples
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted first.")

        vectorizer = self.pipeline.named_steps['tfidf']
        classifier = self.pipeline.named_steps['clf']

        feature_names = vectorizer.get_feature_names_out()
        feature_log_prob = classifier.feature_log_prob_

        importance_by_class = {}

        for idx, class_label in enumerate(self.classes_):
            # Get top features for this class
            top_indices = np.argsort(feature_log_prob[idx])[-n:][::-1]
            top_features = [
                (feature_names[i], feature_log_prob[idx][i])
                for i in top_indices
            ]
            importance_by_class[class_label] = top_features

        return importance_by_class

    def get_prediction_explanation(
        self,
        text: str,
        n_features: int = 10
    ) -> dict[str, Any]:
        """
        Get explanation for a prediction.

        Args:
            text: Text to explain
            n_features: Number of top features to show

        Returns:
            Dictionary with prediction, probabilities, and top features
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted first.")

        prediction = self.predict([text])[0]
        probabilities = self.predict_proba([text])[0]

        # Get feature weights for this document
        vectorizer = self.pipeline.named_steps['tfidf']
        classifier = self.pipeline.named_steps['clf']

        features = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()

        # Get non-zero features
        non_zero_indices = features.nonzero()[1]
        feature_weights = []

        for idx in non_zero_indices:
            feature_name = feature_names[idx]
            tfidf_weight = features[0, idx]
            # Get log probability for predicted class
            pred_class_idx = np.where(self.classes_ == prediction)[0][0]
            log_prob = classifier.feature_log_prob_[pred_class_idx][idx]

            feature_weights.append((feature_name, tfidf_weight * log_prob))

        # Sort by absolute weight
        feature_weights.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            "prediction": prediction,
            "probabilities": {
                class_label: float(prob)
                for class_label, prob in zip(self.classes_, probabilities)
            },
            "top_features": feature_weights[:n_features],
        }

    def save(self, path: str) -> None:
        """
        Save classifier to disk.

        Args:
            path: File path to save to
        """
        import pickle

        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted classifier.")

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "NewsClassifier":
        """
        Load classifier from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded NewsClassifier instance
        """
        import pickle

        with open(path, 'rb') as f:
            return pickle.load(f)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        classes = f", classes={list(self.classes_)}" if self.is_fitted_ else ""
        return (
            f"NewsClassifier("
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, "
            f"alpha={self.alpha}, "
            f"{status}{classes})"
        )
