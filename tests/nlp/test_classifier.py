"""Tests for news classifier."""

import numpy as np
import pytest

from puffin.nlp.classifier import NewsClassifier


@pytest.fixture
def sample_training_data():
    texts = [
        # Bullish examples
        "Stock surged on strong earnings beat and raised guidance.",
        "Company reported record profits and expanding margins.",
        "Shares rallied after positive analyst upgrade.",
        "Revenue growth exceeded expectations, outlook bullish.",
        "Strong performance drove shares to new highs.",

        # Bearish examples
        "Stock plunged on disappointing earnings and lowered guidance.",
        "Company warned of declining profits and rising costs.",
        "Shares tumbled after negative analyst downgrade.",
        "Revenue miss and weak outlook sent stock lower.",
        "Poor performance led to significant losses.",

        # Neutral examples
        "Company announced routine board meeting next month.",
        "Stock traded flat in line with broader market.",
        "Company reaffirmed previous guidance, no change.",
        "Shares ended mixed with no major news.",
        "Trading volume was average for the session.",
    ]

    labels = (
        ["bullish"] * 5 +
        ["bearish"] * 5 +
        ["neutral"] * 5
    )

    return texts, labels


@pytest.fixture
def sample_test_data():
    texts = [
        "Earnings exceeded forecasts, boosting investor confidence.",
        "Results fell short of expectations, disappointing investors.",
        "Company held annual shareholder meeting.",
    ]

    labels = ["bullish", "bearish", "neutral"]

    return texts, labels


def test_classifier_initialization():
    clf = NewsClassifier(max_features=1000, ngram_range=(1, 2), alpha=1.0)

    assert clf.max_features == 1000
    assert clf.ngram_range == (1, 2)
    assert clf.alpha == 1.0
    assert not clf.is_fitted_


def test_classifier_fit(sample_training_data):
    texts, labels = sample_training_data
    clf = NewsClassifier()

    result = clf.fit(texts, labels)

    assert result is clf  # Check for method chaining
    assert clf.is_fitted_
    assert clf.classes_ is not None
    assert len(clf.classes_) == 3  # bullish, bearish, neutral


def test_classifier_predict(sample_training_data, sample_test_data):
    train_texts, train_labels = sample_training_data
    test_texts, _ = sample_test_data

    clf = NewsClassifier()
    clf.fit(train_texts, train_labels)

    predictions = clf.predict(test_texts)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(test_texts)
    assert all(pred in ["bullish", "bearish", "neutral"] for pred in predictions)


def test_classifier_predict_proba(sample_training_data, sample_test_data):
    train_texts, train_labels = sample_training_data
    test_texts, _ = sample_test_data

    clf = NewsClassifier()
    clf.fit(train_texts, train_labels)

    probas = clf.predict_proba(test_texts)

    assert isinstance(probas, np.ndarray)
    assert probas.shape[0] == len(test_texts)
    assert probas.shape[1] == 3  # 3 classes
    assert np.allclose(probas.sum(axis=1), 1.0)  # Probabilities sum to 1
    assert np.all(probas >= 0) and np.all(probas <= 1)  # Valid probabilities


def test_classifier_evaluate(sample_training_data):
    texts, labels = sample_training_data
    clf = NewsClassifier()
    clf.fit(texts, labels)

    # Evaluate on training data (just for testing)
    metrics = clf.evaluate(texts, labels)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics

    # Check metric ranges
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_get_feature_importance(sample_training_data):
    texts, labels = sample_training_data
    clf = NewsClassifier()
    clf.fit(texts, labels)

    importance = clf.get_feature_importance(n=10)

    assert isinstance(importance, dict)
    assert len(importance) == 3  # One entry per class
    assert "bullish" in importance
    assert "bearish" in importance
    assert "neutral" in importance

    # Check structure of each entry
    for class_label, features in importance.items():
        assert isinstance(features, list)
        assert len(features) <= 10
        assert all(isinstance(f[0], str) and isinstance(f[1], (int, float)) for f in features)


def test_get_prediction_explanation(sample_training_data):
    texts, labels = sample_training_data
    clf = NewsClassifier()
    clf.fit(texts, labels)

    text = "Stock jumped on excellent earnings results."
    explanation = clf.get_prediction_explanation(text, n_features=5)

    assert "prediction" in explanation
    assert "probabilities" in explanation
    assert "top_features" in explanation

    assert explanation["prediction"] in ["bullish", "bearish", "neutral"]
    assert isinstance(explanation["probabilities"], dict)
    assert len(explanation["probabilities"]) == 3
    assert isinstance(explanation["top_features"], list)
    assert len(explanation["top_features"]) <= 5


def test_predict_before_fit():
    clf = NewsClassifier()

    with pytest.raises(ValueError, match="must be fitted"):
        clf.predict(["Some text"])


def test_predict_proba_before_fit():
    clf = NewsClassifier()

    with pytest.raises(ValueError, match="must be fitted"):
        clf.predict_proba(["Some text"])


def test_evaluate_before_fit():
    clf = NewsClassifier()

    with pytest.raises(ValueError, match="must be fitted"):
        clf.evaluate(["Some text"], ["label"])


def test_repr(sample_training_data):
    clf = NewsClassifier(max_features=1000, alpha=0.5)

    # Before fitting
    repr_before = repr(clf)
    assert "not fitted" in repr_before
    assert "max_features=1000" in repr_before
    assert "alpha=0.5" in repr_before

    # After fitting
    texts, labels = sample_training_data
    clf.fit(texts, labels)

    repr_after = repr(clf)
    assert "fitted" in repr_after
    assert "classes=" in repr_after


def test_classifier_with_different_alpha(sample_training_data):
    texts, labels = sample_training_data

    # Test with different smoothing parameters
    for alpha in [0.1, 1.0, 10.0]:
        clf = NewsClassifier(alpha=alpha)
        clf.fit(texts, labels)
        predictions = clf.predict(texts)
        assert len(predictions) == len(texts)


def test_classifier_with_different_ngrams(sample_training_data):
    texts, labels = sample_training_data

    # Test with unigrams only
    clf1 = NewsClassifier(ngram_range=(1, 1))
    clf1.fit(texts, labels)

    # Test with unigrams and bigrams
    clf2 = NewsClassifier(ngram_range=(1, 2))
    clf2.fit(texts, labels)

    # Test with trigrams
    clf3 = NewsClassifier(ngram_range=(1, 3))
    clf3.fit(texts, labels)

    # All should work
    for clf in [clf1, clf2, clf3]:
        predictions = clf.predict(texts[:3])
        assert len(predictions) == 3


def test_save_load_classifier(sample_training_data, tmp_path):
    texts, labels = sample_training_data
    clf = NewsClassifier()
    clf.fit(texts, labels)

    # Save classifier
    save_path = tmp_path / "classifier.pkl"
    clf.save(str(save_path))

    # Load classifier
    loaded_clf = NewsClassifier.load(str(save_path))

    # Test loaded classifier
    predictions_original = clf.predict(texts[:3])
    predictions_loaded = loaded_clf.predict(texts[:3])

    assert np.array_equal(predictions_original, predictions_loaded)
    assert loaded_clf.is_fitted_


def test_save_unfitted_classifier():
    clf = NewsClassifier()

    with pytest.raises(ValueError, match="Cannot save unfitted"):
        clf.save("/tmp/classifier.pkl")
