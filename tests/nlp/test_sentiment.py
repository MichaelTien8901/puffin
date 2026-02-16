"""Tests for sentiment analysis."""

import pytest

from puffin.nlp.sentiment import LexiconSentiment, RuleSentiment


@pytest.fixture
def bullish_texts():
    return [
        "Strong earnings beat expectations, profit margins expanded significantly.",
        "Company announced excellent results and raised guidance.",
        "Stock rallied on positive outlook and robust growth.",
        "Outstanding performance with record revenue and profitability.",
    ]


@pytest.fixture
def bearish_texts():
    return [
        "Disappointing results missed estimates, guidance lowered substantially.",
        "Company reported significant losses and declining margins.",
        "Stock plunged on weak outlook and deteriorating fundamentals.",
        "Poor performance with falling revenue and mounting problems.",
    ]


@pytest.fixture
def neutral_texts():
    return [
        "Company held annual shareholder meeting.",
        "Stock traded in line with market averages.",
        "Quarterly filing submitted on schedule.",
    ]


def test_rule_sentiment_initialization():
    sentiment = RuleSentiment()

    assert sentiment is not None
    assert len(sentiment.positive_words) > 0
    assert len(sentiment.negative_words) > 0
    assert not sentiment.case_sensitive


def test_rule_sentiment_bullish(bullish_texts):
    sentiment = RuleSentiment()

    for text in bullish_texts:
        score = sentiment.score(text)
        assert score > 0, f"Expected positive score for: {text}"


def test_rule_sentiment_bearish(bearish_texts):
    sentiment = RuleSentiment()

    for text in bearish_texts:
        score = sentiment.score(text)
        assert score < 0, f"Expected negative score for: {text}"


def test_rule_sentiment_neutral(neutral_texts):
    sentiment = RuleSentiment()

    for text in neutral_texts:
        score = sentiment.score(text)
        # Neutral texts should have scores close to 0
        assert abs(score) < 0.1, f"Expected near-zero score for: {text}"


def test_rule_sentiment_score_range():
    sentiment = RuleSentiment()

    texts = [
        "Excellent outstanding brilliant fantastic achievement success.",
        "Terrible horrible awful disaster failure crisis collapse.",
        "The company exists and has operations.",
    ]

    for text in texts:
        score = sentiment.score(text)
        assert -1.0 <= score <= 1.0, f"Score out of range: {score}"


def test_rule_sentiment_batch_score(bullish_texts, bearish_texts):
    sentiment = RuleSentiment()

    all_texts = bullish_texts + bearish_texts
    scores = sentiment.batch_score(all_texts)

    assert len(scores) == len(all_texts)
    assert all(isinstance(s, float) for s in scores)

    # First half should be mostly positive
    bullish_scores = scores[:len(bullish_texts)]
    assert sum(s > 0 for s in bullish_scores) >= len(bullish_texts) * 0.75

    # Second half should be mostly negative
    bearish_scores = scores[len(bullish_texts):]
    assert sum(s < 0 for s in bearish_scores) >= len(bearish_texts) * 0.75


def test_rule_sentiment_analyze():
    sentiment = RuleSentiment()

    text = "Strong growth and excellent performance despite some challenges."
    analysis = sentiment.analyze(text)

    assert "score" in analysis
    assert "label" in analysis
    assert "positive_count" in analysis
    assert "negative_count" in analysis
    assert "positive_words" in analysis
    assert "negative_words" in analysis
    assert "total_words" in analysis

    assert analysis["label"] in ["bullish", "bearish", "neutral"]
    assert isinstance(analysis["positive_count"], int)
    assert isinstance(analysis["negative_count"], int)
    assert isinstance(analysis["positive_words"], list)
    assert isinstance(analysis["negative_words"], list)


def test_rule_sentiment_custom_words():
    positive_words = {"moon", "rocket", "gains"}
    negative_words = {"crash", "dump", "rekt"}

    sentiment = RuleSentiment(
        positive_words=positive_words,
        negative_words=negative_words
    )

    text1 = "Stock to the moon! Rocket gains!"
    text2 = "Market crash and dump. We're rekt."

    score1 = sentiment.score(text1)
    score2 = sentiment.score(text2)

    assert score1 > 0
    assert score2 < 0


def test_rule_sentiment_case_sensitivity():
    # Case insensitive (default)
    sentiment1 = RuleSentiment(case_sensitive=False)
    score1 = sentiment1.score("EXCELLENT PERFORMANCE")
    assert score1 > 0

    # Case sensitive
    sentiment2 = RuleSentiment(
        positive_words={"excellent", "performance"},
        case_sensitive=True
    )
    score2 = sentiment2.score("EXCELLENT PERFORMANCE")
    assert score2 == 0  # No match due to case


def test_lexicon_sentiment_initialization():
    sentiment = LexiconSentiment()

    assert sentiment is not None
    assert len(sentiment.positive_words) > 0
    assert len(sentiment.negative_words) > 0


def test_lexicon_sentiment_with_weights():
    positive_words = {"great": 1.0, "excellent": 2.0, "outstanding": 3.0}
    negative_words = {"bad": 1.0, "terrible": 2.0, "awful": 3.0}

    sentiment = LexiconSentiment(
        positive_words=positive_words,
        negative_words=negative_words
    )

    # "outstanding" has higher weight than "great"
    score1 = sentiment.score("This is great.")
    score2 = sentiment.score("This is outstanding.")
    assert score2 > score1


def test_lexicon_sentiment_bullish(bullish_texts):
    sentiment = LexiconSentiment()

    for text in bullish_texts:
        score = sentiment.score(text)
        assert score > 0, f"Expected positive score for: {text}"


def test_lexicon_sentiment_bearish(bearish_texts):
    sentiment = LexiconSentiment()

    for text in bearish_texts:
        score = sentiment.score(text)
        assert score < 0, f"Expected negative score for: {text}"


def test_lexicon_sentiment_analyze():
    sentiment = LexiconSentiment()

    text = "Strong excellent growth but facing some difficulties."
    analysis = sentiment.analyze(text)

    assert "score" in analysis
    assert "label" in analysis
    assert "positive_score" in analysis
    assert "negative_score" in analysis
    assert "positive_words" in analysis
    assert "negative_words" in analysis

    assert isinstance(analysis["positive_words"], dict)
    assert isinstance(analysis["negative_words"], dict)


def test_lexicon_sentiment_add_words():
    sentiment = LexiconSentiment(
        positive_words=set(),
        negative_words=set()
    )

    # Initially should be neutral
    score1 = sentiment.score("This stock is mooning.")
    assert score1 == 0

    # Add custom word
    sentiment.add_positive_word("mooning", weight=2.0)
    score2 = sentiment.score("This stock is mooning.")
    assert score2 > 0


def test_lexicon_sentiment_batch_score():
    sentiment = LexiconSentiment()

    texts = [
        "Excellent performance and strong growth.",
        "Terrible losses and declining revenue.",
        "Normal trading activity.",
    ]

    scores = sentiment.batch_score(texts)

    assert len(scores) == len(texts)
    assert scores[0] > 0  # Positive
    assert scores[1] < 0  # Negative
    assert abs(scores[2]) < abs(scores[0])  # More neutral


def test_empty_text():
    sentiment_rule = RuleSentiment()
    sentiment_lexicon = LexiconSentiment()

    score1 = sentiment_rule.score("")
    score2 = sentiment_lexicon.score("")

    assert score1 == 0
    assert score2 == 0


def test_text_with_no_sentiment_words():
    sentiment = RuleSentiment()

    text = "The meeting is scheduled for Tuesday."
    score = sentiment.score(text)

    assert score == 0


def test_mixed_sentiment():
    sentiment = RuleSentiment()

    # Balanced mix of positive and negative
    text = "Strong growth but significant challenges and risks remain."
    score = sentiment.score(text)

    # Score should be relatively balanced
    assert abs(score) < 0.2


def test_repr():
    sentiment1 = RuleSentiment()
    repr1 = repr(sentiment1)

    assert "RuleSentiment" in repr1
    assert "positive=" in repr1
    assert "negative=" in repr1

    sentiment2 = LexiconSentiment()
    repr2 = repr(sentiment2)

    assert "LexiconSentiment" in repr2
    assert "positive=" in repr2
    assert "negative=" in repr2


def test_financial_specific_sentiment():
    sentiment = RuleSentiment()

    # Test financial-specific terms
    financial_texts = [
        "EBITDA margins expanded, ROE improved significantly.",
        "Earnings beat, revenue growth accelerated.",
        "Debt levels increased, cash flow deteriorated.",
        "Bankruptcy risk, credit downgrade.",
    ]

    for text in financial_texts[:2]:
        score = sentiment.score(text)
        # May not be strongly positive with default lexicon, but shouldn't be negative
        assert score >= 0

    for text in financial_texts[2:]:
        score = sentiment.score(text)
        assert score <= 0
