"""Tests for earnings call transcript parser."""

import pytest

from puffin.data.transcript_parser import TranscriptParser


@pytest.fixture
def parser():
    return TranscriptParser()


@pytest.fixture
def sample_transcript():
    return """
    John Smith - CEO: Good morning everyone. I'm pleased to report strong revenue growth
    of 15% year-over-year. Our revenue was $1.2 billion this quarter, exceeding guidance.
    Gross margin improved to 45%, and earnings per share came in at $2.50, beating
    expectations. We remain optimistic about future growth opportunities.

    Jane Doe - CFO: Thank you John. Operating margin was 25%, and we generated excellent
    cash flow. However, we do face some headwind from currency fluctuations.

    Operator: We'll now begin the question-and-answer session. Our first question comes
    from Mike Johnson.

    Mike Johnson - Analyst: Thanks for taking my question. Can you provide more color on
    the guidance for next quarter?

    John Smith - CEO: We're confident in our ability to deliver. We expect revenue growth
    of 10-12% next quarter, despite some challenging market conditions.
    """


@pytest.fixture
def simple_transcript():
    return "Revenue was $100 million. Earnings per share was $1.50."


def test_parser_initialization(parser):
    assert parser is not None
    assert "prepared_remarks" in parser.section_markers
    assert "qa_session" in parser.section_markers


def test_parse_empty_text(parser):
    result = parser.parse("")

    assert result["prepared_remarks"] == ""
    assert result["qa_session"] == ""
    assert result["participants"] == []


def test_parse_with_sections(parser, sample_transcript):
    result = parser.parse(sample_transcript)

    assert "prepared_remarks" in result
    assert "qa_session" in result
    assert "participants" in result

    # Should split at Q&A marker
    assert len(result["prepared_remarks"]) > 0
    assert len(result["qa_session"]) > 0
    assert "question-and-answer" in result["qa_session"].lower()


def test_parse_no_qa_section(parser, simple_transcript):
    result = parser.parse(simple_transcript)

    # Should treat all as prepared remarks if no Q&A marker
    assert len(result["prepared_remarks"]) > 0
    assert result["qa_session"] == ""


def test_extract_participants(parser, sample_transcript):
    result = parser.parse(sample_transcript)

    participants = result["participants"]
    assert "John Smith" in participants
    assert "Jane Doe" in participants
    assert "Mike Johnson" in participants
    # Operator should be filtered out
    assert "Operator" not in participants


def test_extract_metrics_revenue(parser):
    transcript = {
        "prepared_remarks": "Revenue was $1.2 billion this quarter.",
        "qa_session": "",
    }

    metrics = parser.extract_metrics(transcript)

    # Should find revenue metric
    revenue_metrics = [m for m in metrics if m["metric_type"] == "revenue"]
    assert len(revenue_metrics) > 0
    assert revenue_metrics[0]["value"] == 1.2
    assert revenue_metrics[0]["unit"] in ["billion", "B"]


def test_extract_metrics_earnings(parser):
    transcript = {
        "prepared_remarks": "Earnings per share was $2.50",
        "qa_session": "",
    }

    metrics = parser.extract_metrics(transcript)

    earnings_metrics = [m for m in metrics if m["metric_type"] == "earnings"]
    assert len(earnings_metrics) > 0
    assert earnings_metrics[0]["value"] == 2.50


def test_extract_metrics_margin(parser):
    transcript = {
        "prepared_remarks": "Gross margin improved to 45%",
        "qa_session": "",
    }

    metrics = parser.extract_metrics(transcript)

    margin_metrics = [m for m in metrics if m["metric_type"] == "margin"]
    assert len(margin_metrics) > 0
    assert margin_metrics[0]["value"] == 45


def test_extract_metrics_growth(parser):
    transcript = {
        "prepared_remarks": "Revenue growth of 15% year-over-year",
        "qa_session": "",
    }

    metrics = parser.extract_metrics(transcript)

    growth_metrics = [m for m in metrics if m["metric_type"] == "growth"]
    assert len(growth_metrics) > 0
    assert growth_metrics[0]["value"] == 15


def test_extract_metrics_multiple(parser, sample_transcript):
    result = parser.parse(sample_transcript)
    metrics = parser.extract_metrics(result)

    # Should extract multiple metrics
    assert len(metrics) > 0

    metric_types = {m["metric_type"] for m in metrics}
    assert "revenue" in metric_types
    assert "margin" in metric_types or "earnings" in metric_types


def test_sentiment_sections(parser, sample_transcript):
    result = parser.parse(sample_transcript)
    sentiment = parser.sentiment_sections(result)

    assert "prepared_remarks" in sentiment
    assert "qa_session" in sentiment

    # Check sentiment structure
    prepared_sentiment = sentiment["prepared_remarks"]
    assert "score" in prepared_sentiment
    assert "positive_count" in prepared_sentiment
    assert "negative_count" in prepared_sentiment
    assert "total_words" in prepared_sentiment


def test_sentiment_positive(parser):
    transcript = {
        "prepared_remarks": "We had strong growth and excellent performance. "
                          "Very optimistic about the future.",
        "qa_session": "",
    }

    sentiment = parser.sentiment_sections(transcript)
    prepared_sentiment = sentiment["prepared_remarks"]

    # Should be positive
    assert prepared_sentiment["score"] > 0
    assert prepared_sentiment["positive_count"] > 0


def test_sentiment_negative(parser):
    transcript = {
        "prepared_remarks": "We face challenging conditions with weak demand. "
                          "Disappointing results and declining margins.",
        "qa_session": "",
    }

    sentiment = parser.sentiment_sections(transcript)
    prepared_sentiment = sentiment["prepared_remarks"]

    # Should be negative
    assert prepared_sentiment["score"] < 0
    assert prepared_sentiment["negative_count"] > 0


def test_sentiment_neutral(parser):
    transcript = {
        "prepared_remarks": "The company operates in various markets. "
                          "We have several products and services.",
        "qa_session": "",
    }

    sentiment = parser.sentiment_sections(transcript)
    prepared_sentiment = sentiment["prepared_remarks"]

    # Should be neutral (score near 0)
    assert abs(prepared_sentiment["score"]) < 0.5


def test_calculate_sentiment_empty(parser):
    sentiment = parser._calculate_sentiment("")

    assert sentiment["score"] == 0.0
    assert sentiment["positive_count"] == 0
    assert sentiment["negative_count"] == 0
    assert sentiment["total_words"] == 0
