"""Parser for earnings call transcripts."""

import re
from typing import Optional


class TranscriptParser:
    """Parser for earnings call transcripts.

    Examples:
        >>> parser = TranscriptParser()
        >>> parsed = parser.parse(raw_transcript_text)
        >>> metrics = parser.extract_metrics(parsed)
        >>> sentiment = parser.sentiment_sections(parsed)
    """

    def __init__(self):
        """Initialize the transcript parser."""
        # Common section markers in earnings transcripts
        self.section_markers = {
            "prepared_remarks": [
                "prepared remarks",
                "opening remarks",
                "management discussion",
                "operator",
            ],
            "qa_session": [
                "question-and-answer",
                "q&a session",
                "questions and answers",
                "operator: our first question",
            ],
        }

        # Financial metric patterns
        self.metric_patterns = {
            "revenue": r"revenue[s]?\s+(?:of|was|were|at)?\s*\$?([\d,.]+)\s*(million|billion|M|B)?",
            "earnings": r"(?:earnings|EPS|earnings per share)\s+(?:of|was|were|at)?\s*\$?([\d,.]+)",
            "margin": r"(?:gross|operating|net)?\s*margin[s]?\s+(?:of|was|were|at)?\s*([\d,.]+)%?",
            "growth": r"(?:revenue|sales|earnings)?\s*growth\s+(?:of|was|were|at)?\s*([\d,.]+)%?",
            "guidance": r"guidance\s+(?:of|for)?\s*\$?([\d,.]+)\s*(million|billion|M|B)?",
        }

        # Sentiment words
        self.positive_words = {
            "strong", "growth", "improved", "increase", "positive", "excellent",
            "record", "outstanding", "successful", "exceed", "beat", "optimistic",
            "momentum", "opportunities", "robust", "solid", "confident",
        }

        self.negative_words = {
            "weak", "decline", "decrease", "negative", "poor", "challenging",
            "headwind", "pressure", "concern", "risk", "uncertainty", "difficult",
            "lower", "miss", "below", "disappointing", "cautious",
        }

    def parse(self, raw_text: str) -> dict:
        """Parse raw transcript text into structured sections.

        Args:
            raw_text: Raw transcript text

        Returns:
            Dictionary with keys: prepared_remarks, qa_session, participants
        """
        if not raw_text:
            return {
                "prepared_remarks": "",
                "qa_session": "",
                "participants": [],
            }

        # Normalize text
        text = raw_text.strip()
        text_lower = text.lower()

        # Find section boundaries
        prepared_remarks = ""
        qa_session = ""

        # Look for Q&A section start
        qa_start = None
        for marker in self.section_markers["qa_session"]:
            match = re.search(re.escape(marker), text_lower)
            if match:
                qa_start = match.start()
                break

        if qa_start is not None:
            prepared_remarks = text[:qa_start].strip()
            qa_session = text[qa_start:].strip()
        else:
            # If no Q&A marker found, treat entire text as prepared remarks
            prepared_remarks = text

        # Extract participants (people who spoke)
        participants = self._extract_participants(text)

        return {
            "prepared_remarks": prepared_remarks,
            "qa_session": qa_session,
            "participants": participants,
        }

    def _extract_participants(self, text: str) -> list[str]:
        """Extract participant names from transcript.

        Args:
            text: Transcript text

        Returns:
            List of participant names
        """
        # Common patterns for speaker identification
        # "John Smith - CEO:" or "John Smith, Chief Executive Officer:"
        speaker_pattern = r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[-:,]"

        participants = set()
        for line in text.split("\n"):
            match = re.match(speaker_pattern, line.strip())
            if match:
                name = match.group(1).strip()
                # Filter out common false positives
                if len(name.split()) >= 2 and name not in ["Operator", "Conference Call"]:
                    participants.add(name)

        return sorted(list(participants))

    def extract_metrics(self, transcript: dict) -> list[dict]:
        """Extract mentioned financial metrics from transcript.

        Args:
            transcript: Parsed transcript dictionary from parse()

        Returns:
            List of dictionaries with metric_type, value, unit, context
        """
        metrics = []

        # Combine all text sections
        full_text = transcript.get("prepared_remarks", "") + " " + transcript.get("qa_session", "")

        for metric_name, pattern in self.metric_patterns.items():
            matches = re.finditer(pattern, full_text, re.IGNORECASE)

            for match in matches:
                # Extract context (surrounding text)
                start = max(0, match.start() - 100)
                end = min(len(full_text), match.end() + 100)
                context = full_text[start:end].strip()

                # Parse value and unit
                value = None
                unit = None

                if len(match.groups()) >= 1:
                    value_str = match.group(1).replace(",", "")
                    try:
                        value = float(value_str)
                    except ValueError:
                        continue

                if len(match.groups()) >= 2:
                    unit = match.group(2)

                metrics.append({
                    "metric_type": metric_name,
                    "value": value,
                    "unit": unit,
                    "context": context,
                })

        return metrics

    def sentiment_sections(self, transcript: dict) -> dict:
        """Analyze sentiment of different transcript sections.

        Args:
            transcript: Parsed transcript dictionary from parse()

        Returns:
            Dictionary mapping section names to sentiment scores
        """
        results = {}

        for section_name in ["prepared_remarks", "qa_session"]:
            text = transcript.get(section_name, "")
            if text:
                results[section_name] = self._calculate_sentiment(text)

        return results

    def _calculate_sentiment(self, text: str) -> dict:
        """Calculate basic sentiment score for text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment metrics
        """
        # Convert to lowercase and tokenize
        words = re.findall(r'\b\w+\b', text.lower())

        if not words:
            return {
                "score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "total_words": 0,
            }

        # Count sentiment words
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            score = (positive_count - negative_count) / total_sentiment_words
        else:
            score = 0.0

        return {
            "score": score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_words": len(words),
        }
