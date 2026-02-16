"""NLP pipeline with spaCy for text processing and entity extraction."""

import re
from dataclasses import dataclass, field
from typing import Any

# Try importing spaCy, but provide fallback if not available
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Language = Any


@dataclass
class ProcessedDoc:
    """Container for processed document information."""
    text: str
    tokens: list[str] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)
    entities: list[tuple[str, str]] = field(default_factory=list)  # (text, label)
    sentences: list[str] = field(default_factory=list)
    pos_tags: list[str] = field(default_factory=list)


class SimpleTokenizer:
    """Simple regex-based tokenizer as fallback when spaCy is not available."""

    def __init__(self):
        self.word_pattern = re.compile(r'\b\w+\b')
        self.sentence_pattern = re.compile(r'[.!?]+')

    def tokenize(self, text: str) -> list[str]:
        """Extract tokens from text using regex."""
        return self.word_pattern.findall(text.lower())

    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]


class NLPPipeline:
    """
    NLP pipeline for processing text with spaCy or fallback tokenizer.

    Attributes:
        nlp: spaCy language model or None if not available
        use_fallback: Whether to use simple regex tokenizer
    """

    def __init__(self, model: str = "en_core_web_sm", disable: list[str] | None = None):
        """
        Initialize NLP pipeline.

        Args:
            model: spaCy model name to load
            disable: List of pipeline components to disable for speed
        """
        self.use_fallback = not SPACY_AVAILABLE
        self.nlp = None

        if not self.use_fallback:
            try:
                self.nlp = spacy.load(model, disable=disable or [])
            except OSError:
                # Model not downloaded, try to use blank model
                try:
                    self.nlp = spacy.blank("en")
                    self.use_fallback = True
                except Exception:
                    self.use_fallback = True

        if self.use_fallback:
            self.tokenizer = SimpleTokenizer()

    def process(self, text: str) -> ProcessedDoc:
        """
        Process text and extract linguistic features.

        Args:
            text: Input text to process

        Returns:
            ProcessedDoc with tokens, lemmas, entities, and sentences
        """
        if self.use_fallback:
            return self._process_fallback(text)

        doc = self.nlp(text)

        tokens = [token.text for token in doc]
        lemmas = [token.lemma_ for token in doc]
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        sentences = [sent.text for sent in doc.sents]
        pos_tags = [token.pos_ for token in doc]

        return ProcessedDoc(
            text=text,
            tokens=tokens,
            lemmas=lemmas,
            entities=entities,
            sentences=sentences,
            pos_tags=pos_tags
        )

    def _process_fallback(self, text: str) -> ProcessedDoc:
        """Process text using simple regex tokenizer."""
        tokens = self.tokenizer.tokenize(text)
        sentences = self.tokenizer.split_sentences(text)

        return ProcessedDoc(
            text=text,
            tokens=tokens,
            lemmas=tokens,  # No lemmatization in fallback
            entities=[],
            sentences=sentences,
            pos_tags=[]
        )

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        """
        Extract named entities focusing on financial relevance.

        Extracts: ORG, MONEY, PERCENT, DATE entities.

        Args:
            text: Input text

        Returns:
            List of (entity_text, entity_label) tuples
        """
        if self.use_fallback:
            return self._extract_entities_fallback(text)

        doc = self.nlp(text)
        relevant_labels = {"ORG", "MONEY", "PERCENT", "DATE", "GPE", "PRODUCT"}

        entities = [
            (ent.text, ent.label_)
            for ent in doc.ents
            if ent.label_ in relevant_labels
        ]

        return entities

    def _extract_entities_fallback(self, text: str) -> list[tuple[str, str]]:
        """Extract entities using regex patterns."""
        entities = []

        # Extract percentages
        percent_pattern = re.compile(r'\b\d+(?:\.\d+)?%')
        for match in percent_pattern.finditer(text):
            entities.append((match.group(), "PERCENT"))

        # Extract money amounts
        money_pattern = re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:[BM])?')
        for match in money_pattern.finditer(text):
            entities.append((match.group(), "MONEY"))

        # Extract dates (simple patterns)
        date_pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b')
        for match in date_pattern.finditer(text):
            entities.append((match.group(), "DATE"))

        return entities

    def extract_financial_terms(self, text: str) -> list[str]:
        """
        Extract financial and trading terms from text.

        Args:
            text: Input text

        Returns:
            List of financial terms found in text
        """
        financial_keywords = {
            # Market terms
            "stock", "stocks", "share", "shares", "equity", "equities",
            "bond", "bonds", "treasury", "treasuries",
            "option", "options", "derivative", "derivatives",
            "futures", "forward", "swap", "swaps",

            # Market participants
            "investor", "investors", "trader", "traders",
            "analyst", "analysts", "broker", "brokers",

            # Financial metrics
            "price", "prices", "volume", "volatility",
            "earnings", "revenue", "profit", "loss",
            "ebitda", "eps", "pe", "p/e", "dividend", "yield",
            "margin", "margins", "valuation",

            # Actions
            "buy", "buying", "sell", "selling", "trade", "trading",
            "invest", "investing", "hedge", "hedging",
            "short", "shorting", "long",

            # Market conditions
            "bull", "bullish", "bear", "bearish",
            "rally", "rallied", "decline", "declined",
            "growth", "recession", "inflation", "deflation",

            # Corporate actions
            "merger", "acquisition", "buyback", "ipo",
            "dividend", "split", "spinoff",

            # Instruments
            "etf", "index", "indices", "fund", "funds",
            "portfolio", "portfolios", "asset", "assets",

            # Analysis
            "technical", "fundamental", "analysis",
            "indicator", "indicators", "signal", "signals",
            "trend", "trends", "support", "resistance",
        }

        # Tokenize and extract matching terms
        tokens = self.tokenizer.tokenize(text) if self.use_fallback else [
            token.text.lower() for token in self.nlp(text)
        ]

        found_terms = [token for token in tokens if token in financial_keywords]

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in found_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms

    def batch_process(self, texts: list[str]) -> list[ProcessedDoc]:
        """
        Process multiple texts efficiently.

        Args:
            texts: List of text strings to process

        Returns:
            List of ProcessedDoc objects
        """
        if self.use_fallback:
            return [self._process_fallback(text) for text in texts]

        # Use spaCy's pipe for efficient batch processing
        results = []
        for doc in self.nlp.pipe(texts):
            tokens = [token.text for token in doc]
            lemmas = [token.lemma_ for token in doc]
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            sentences = [sent.text for sent in doc.sents]
            pos_tags = [token.pos_ for token in doc]

            results.append(ProcessedDoc(
                text=doc.text,
                tokens=tokens,
                lemmas=lemmas,
                entities=entities,
                sentences=sentences,
                pos_tags=pos_tags
            ))

        return results
