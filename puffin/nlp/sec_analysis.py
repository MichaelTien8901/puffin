"""
SEC filing analysis using word embeddings.

This module provides tools for analyzing SEC filings (10-K, 10-Q, etc.) using
word embeddings to track changes in language, sentiment, and topics over time.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from scipy.spatial.distance import cosine


class SECFilingAnalyzer:
    """
    Analyze SEC filings using word embeddings.

    This class provides methods to extract and analyze key sections of SEC filings,
    compare filings across time periods, and detect significant language changes.

    Example:
        >>> from puffin.nlp.embeddings import Word2VecTrainer
        >>> analyzer = SECFilingAnalyzer()
        >>> model = Word2VecTrainer().load('word2vec.model')
        >>> result = analyzer.analyze_10k(text, model)
    """

    def __init__(self):
        """Initialize SECFilingAnalyzer."""
        pass

    def _extract_section(self, text: str, section_pattern: str) -> str:
        """
        Extract a section from SEC filing text.

        Args:
            text: Full filing text
            section_pattern: Regex pattern to match section

        Returns:
            Extracted section text
        """
        # Common section markers in SEC filings
        match = re.search(section_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0)
        return ""

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (lowercase words)
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens

    def _get_embedding(self, text: str, model) -> np.ndarray:
        """
        Get embedding for text using a word embedding model.

        Args:
            text: Text to embed
            model: Word embedding model (Word2Vec, GloVe, etc.)

        Returns:
            Text embedding (average of word vectors)
        """
        tokens = self._tokenize(text)

        vectors = []
        for token in tokens:
            try:
                # Try Word2Vec API
                if hasattr(model, 'wv'):
                    vectors.append(model.wv[token])
                # Try GloVe API
                elif hasattr(model, 'word_vector'):
                    vec = model.word_vector(token)
                    if vec is not None:
                        vectors.append(vec)
                # Try direct indexing
                else:
                    vectors.append(model[token])
            except (KeyError, AttributeError):
                continue

        if not vectors:
            # Return zero vector if no words found
            vector_size = self._get_vector_size(model)
            return np.zeros(vector_size)

        return np.mean(vectors, axis=0)

    def _get_vector_size(self, model) -> int:
        """
        Get vector size from embedding model.

        Args:
            model: Word embedding model

        Returns:
            Vector dimensionality
        """
        if hasattr(model, 'vector_size'):
            return model.vector_size
        elif hasattr(model, 'wv') and hasattr(model.wv, 'vector_size'):
            return model.wv.vector_size
        else:
            # Try to infer from first vector
            try:
                if hasattr(model, 'vectors') and len(model.vectors) > 0:
                    first_vec = next(iter(model.vectors.values()))
                    return len(first_vec)
            except:
                pass
        return 100  # Default fallback

    def analyze_10k(
        self,
        text: str,
        model,
        prior_text: Optional[str] = None
    ) -> Dict[str, Union[np.ndarray, float, None]]:
        """
        Analyze a 10-K filing and extract key section embeddings.

        Args:
            text: 10-K filing text
            model: Word embedding model
            prior_text: Prior year's 10-K text for comparison (optional)

        Returns:
            Dictionary with:
                - risk_factors_embedding: Embedding of risk factors section
                - mda_embedding: Embedding of MD&A section
                - full_text_embedding: Embedding of full text
                - similarity_to_prior: Cosine similarity to prior filing (if provided)

        Example:
            >>> result = analyzer.analyze_10k(text_2023, model, prior_text=text_2022)
            >>> print(result['similarity_to_prior'])
            0.85
        """
        # Extract key sections
        # Item 1A: Risk Factors
        risk_factors = self._extract_section(
            text,
            r'item\s+1a\.?\s+risk\s+factors.*?(?=item\s+1b|item\s+2|\Z)'
        )

        # Item 7: MD&A (Management's Discussion and Analysis)
        mda = self._extract_section(
            text,
            r'item\s+7\.?\s+management.*?discussion.*?analysis.*?(?=item\s+7a|item\s+8|\Z)'
        )

        # Get embeddings
        result = {
            'risk_factors_embedding': self._get_embedding(risk_factors, model) if risk_factors else None,
            'mda_embedding': self._get_embedding(mda, model) if mda else None,
            'full_text_embedding': self._get_embedding(text, model),
            'similarity_to_prior': None
        }

        # Calculate similarity to prior filing if provided
        if prior_text is not None:
            prior_embedding = self._get_embedding(prior_text, model)
            current_embedding = result['full_text_embedding']

            if np.any(prior_embedding) and np.any(current_embedding):
                # Cosine similarity (1 - cosine distance)
                similarity = 1 - cosine(current_embedding, prior_embedding)
                result['similarity_to_prior'] = float(similarity)

        return result

    def compare_filings(
        self,
        texts: List[str],
        dates: List[Union[str, datetime]],
        model
    ) -> pd.DataFrame:
        """
        Compare multiple filings over time.

        Args:
            texts: List of filing texts
            dates: List of filing dates
            model: Word embedding model

        Returns:
            DataFrame with columns:
                - date: Filing date
                - similarity_to_previous: Cosine similarity to previous filing
                - change_magnitude: 1 - similarity (larger = more change)
                - embedding: Full text embedding

        Example:
            >>> texts = [text_2021, text_2022, text_2023]
            >>> dates = ['2021-12-31', '2022-12-31', '2023-12-31']
            >>> df = analyzer.compare_filings(texts, dates, model)
            >>> print(df[['date', 'similarity_to_previous']])
        """
        if len(texts) != len(dates):
            raise ValueError("Number of texts must match number of dates")

        # Convert dates to datetime if needed
        dates = [pd.to_datetime(d) if isinstance(d, str) else d for d in dates]

        # Sort by date
        sorted_pairs = sorted(zip(dates, texts), key=lambda x: x[0])
        dates, texts = zip(*sorted_pairs)

        results = []
        prev_embedding = None

        for date, text in zip(dates, texts):
            embedding = self._get_embedding(text, model)

            similarity_to_prev = None
            change_magnitude = None

            if prev_embedding is not None and np.any(prev_embedding) and np.any(embedding):
                similarity_to_prev = 1 - cosine(embedding, prev_embedding)
                change_magnitude = 1 - similarity_to_prev

            results.append({
                'date': date,
                'similarity_to_previous': similarity_to_prev,
                'change_magnitude': change_magnitude,
                'embedding': embedding
            })

            prev_embedding = embedding

        return pd.DataFrame(results)

    def detect_language_changes(
        self,
        texts: List[str],
        dates: List[Union[str, datetime]],
        threshold: float = 0.15
    ) -> List[Dict[str, Union[str, float, List[str]]]]:
        """
        Detect significant language changes between filings.

        This method identifies filings where the language has changed significantly
        compared to the previous filing, based on vocabulary overlap and new terms.

        Args:
            texts: List of filing texts
            dates: List of filing dates
            threshold: Threshold for detecting significant change (0-1)

        Returns:
            List of dictionaries with:
                - date: Filing date
                - change_score: Magnitude of change (0-1)
                - new_terms: List of new terms appearing in this filing
                - removed_terms: List of terms removed from previous filing

        Example:
            >>> changes = analyzer.detect_language_changes(texts, dates, threshold=0.2)
            >>> for change in changes:
            ...     print(f"{change['date']}: {change['change_score']:.2f}")
        """
        if len(texts) != len(dates):
            raise ValueError("Number of texts must match number of dates")

        # Convert dates to datetime if needed
        dates = [pd.to_datetime(d) if isinstance(d, str) else d for d in dates]

        # Sort by date
        sorted_pairs = sorted(zip(dates, texts), key=lambda x: x[0])
        dates, texts = zip(*sorted_pairs)

        changes = []
        prev_vocab = None

        for i, (date, text) in enumerate(zip(dates, texts)):
            tokens = self._tokenize(text)
            vocab = set(tokens)

            if prev_vocab is not None:
                # Calculate Jaccard similarity
                intersection = len(vocab & prev_vocab)
                union = len(vocab | prev_vocab)
                jaccard_sim = intersection / union if union > 0 else 0
                change_score = 1 - jaccard_sim

                # Identify new and removed terms
                new_terms = list(vocab - prev_vocab)[:50]  # Top 50 new terms
                removed_terms = list(prev_vocab - vocab)[:50]  # Top 50 removed terms

                # Only record if change exceeds threshold
                if change_score >= threshold:
                    changes.append({
                        'date': str(date),
                        'change_score': float(change_score),
                        'new_terms': new_terms,
                        'removed_terms': removed_terms
                    })

            prev_vocab = vocab

        return changes

    def extract_risk_sentiment(
        self,
        text: str,
        model,
        risk_keywords: Optional[List[str]] = None
    ) -> Dict[str, Union[float, List[Tuple[str, float]]]]:
        """
        Extract risk-related sentiment from filing text.

        Args:
            text: Filing text
            model: Word embedding model
            risk_keywords: List of risk-related keywords (optional)

        Returns:
            Dictionary with:
                - risk_score: Overall risk score (0-1)
                - top_risk_terms: List of (term, score) tuples

        Example:
            >>> sentiment = analyzer.extract_risk_sentiment(text, model)
            >>> print(f"Risk score: {sentiment['risk_score']:.2f}")
        """
        if risk_keywords is None:
            risk_keywords = [
                'risk', 'uncertainty', 'volatile', 'litigation', 'regulation',
                'adverse', 'decline', 'loss', 'failure', 'default'
            ]

        tokens = self._tokenize(text)
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        # Calculate risk-related term frequencies
        total_tokens = len(tokens)
        risk_scores = []

        for keyword in risk_keywords:
            try:
                # Get embedding for risk keyword
                if hasattr(model, 'wv'):
                    keyword_vec = model.wv[keyword]
                elif hasattr(model, 'word_vector'):
                    keyword_vec = model.word_vector(keyword)
                else:
                    continue

                # Find similar terms in document
                for token, count in token_counts.items():
                    try:
                        if hasattr(model, 'wv'):
                            token_vec = model.wv[token]
                        elif hasattr(model, 'word_vector'):
                            token_vec = model.word_vector(token)
                        else:
                            continue

                        # Calculate similarity
                        similarity = 1 - cosine(keyword_vec, token_vec)
                        if similarity > 0.5:  # Only count similar terms
                            frequency = count / total_tokens
                            risk_scores.append((token, similarity * frequency))
                    except (KeyError, AttributeError):
                        continue
            except (KeyError, AttributeError):
                continue

        # Sort by risk score
        risk_scores.sort(key=lambda x: x[1], reverse=True)
        top_risk_terms = risk_scores[:20]

        # Calculate overall risk score
        risk_score = sum(score for _, score in top_risk_terms)

        return {
            'risk_score': float(min(risk_score, 1.0)),  # Cap at 1.0
            'top_risk_terms': top_risk_terms
        }
