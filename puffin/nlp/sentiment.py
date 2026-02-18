"""Financial sentiment analysis using lexicon-based approaches."""

import re
from typing import Any

import numpy as np


# Loughran-McDonald financial sentiment word lists (subset)
# Full lists available at: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
LOUGHRAN_MCDONALD_POSITIVE = {
    "able", "abundance", "abundant", "acclaimed", "accomplish", "accomplished",
    "achievement", "achieve", "achieving", "adequate", "advancing", "advantage",
    "advantageous", "affordable", "alliance", "assure", "attain", "attractive",
    "audited", "beneficial", "beneficially", "benefit", "benefiting", "benefitted",
    "best", "better", "bolstered", "bolstering", "boom", "booming", "boost",
    "boosted", "breakthrough", "brilliant", "bullish", "capable", "clarity",
    "collaborative", "commitment", "confident", "constructive", "creative",
    "desired", "desirable", "despite", "distinction", "distinctive", "distinguished",
    "diversified", "diversification", "enhance", "enhanced", "enhancement",
    "excellent", "exceptional", "excited", "exciting", "exemplary", "fantastic",
    "favorable", "favorably", "favored", "friendly", "gain", "gained", "gaining",
    "gains", "good", "great", "greater", "greatest", "greatly", "grow", "growing",
    "grown", "growth", "high", "higher", "highest", "improve",
    "improved", "improvement", "improving", "increase", "increased", "increasing",
    "innovation", "innovative", "leader", "leading", "lucrative", "markedly",
    "notable", "outpace", "outpaced", "outperform", "outperformed", "pleasant",
    "pleased", "popular", "positive", "positively", "premier", "premium",
    "prestigious", "profit", "profitable", "profitably", "profitability", "profits",
    "progress", "progressing", "prominent", "promising", "prosperous", "prosperity",
    "quality", "rebound", "rebounded", "recover", "recovered", "recovering",
    "recovery", "reliable", "remarkable", "resolve", "resolved", "respect",
    "respected", "respectful", "reward", "rewarded", "rewarding", "rise", "rising",
    "robust", "solid", "stability", "stable", "strength", "strengthen", "strengthening",
    "strong", "stronger", "strongest", "succeed", "succeeded", "succeeding",
    "success", "successful", "successfully", "superior", "support", "supported",
    "supportive", "sustainable", "top", "tremendous", "turnaround", "unique",
    "unprecedented", "valuable", "value", "vibrant", "win", "winner", "winning",
}

LOUGHRAN_MCDONALD_NEGATIVE = {
    "abandon", "abandoned", "abandoning", "abandonment", "aborts", "adverse",
    "adversely", "against", "allegations", "allege", "alleged", "alleges",
    "alleging", "annoy", "annoyance", "annoyed", "annoying", "annoys", "anxiety",
    "arrest", "arrested", "arresting", "arrests", "artificially", "attack",
    "attacked", "attacking", "attacks", "bad", "badly", "bankrupt", "bankruptcy",
    "barrier", "barriers", "breach", "breached", "breaches", "breaching", "bribe",
    "burden", "cancel", "cancelled", "cancelling", "challenge", "challenged",
    "challenging", "claim", "claimed", "claiming", "claims", "complain", "complaint",
    "complaints", "concerning", "concerns", "conflict", "conflicting", "conflicts",
    "constrain", "constrained", "constraining", "constraint", "constraints",
    "contaminate", "contaminated", "contamination", "controversial", "controversy",
    "cost", "costly", "costs", "critical", "criticise", "criticised", "criticises",
    "criticising", "criticism", "criticisms", "criticize", "criticized", "criticizes",
    "damage", "damaged", "damages", "damaging", "danger", "dangerous", "dangerously",
    "dangers", "deadlock", "debt", "decline", "declined", "declines", "declining",
    "decrease", "decreased", "decreases", "decreasing", "defect", "defective",
    "defects", "deficiency", "deficit", "degrade", "delay", "delayed", "delaying",
    "delays", "denial", "denied", "denies", "deny", "denying", "deteriorate",
    "deteriorated", "deteriorates", "deteriorating", "deterioration", "detriment",
    "detrimental", "difficult", "difficulties", "difficulty", "diminish",
    "diminished", "diminishes", "diminishing", "disadvantage", "disadvantaged",
    "disadvantageous", "disadvantages", "disappoint", "disappointed", "disappointing",
    "disappointment", "disaster", "disastrous", "disclose", "disclosed", "discloses",
    "disclosing", "discontinue", "discontinued", "discontinues", "discontinuing",
    "discourage", "discouraged", "discourages", "discouraging", "dismiss",
    "dismissed", "dismissing", "disruption", "disruptive", "distress", "doubt",
    "doubts", "downturn", "drop", "dropped", "dropping", "drops", "eliminate",
    "eliminated", "eliminates", "eliminating", "elimination", "error", "errors",
    "fail", "failed", "failing", "fails", "failure", "failures", "fall", "fallen",
    "falling", "falls", "false", "falsely", "falsified", "falsifies", "falsify",
    "fault", "faulty", "fear", "fears", "fine", "fined", "fines", "fining",
    "fraud", "fraudulent", "fraudulently", "harm", "harmed", "harmful", "harming",
    "harms", "harsh", "harshly", "hinder", "hindered", "hindering", "hinders",
    "hurt", "hurting", "hurts", "illegal", "illegally", "impair", "impaired",
    "impairing", "impairment", "impairs", "impossible", "inability", "inadequacy",
    "inadequate", "inadvertent", "inadvertently", "incompatible", "incompetence",
    "incompetent", "incorrect", "incorrectly", "ineffective", "ineffectively",
    "inefficiency", "inefficient", "inferior", "inflated", "inflation", "infringe",
    "infringed", "infringement", "infringes", "infringing", "injunction", "injure",
    "injured", "injuries", "injuring", "injury", "insolvency", "insolvent",
    "insufficient", "interrupt", "interrupted", "interruption", "interruptions",
    "interrupts", "investigation", "investigations", "lawsuit", "lawsuits",
    "liability", "liabilities", "liable", "limit", "limitation", "limitations",
    "limited", "limiting", "limits", "liquidate", "liquidated", "liquidating",
    "liquidation", "litigation", "loss", "losses", "lost", "low", "lower",
    "lowest", "lowering", "misapplied", "misappropriate", "misappropriated",
    "misappropriating", "mismanage", "mismanaged", "mismanagement", "misstate",
    "misstated", "misstatement", "misstatements", "misstating", "mistake",
    "mistakes", "negative", "negatively", "negligence", "negligent", "nonpayment",
    "obstacle", "obstacles", "obsolescence", "obsolete", "opposed", "opposes",
    "oppose", "opposing", "opposition", "overpaid", "overpayment", "overpayments",
    "penalty", "penalties", "poor", "poorly", "problem", "problematic", "problems",
    "prosecute", "prosecuted", "prosecutes", "prosecuting", "prosecution",
    "protest", "protested", "protesting", "protests", "question", "questionable",
    "questioned", "questioning", "questions", "recall", "recalled", "recalling",
    "recalls", "recession", "recessionary", "refusal", "refuse", "refused",
    "refuses", "refusing", "reject", "rejected", "rejecting", "rejection",
    "rejects", "repudiate", "repudiated", "repudiates", "repudiating", "resign",
    "resigned", "resigning", "resignations", "resigns", "restate", "restated",
    "restatement", "restatements", "restates", "restating", "restructure",
    "restructured", "restructures", "restructuring", "restructurings", "risk",
    "risks", "risky", "sanction", "sanctioned", "sanctions", "scandal", "scandals",
    "sever", "severe", "severely", "severity", "shrink", "shrinkage", "shrinking",
    "shrinks", "shut", "shutdown", "shutdowns", "shortage", "shortages", "shortfall",
    "shrink", "slow", "slowdown", "slowed", "slower", "slowing", "slowly", "slows",
    "stagnant", "stagnation", "strain", "strained", "strains", "stress", "stressed",
    "stresses", "stressful", "stressing", "stringent", "struggling", "sue", "sued",
    "sues", "suing", "suspend", "suspended", "suspending", "suspends", "suspension",
    "suspensions", "taint", "tainted", "taints", "tariff", "tariffs", "terminate",
    "terminated", "terminates", "terminating", "termination", "terminations",
    "threat", "threaten", "threatened", "threatening", "threatens", "threats",
    "trouble", "troubled", "troubles", "unable", "uncertain", "uncertainly",
    "uncertainties", "uncertainty", "unclear", "uncompetitive", "undermine",
    "undermined", "undermines", "undermining", "underpaid", "underpayment",
    "underpayments", "unfair", "unfavorable", "unfavorably", "unfavourable",
    "unfeasible", "unintended", "unintentional", "unintentionally", "unjust",
    "unjustifiable", "unjustified", "unjustly", "unknown", "unlawful", "unlawfully",
    "unnecessary", "unpaid", "unplanned", "unpopular", "unpredictable",
    "unprofitable", "unqualified", "unrealistic", "unreasonable", "unreasonably",
    "unresolved", "unsafe", "unsatisfactory", "unsound", "unstable", "unsuccessful",
    "unsuccessfully", "unsure", "unsustainable", "untenable", "unusual", "unusually",
    "unwilling", "unwillingness", "urgently", "useless", "violate", "violated",
    "violates", "violating", "violation", "violations", "vulnerability", "vulnerable",
    "warn", "warned", "warning", "warnings", "warns", "weak", "weaken", "weakened",
    "weakening", "weakens", "weaker", "weakness", "weaknesses", "worst", "worthless",
    "wrong", "wrongful", "wrongfully",
}


class RuleSentiment:
    """
    Rule-based sentiment analyzer using Loughran-McDonald financial lexicon.

    Designed specifically for financial text, not general-purpose sentiment.
    """

    def __init__(
        self,
        positive_words: set[str] | None = None,
        negative_words: set[str] | None = None,
        case_sensitive: bool = False
    ):
        """
        Initialize rule-based sentiment analyzer.

        Args:
            positive_words: Set of positive words (defaults to Loughran-McDonald)
            negative_words: Set of negative words (defaults to Loughran-McDonald)
            case_sensitive: Whether to match case-sensitively
        """
        self.positive_words = positive_words or LOUGHRAN_MCDONALD_POSITIVE
        self.negative_words = negative_words or LOUGHRAN_MCDONALD_NEGATIVE
        self.case_sensitive = case_sensitive

        if not case_sensitive:
            self.positive_words = {w.lower() for w in self.positive_words}
            self.negative_words = {w.lower() for w in self.negative_words}

    def score(self, text: str) -> float:
        """
        Compute sentiment score for text.

        Args:
            text: Input text

        Returns:
            Sentiment score in range [-1, 1]
            Positive = bullish, Negative = bearish, ~0 = neutral
        """
        words = self._tokenize(text)

        if not words:
            return 0.0

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0

        # Score normalized to [-1, 1]
        score = (positive_count - negative_count) / len(words)

        return max(-1.0, min(1.0, score))

    def batch_score(self, texts: list[str]) -> list[float]:
        """
        Score multiple texts.

        Args:
            texts: List of text strings

        Returns:
            List of sentiment scores
        """
        return [self.score(text) for text in texts]

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyze text and return detailed sentiment information.

        Args:
            text: Input text

        Returns:
            Dictionary with score, label, and matched words
        """
        words = self._tokenize(text)

        positive_matches = [w for w in words if w in self.positive_words]
        negative_matches = [w for w in words if w in self.negative_words]

        score = self.score(text)

        # Determine label
        if score > 0.05:
            label = "bullish"
        elif score < -0.05:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "score": score,
            "label": label,
            "positive_count": len(positive_matches),
            "negative_count": len(negative_matches),
            "positive_words": positive_matches,
            "negative_words": negative_matches,
            "total_words": len(words),
        }

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        if not self.case_sensitive:
            text = text.lower()

        # Extract words using regex
        words = re.findall(r'\b\w+\b', text)
        return words

    def __repr__(self) -> str:
        return (
            f"RuleSentiment("
            f"positive={len(self.positive_words)}, "
            f"negative={len(self.negative_words)}, "
            f"case_sensitive={self.case_sensitive})"
        )


class LexiconSentiment:
    """
    Lexicon-based sentiment analyzer with customizable word lists.

    More flexible than RuleSentiment, allows custom weighted lexicons.
    """

    def __init__(
        self,
        positive_words: dict[str, float] | set[str] | None = None,
        negative_words: dict[str, float] | set[str] | None = None,
        case_sensitive: bool = False
    ):
        """
        Initialize lexicon-based sentiment analyzer.

        Args:
            positive_words: Dictionary of word -> weight or set of words (weight=1.0)
            negative_words: Dictionary of word -> weight or set of words (weight=1.0)
            case_sensitive: Whether to match case-sensitively
        """
        self.case_sensitive = case_sensitive

        # Convert sets to dictionaries with uniform weights
        if isinstance(positive_words, set) or positive_words is None:
            words = positive_words or LOUGHRAN_MCDONALD_POSITIVE
            self.positive_words = {w: 1.0 for w in words}
        else:
            self.positive_words = positive_words

        if isinstance(negative_words, set) or negative_words is None:
            words = negative_words or LOUGHRAN_MCDONALD_NEGATIVE
            self.negative_words = {w: 1.0 for w in words}
        else:
            self.negative_words = negative_words

        if not case_sensitive:
            self.positive_words = {k.lower(): v for k, v in self.positive_words.items()}
            self.negative_words = {k.lower(): v for k, v in self.negative_words.items()}

    def score(self, text: str) -> float:
        """
        Compute weighted sentiment score for text.

        Args:
            text: Input text

        Returns:
            Sentiment score in range [-1, 1]
        """
        words = self._tokenize(text)

        if not words:
            return 0.0

        positive_score = sum(
            self.positive_words.get(word, 0.0) for word in words
        )
        negative_score = sum(
            self.negative_words.get(word, 0.0) for word in words
        )

        # Normalize by text length
        total_score = (positive_score - negative_score) / len(words)

        return np.tanh(total_score)  # Squash to [-1, 1]

    def batch_score(self, texts: list[str]) -> list[float]:
        """
        Score multiple texts.

        Args:
            texts: List of text strings

        Returns:
            List of sentiment scores
        """
        return [self.score(text) for text in texts]

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyze text and return detailed sentiment information.

        Args:
            text: Input text

        Returns:
            Dictionary with score, label, and matched words with weights
        """
        words = self._tokenize(text)

        positive_matches = {
            w: self.positive_words[w]
            for w in words if w in self.positive_words
        }
        negative_matches = {
            w: self.negative_words[w]
            for w in words if w in self.negative_words
        }

        score = self.score(text)

        # Determine label
        if score > 0.05:
            label = "bullish"
        elif score < -0.05:
            label = "bearish"
        else:
            label = "neutral"

        return {
            "score": score,
            "label": label,
            "positive_score": sum(positive_matches.values()),
            "negative_score": sum(negative_matches.values()),
            "positive_words": positive_matches,
            "negative_words": negative_matches,
            "total_words": len(words),
        }

    def add_positive_word(self, word: str, weight: float = 1.0) -> None:
        """Add a positive word to lexicon."""
        if not self.case_sensitive:
            word = word.lower()
        self.positive_words[word] = weight

    def add_negative_word(self, word: str, weight: float = 1.0) -> None:
        """Add a negative word to lexicon."""
        if not self.case_sensitive:
            word = word.lower()
        self.negative_words[word] = weight

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        if not self.case_sensitive:
            text = text.lower()

        words = re.findall(r'\b\w+\b', text)
        return words

    def __repr__(self) -> str:
        return (
            f"LexiconSentiment("
            f"positive={len(self.positive_words)}, "
            f"negative={len(self.negative_words)}, "
            f"case_sensitive={self.case_sensitive})"
        )
