"""
llm-response-classifier: Classify LLM response text into labeled categories using rule patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClassifyResult:
    """Outcome of classifying a single piece of text.

    Attributes:
        label: The winning class label, or the classifier's default label when
            nothing matched.
        confidence: A 0.0-1.0 value equal to the winning class's score divided
            by the sum of all class scores. ``1.0`` means only one class
            matched; ``0.0`` means nothing matched.
        matched_pattern: The first raw pattern (as registered) that matched for
            the winning class, or ``None`` when nothing matched.
        scores: The raw ``(match_fraction * weight)`` score for every
            registered class, keyed by label.
    """

    label: str
    confidence: float
    matched_pattern: Optional[str]
    scores: dict[str, float]

    @property
    def is_uncertain(self) -> bool:
        """True when confidence is below 0.5 (an ambiguous or no-match result)."""
        return self.confidence < 0.5


@dataclass
class _Class:
    label: str
    patterns: list[str]
    weight: float = 1.0
    _compiled: list[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        try:
            self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        except re.error as exc:
            raise ValueError(
                f"invalid regex pattern for class {self.label!r}: {exc}"
            ) from exc

    def score(self, text: str) -> tuple[float, Optional[str]]:
        """Return ``(match_fraction * weight, first_matched_pattern)``.

        ``match_fraction`` is the share of this class's patterns that matched
        ``text`` (0.0-1.0). A class with no patterns always scores 0.0.
        """
        if not self._compiled:
            return 0.0, None
        matches = 0
        first_match: Optional[str] = None
        for p, raw in zip(self._compiled, self.patterns):
            if p.search(text):
                if first_match is None:
                    first_match = raw
                matches += 1
        return (matches / len(self._compiled)) * self.weight, first_match


class ResponseClassifier:
    """
    Classify LLM response text into named categories.

    Usage::

        clf = ResponseClassifier()
        clf.add_class("affirm", patterns=["yes", "sure", "absolutely"])
        clf.add_class("deny",   patterns=["no", "cannot", "won't"])

        result = clf.classify("Yes, I can help with that!")
        # result.label == "affirm"
    """

    def __init__(self, default_label: str = "unknown") -> None:
        self._classes: list[_Class] = []
        self._default = default_label

    def add_class(
        self,
        label: str,
        patterns: list[str],
        weight: float = 1.0,
    ) -> "ResponseClassifier":
        """Register a class with regex patterns (case-insensitive).

        Args:
            label: Name for this class. Must be non-empty.
            patterns: Regex strings; a class matches a text when any of them
                does. Compiled with ``re.IGNORECASE``.
            weight: Multiplier applied to this class's match fraction. Must be
                positive. Higher weights make the class win ties and dominate
                scoring.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            ValueError: If ``label`` is empty, ``weight`` is not positive, or
                any pattern is not a valid regular expression.
        """
        if not label:
            raise ValueError("label must be a non-empty string")
        if weight <= 0:
            raise ValueError(f"weight must be positive, got {weight!r}")
        self._classes.append(_Class(label=label, patterns=patterns, weight=weight))
        return self

    def classify(self, text: str) -> ClassifyResult:
        """Classify ``text`` against all registered classes.

        Args:
            text: The text to classify. ``None`` is treated as an empty string.

        Returns:
            A :class:`ClassifyResult`. When no class matches (or none are
            registered), the result uses the classifier's default label with a
            confidence of 0.0.
        """
        if text is None:
            text = ""
        if not self._classes:
            return ClassifyResult(
                label=self._default,
                confidence=0.0,
                matched_pattern=None,
                scores={},
            )

        scores: dict[str, float] = {}
        first_matches: dict[str, Optional[str]] = {}
        for cls in self._classes:
            s, fm = cls.score(text)
            scores[cls.label] = s
            first_matches[cls.label] = fm

        best_label = max(scores, key=lambda l: scores[l])
        best_score = scores[best_label]

        if best_score == 0.0:
            return ClassifyResult(
                label=self._default,
                confidence=0.0,
                matched_pattern=None,
                scores=scores,
            )

        total = sum(scores.values()) or 1.0
        confidence = min(best_score / total, 1.0)

        return ClassifyResult(
            label=best_label,
            confidence=confidence,
            matched_pattern=first_matches.get(best_label),
            scores=scores,
        )

    def classify_many(self, texts: list[str]) -> list[ClassifyResult]:
        """Classify a list of texts, returning one result per input."""
        return [self.classify(t) for t in texts]

    @property
    def labels(self) -> list[str]:
        """The labels of all registered classes, in registration order."""
        return [c.label for c in self._classes]


# -- built-in classifiers -------------------------------------------------

def yesno_classifier() -> ResponseClassifier:
    """Pre-built yes/no classifier."""
    clf = ResponseClassifier(default_label="unknown")
    clf.add_class("yes", [
        r"\byes\b", r"\bsure\b", r"\babsolutely\b", r"\bcertainly\b",
        r"\bof course\b", r"\bcorrect\b", r"\baffirmative\b", r"\byeah\b",
        r"\bI can\b", r"\bI will\b",
    ])
    clf.add_class("no", [
        r"\bno\b", r"\bnot\b", r"\bnever\b", r"\bcannot\b", r"\bcan't\b",
        r"\bwon't\b", r"\bunable\b", r"\brefuse\b", r"\bdeny\b", r"\bnegative\b",
    ])
    return clf


def sentiment_classifier() -> ResponseClassifier:
    """Pre-built positive/neutral/negative sentiment classifier."""
    clf = ResponseClassifier(default_label="neutral")
    clf.add_class("positive", [
        r"\bgreat\b", r"\bexcellent\b", r"\bhappy\b", r"\bwonderful\b",
        r"\bfantastic\b", r"\blove\b", r"\bperfect\b", r"\bamazing\b",
    ])
    clf.add_class("negative", [
        r"\bterrible\b", r"\bawful\b", r"\bhate\b", r"\bworse\b",
        r"\bbad\b", r"\bhorrible\b", r"\bdisappoint\b", r"\bfrustrat\b",
    ])
    clf.add_class("neutral", [
        r"\bokay\b", r"\bfine\b", r"\balright\b", r"\bneutral\b", r"\bso-so\b",
    ])
    return clf


def intent_classifier() -> ResponseClassifier:
    """Pre-built intent classifier: question / statement / instruction."""
    clf = ResponseClassifier(default_label="statement")
    clf.add_class("question", [
        r"\?$", r"^(what|who|where|when|why|how|is|are|do|does|can|could|would|should)\b",
    ])
    clf.add_class("instruction", [
        r"^(please|kindly|make sure|ensure|do|don't|avoid|use|add|remove|fix|update)\b",
    ])
    clf.add_class("statement", [
        r"\.$", r"\bI (think|believe|know|want|need)\b",
    ])
    return clf


__all__ = [
    "ResponseClassifier", "ClassifyResult",
    "yesno_classifier", "sentiment_classifier", "intent_classifier",
]
