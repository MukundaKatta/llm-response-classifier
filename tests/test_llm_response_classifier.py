"""Unit tests for llm_response_classifier (standard-library unittest only).

Run from the repository root with::

    python3 -m unittest discover -s tests
"""
import os
import sys
import unittest

# Make the src/ layout package importable when running without an install.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llm_response_classifier import (  # noqa: E402  (import after sys.path tweak)
    ClassifyResult,
    ResponseClassifier,
    intent_classifier,
    sentiment_classifier,
    yesno_classifier,
)


def make_clf() -> ResponseClassifier:
    clf = ResponseClassifier()
    clf.add_class("yes", [r"\byes\b", r"\bsure\b", r"\babsolutely\b"])
    clf.add_class("no", [r"\bno\b", r"\bnever\b", r"\bnot\b"])
    return clf


class CoreClassifierTests(unittest.TestCase):
    def test_basic_match(self) -> None:
        r = make_clf().classify("Yes, I can help!")
        self.assertEqual(r.label, "yes")
        self.assertGreater(r.confidence, 0)

    def test_returns_classify_result_type(self) -> None:
        r = make_clf().classify("yes")
        self.assertIsInstance(r, ClassifyResult)

    def test_no_match_returns_default(self) -> None:
        r = make_clf().classify("blah blah blah")
        self.assertEqual(r.label, "unknown")
        self.assertEqual(r.confidence, 0.0)
        self.assertIsNone(r.matched_pattern)

    def test_custom_default_label(self) -> None:
        clf = ResponseClassifier(default_label="other")
        self.assertEqual(clf.classify("nothing matches").label, "other")

    def test_matched_pattern_populated(self) -> None:
        r = make_clf().classify("sure thing")
        self.assertIsNotNone(r.matched_pattern)
        self.assertEqual(r.matched_pattern, r"\bsure\b")

    def test_scores_contains_all_classes(self) -> None:
        r = make_clf().classify("yes")
        self.assertIn("yes", r.scores)
        self.assertIn("no", r.scores)

    def test_case_insensitive(self) -> None:
        self.assertEqual(make_clf().classify("YES I WILL").label, "yes")

    def test_is_uncertain_true(self) -> None:
        self.assertTrue(make_clf().classify("blah").is_uncertain)

    def test_is_uncertain_false(self) -> None:
        # Only the "yes" class matches, so confidence is 1.0 (not uncertain).
        self.assertFalse(make_clf().classify("yes absolutely sure!").is_uncertain)

    def test_empty_classifier(self) -> None:
        clf = ResponseClassifier()
        r = clf.classify("anything")
        self.assertEqual(r.label, "unknown")
        self.assertEqual(r.scores, {})

    def test_classify_many(self) -> None:
        results = make_clf().classify_many(["yes", "no", "blah"])
        self.assertEqual(results[0].label, "yes")
        self.assertEqual(results[1].label, "no")
        self.assertEqual(results[2].label, "unknown")

    def test_classify_many_empty_list(self) -> None:
        self.assertEqual(make_clf().classify_many([]), [])

    def test_labels_property(self) -> None:
        self.assertEqual(set(make_clf().labels), {"yes", "no"})

    def test_labels_preserve_order(self) -> None:
        self.assertEqual(make_clf().labels, ["yes", "no"])

    def test_add_class_returns_self(self) -> None:
        clf = ResponseClassifier()
        self.assertIs(clf.add_class("x", ["pattern"]), clf)

    def test_add_class_is_chainable(self) -> None:
        clf = (
            ResponseClassifier()
            .add_class("a", [r"a"])
            .add_class("b", [r"b"])
        )
        self.assertEqual(clf.labels, ["a", "b"])

    def test_weight_affects_score(self) -> None:
        clf = ResponseClassifier()
        clf.add_class("high", [r"\bword\b"], weight=10.0)
        clf.add_class("low", [r"\bword\b"], weight=1.0)
        self.assertEqual(clf.classify("word").label, "high")

    def test_confidence_is_one_when_single_class_matches(self) -> None:
        clf = ResponseClassifier()
        clf.add_class("a", [r"x"])
        clf.add_class("b", [r"y"])
        self.assertEqual(clf.classify("x").confidence, 1.0)

    def test_confidence_split_when_two_classes_match(self) -> None:
        clf = ResponseClassifier()
        clf.add_class("a", [r"x"])
        clf.add_class("b", [r"y"])
        r = clf.classify("x y")
        self.assertAlmostEqual(r.confidence, 0.5)


class ValidationAndHardeningTests(unittest.TestCase):
    def test_none_text_treated_as_empty(self) -> None:
        r = yesno_classifier().classify(None)  # type: ignore[arg-type]
        self.assertEqual(r.label, "unknown")
        self.assertEqual(r.confidence, 0.0)

    def test_empty_string(self) -> None:
        self.assertEqual(yesno_classifier().classify("").label, "unknown")

    def test_empty_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResponseClassifier().add_class("", ["p"])

    def test_zero_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResponseClassifier().add_class("x", ["p"], weight=0)

    def test_negative_weight_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ResponseClassifier().add_class("x", ["p"], weight=-1.0)

    def test_invalid_regex_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            ResponseClassifier().add_class("bad", ["(unclosed"])

    def test_empty_patterns_never_matches(self) -> None:
        clf = ResponseClassifier()
        clf.add_class("empty", [])
        r = clf.classify("anything at all")
        self.assertEqual(r.label, "unknown")
        self.assertEqual(r.scores["empty"], 0.0)


class BuiltinClassifierTests(unittest.TestCase):
    def test_yesno_yes(self) -> None:
        self.assertEqual(yesno_classifier().classify("Yes, of course!").label, "yes")

    def test_yesno_no(self) -> None:
        self.assertEqual(
            yesno_classifier().classify("No, I cannot do that.").label, "no"
        )

    def test_yesno_default_label(self) -> None:
        self.assertEqual(yesno_classifier().classify("xyzzy").label, "unknown")

    def test_sentiment_positive(self) -> None:
        r = sentiment_classifier().classify("That is absolutely amazing and wonderful!")
        self.assertEqual(r.label, "positive")

    def test_sentiment_negative(self) -> None:
        r = sentiment_classifier().classify("This is terrible and horrible.")
        self.assertEqual(r.label, "negative")

    def test_sentiment_default_neutral(self) -> None:
        self.assertEqual(sentiment_classifier().classify("xyzzy").label, "neutral")

    def test_intent_question(self) -> None:
        r = intent_classifier().classify("What is the capital of France?")
        self.assertEqual(r.label, "question")

    def test_intent_instruction(self) -> None:
        r = intent_classifier().classify("Please make sure to include tests.")
        self.assertEqual(r.label, "instruction")

    def test_intent_default_statement(self) -> None:
        self.assertEqual(intent_classifier().classify("xyzzy").label, "statement")

    def test_builtin_labels(self) -> None:
        self.assertEqual(set(yesno_classifier().labels), {"yes", "no"})
        self.assertEqual(
            set(sentiment_classifier().labels), {"positive", "negative", "neutral"}
        )


if __name__ == "__main__":
    unittest.main()
