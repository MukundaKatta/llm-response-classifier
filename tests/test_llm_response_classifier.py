from llm_response_classifier import (
    ResponseClassifier,
    yesno_classifier,
    sentiment_classifier,
    intent_classifier,
)


def make_clf():
    clf = ResponseClassifier()
    clf.add_class("yes", [r"\byes\b", r"\bsure\b", r"\babsolutely\b"])
    clf.add_class("no", [r"\bno\b", r"\bnever\b", r"\bnot\b"])
    return clf


def test_basic_match():
    clf = make_clf()
    r = clf.classify("Yes, I can help!")
    assert r.label == "yes"
    assert r.confidence > 0


def test_no_match_returns_default():
    clf = make_clf()
    r = clf.classify("blah blah blah")
    assert r.label == "unknown"
    assert r.confidence == 0.0
    assert r.matched_pattern is None


def test_custom_default_label():
    clf = ResponseClassifier(default_label="other")
    r = clf.classify("nothing matches")
    assert r.label == "other"


def test_matched_pattern_populated():
    clf = make_clf()
    r = clf.classify("sure thing")
    assert r.matched_pattern is not None


def test_scores_contains_all_classes():
    clf = make_clf()
    r = clf.classify("yes")
    assert "yes" in r.scores
    assert "no" in r.scores


def test_case_insensitive():
    clf = make_clf()
    r = clf.classify("YES I WILL")
    assert r.label == "yes"


def test_is_uncertain_true():
    clf = make_clf()
    r = clf.classify("blah")
    assert r.is_uncertain


def test_is_uncertain_false():
    clf = make_clf()
    r = clf.classify("yes absolutely sure!")
    # multiple patterns match -> high confidence
    assert not r.is_uncertain


def test_empty_classifier():
    clf = ResponseClassifier()
    r = clf.classify("anything")
    assert r.label == "unknown"
    assert r.scores == {}


def test_classify_many():
    clf = make_clf()
    results = clf.classify_many(["yes", "no", "blah"])
    assert results[0].label == "yes"
    assert results[1].label == "no"
    assert results[2].label == "unknown"


def test_labels_property():
    clf = make_clf()
    assert set(clf.labels) == {"yes", "no"}


def test_add_class_returns_self():
    clf = ResponseClassifier()
    ret = clf.add_class("x", ["pattern"])
    assert ret is clf


def test_weight_affects_score():
    clf = ResponseClassifier()
    clf.add_class("high", [r"\bword\b"], weight=10.0)
    clf.add_class("low", [r"\bword\b"], weight=1.0)
    r = clf.classify("word")
    assert r.label == "high"


def test_negative_weight_only_falls_back_to_default():
    # A class whose only match yields a non-positive score must not be reported
    # as a confident result; it should fall back to the default label.
    clf = ResponseClassifier(default_label="unknown")
    clf.add_class("penalized", [r"\bx\b"], weight=-1.0)
    r = clf.classify("x")
    assert r.label == "unknown"
    assert r.confidence == 0.0
    assert r.matched_pattern is None


def test_confidence_stays_within_unit_range_with_negative_weights():
    # Mixing positive and negative weights must keep confidence in [0.0, 1.0].
    clf = ResponseClassifier()
    clf.add_class("pos", [r"good"], weight=1.0)
    clf.add_class("neg", [r"bad"], weight=-5.0)
    r = clf.classify("good bad")
    assert r.label == "pos"
    assert 0.0 <= r.confidence <= 1.0


def test_confidence_never_negative():
    clf = make_clf()
    for text in ["yes", "no", "blah", "yes no never", ""]:
        r = clf.classify(text)
        assert 0.0 <= r.confidence <= 1.0


# -- built-in classifiers --------------------------------------------------


def test_yesno_yes():
    clf = yesno_classifier()
    r = clf.classify("Yes, of course!")
    assert r.label == "yes"


def test_yesno_no():
    clf = yesno_classifier()
    r = clf.classify("No, I cannot do that.")
    assert r.label == "no"


def test_sentiment_positive():
    clf = sentiment_classifier()
    r = clf.classify("That is absolutely amazing and wonderful!")
    assert r.label == "positive"


def test_sentiment_negative():
    clf = sentiment_classifier()
    r = clf.classify("This is terrible and horrible.")
    assert r.label == "negative"


def test_intent_question():
    clf = intent_classifier()
    r = clf.classify("What is the capital of France?")
    assert r.label == "question"


def test_intent_instruction():
    clf = intent_classifier()
    r = clf.classify("Please make sure to include tests.")
    assert r.label == "instruction"
