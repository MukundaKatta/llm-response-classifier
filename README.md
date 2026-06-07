# llm-response-classifier

Lightweight, dependency-free classification of LLM (or any) response text into
labeled categories using case-insensitive regex rule patterns.

It is intentionally simple: there is no model to train and nothing to download.
You declare classes and their patterns, and each piece of text is scored against
every class. This makes it a good fit for routing, guardrails, intent detection,
and quick heuristics where a full ML classifier would be overkill.

## Features

- Pure Python, zero runtime dependencies (standard library only).
- Case-insensitive regex matching with optional per-class weights.
- A `confidence` score and an `is_uncertain` helper for thresholding.
- Batch classification via `classify_many`.
- Ready-made `yesno`, `sentiment`, and `intent` classifiers.
- Typed (`py.typed`) and validated inputs with clear error messages.

## Install

```
pip install llm-response-classifier
```

Or from a checkout of this repository:

```
pip install .
```

## Usage

```python
from llm_response_classifier import (
    ResponseClassifier,
    yesno_classifier,
    sentiment_classifier,
)

# Build a custom classifier.
clf = ResponseClassifier()
clf.add_class("affirm", patterns=[r"\byes\b", r"\bsure\b", r"\babsolutely\b"])
clf.add_class("deny",   patterns=[r"\bno\b",  r"\bnever\b"])

result = clf.classify("Yes, I can help with that!")
print(result.label)            # "affirm"
print(result.confidence)       # 1.0  (only one class matched)
print(result.matched_pattern)  # "\\byes\\b"
print(result.scores)           # {"affirm": 1.0, "deny": 0.0}

# Built-in classifiers.
yn = yesno_classifier()
print(yn.classify("No, I cannot do that.").label)   # "no"

sm = sentiment_classifier()
print(sm.classify("This is amazing!").label)         # "positive"
```

`add_class` returns the classifier, so calls can be chained:

```python
clf = (
    ResponseClassifier(default_label="unknown")
    .add_class("greeting", [r"\bhello\b", r"\bhi\b"])
    .add_class("farewell", [r"\bbye\b", r"\bgoodbye\b"])
)
```

### Weights

A weight multiplies a class's score, which is useful for breaking ties or
prioritizing a class:

```python
clf = ResponseClassifier()
clf.add_class("urgent", [r"\bnow\b"], weight=5.0)
clf.add_class("normal", [r"\bnow\b"], weight=1.0)
print(clf.classify("do it now").label)   # "urgent"
```

### Thresholding on confidence

```python
result = clf.classify(text)
if result.is_uncertain:          # confidence < 0.5
    handle_fallback(text)
else:
    route(result.label)
```

## API

### `ResponseClassifier(default_label="unknown")`

- `add_class(label, patterns, weight=1.0) -> ResponseClassifier`
  Register a class. `patterns` is a list of regex strings compiled with
  `re.IGNORECASE`; a class matches when **any** of its patterns match. Raises
  `ValueError` if `label` is empty, `weight` is not positive, or a pattern is
  not a valid regular expression. Returns `self` for chaining.
- `classify(text) -> ClassifyResult`
  Score `text` against every registered class and return the winner. `None` is
  treated as an empty string. With no registered classes, or when nothing
  matches, the result uses `default_label` with confidence `0.0`.
- `classify_many(texts) -> list[ClassifyResult]`
  Classify a list of texts, one result per input.
- `labels -> list[str]`
  Registered labels, in registration order.

### `ClassifyResult`

A dataclass with:

- `label: str` — the winning label (or `default_label` when nothing matched).
- `confidence: float` — the winning class's score divided by the sum of all
  class scores, in `[0.0, 1.0]`. `1.0` means a single class matched; `0.0`
  means nothing matched.
- `matched_pattern: str | None` — the first raw pattern that matched for the
  winning class.
- `scores: dict[str, float]` — the `match_fraction * weight` score for each
  class. `match_fraction` is the share of a class's patterns that matched.
- `is_uncertain: bool` (property) — `True` when `confidence < 0.5`.

### Built-in classifiers

- `yesno_classifier()` — labels `yes` / `no`, default `unknown`.
- `sentiment_classifier()` — labels `positive` / `negative` / `neutral`,
  default `neutral`.
- `intent_classifier()` — labels `question` / `instruction` / `statement`,
  default `statement`.

## Development

Run the test suite with the standard library only (no third-party deps
required):

```
python3 -m unittest discover -s tests
```

## License

MIT
