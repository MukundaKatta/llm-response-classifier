# llm-response-classifier

Classify LLM response text into labeled categories using regex rule patterns.

## Install

```
pip install llm-response-classifier
```

## Usage

```python
from llm_response_classifier import ResponseClassifier, yesno_classifier, sentiment_classifier

# Custom classifier
clf = ResponseClassifier()
clf.add_class("affirm", patterns=[r"\byes\b", r"\bsure\b", r"\babsolutely\b"])
clf.add_class("deny",   patterns=[r"\bno\b",  r"\bnever\b"])

result = clf.classify("Yes, I can help with that!")
print(result.label, result.confidence, result.matched_pattern)

# Built-in classifiers
yn = yesno_classifier()
print(yn.classify("No, I cannot do that.").label)  # "no"

sm = sentiment_classifier()
print(sm.classify("This is amazing!").label)  # "positive"
```
