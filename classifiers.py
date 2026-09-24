"""Dialog act classifiers for the restaurant recommendation dialog system.""" 

import re
from typing import override

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier


class Classifier:
  """Common interface every dialog act classifier implements."""
  def run(self, msg) -> str:
    return "none"
  
  @override
  def __str__(self) -> str:
    return "BaseClassifier"


class RuleClassifier(Classifier):
  """Keyword-based baseline: matches chosen keywords per dialog act."""
  
  KEYWORDS: dict[str, list[str]] = {
    "ack": ["okay", "im good", "thatll do", "good", "kay", "ok", "fine", "sure", "alright"],
    "affirm": ["yes", "right", "correct", "indeed", "true", "yeah", "perfect", "excellent"],
    "bye": ["goodbye", "good bye", "bye", "thats all", "done", "finished"],
    "confirm": ["is it", "is that", "are they", "does it", "does that", "do they", "has it", "has that", "have they"],
    "deny": ["dont", "incorrect", "false", "not", "wrong"],
    "hello": ["hi", "hello", "halo", "good morning", "goodmorning", "good afternoon", "goodafternoon", "good evening", "goodevening"],
    "negate": ["no"],
    "repeat": ["back", "repeat", "again"],
    "reqalts": ["anything else", "anything different", "what about", "how about", "instead"],
    "reqmore": ["more"],
    "request": ["whats", "what is", "could", "can", "what", "address", "phone number", "price", "type", "area", "neighborhood", "post code", "postal code"],
    "restart": ["reset", "start over", "start again"],
    "thankyou": ["thank you", "thanks", "thank"],
    "null": ["uh", "unintelligible", "um", "cough", "noise", "silence", "laughter", "oh", "sil", "sorry", "breathing", "tvnoise"]
  }

  @override
  def run(self, msg) -> str:
    # Acts sorted in order of specificity
    for act in ["reqalts", "request", "reqmore", "confirm", "repeat", "thankyou", "hello", "bye", "affirm", "deny", "ack", "negate"]:
      # Looking for separate words, so preceded by space or nothing and ending in space or nothing (so 'noise' is not classified as 'no')
      if any(re.search(f"(^|\\s){kw}(\\s|$)", msg) for kw in self.KEYWORDS[act]):
        return act

    # The 'null' keywords are exact message matches, so they don't need re.search
    if msg in self.KEYWORDS["null"]:
      return "null"

    return "inform"
  
  @override
  def __str__(self) -> str:
    return "RuleClassifier"


class LRClassifier(Classifier):
  """Logistic regression on bag-of-words features."""
  
  # Adaptation from https://gist.github.com/sebleier/554280
  STOP_WORDS = [
      "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
      "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
      "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
      "who", "whom", "this", "that", "these", "those", "am", "was",
      "were", "be", "been", "being", "has", "had", "having", "do", "does", "did", "doing",
      "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
      "by", "for", "with", "about", "against", "between", "into", "through",
      "for", "above", "below", "to", "from", "on", "off", "over",
      "under", "then", "once", "here", "there",
      "all", "both", "each", "few", "more", "most", "some", "such", "nor",
      "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
      "don", "should"]

  
  vectorizer: CountVectorizer
  vocabulary: set[str]
  model: LogisticRegression

  def __init__(self):
    self.vectorizer = CountVectorizer(stop_words=self.STOP_WORDS)
    self.model = LogisticRegression(random_state=7)

  def _clean_msg(self, msg) -> str:
    return " ".join(word if word in self.vocabulary else "OOV" for word in msg.split())

  def fit(self, X_train, y_train):
    self.vocabulary = set()
    for utterance in X_train:
      self.vocabulary = self.vocabulary.union(set(utterance.split()))

    X = self.vectorizer.fit_transform(np.concatenate((["OOV"], X_train)))
    self.model.fit(X, np.concatenate((["null"], y_train)))

  @override
  def run(self, msg) -> str:
    vec = self.vectorizer.transform([self._clean_msg(msg)])
    return self.model.predict(vec)[0]


class MLPClassifier(Classifier):
  """Multi-layer perceptron on bag-of-words features."""
  
  vectorizer: CountVectorizer
  vocabulary: set[str]
  model: SklearnMLPClassifier

  def __init__(self):
    self.vectorizer = CountVectorizer(stop_words=LRClassifier.STOP_WORDS)
    self.model = SklearnMLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=7)

  def _clean_msg(self, msg) -> str:
    """Replace words unseen during training with an 'OOV' placeholder."""
    return " ".join(word if word in self.vocabulary else "OOV" for word in msg.split())

  def fit(self, X_train, y_train):
    """Train the classifier on a list of utterances and their acts."""
    self.vocabulary = set()
    for utterance in X_train:
      self.vocabulary = self.vocabulary.union(set(utterance.split()))

    X = self.vectorizer.fit_transform(np.concatenate((["OOV"], X_train)))
    self.model.fit(X, np.concatenate((["null"], y_train)))

  @override
  def run(self, msg) -> str:
    vec = self.vectorizer.transform([self._clean_msg(msg)])
    return self.model.predict(vec)[0]
  
