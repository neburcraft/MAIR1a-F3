"""Dialog act classifiers for the restaurant recommendation dialog system."""

from enum import Enum
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAMES = [
  "rule", "lr", "clean_lr", "mlp", "clean_mlp",
  "embedded_lr", "clean_embedded_lr", "embedded_mlp", "clean_embedded_mlp",
]

class Act(Enum):
  NULL = "null"
  ACK = "ack"
  AFFIRM = "affirm"
  BYE = "bye"
  CONFIRM = "confirm"
  DENY = "deny"
  HELLO = "hello"
  NEGATE = "negate"
  REPEAT = "repeat"
  REQALTS = "reqalts"
  REQMORE = "reqmore"
  REQUEST = "request"
  RESTART = "restart"
  THANKYOU = "thankyou"
  INFORM = "inform"


class Classifier:
  """Common interface every dialog act classifier implements"""
  def run(self, msg) -> Act:
    return Act.NULL
  
  def __str__(self) -> str:
    return "BaseClassifier"


class RuleClassifier(Classifier):
  """Keyword-based baseline: matches chosen keywords per dialog act"""
  
  KEYWORDS: dict[Act, list[str]] = {
    Act.ACK: ["okay", "im good", "thatll do", "good", "kay", "ok", "fine", "sure", "alright"],
    Act.AFFIRM: ["yes", "right", "correct", "indeed", "true", "yeah", "perfect", "excellent"],
    Act.BYE: ["goodbye", "good bye", "bye", "thats all", "done", "finished"],
    Act.CONFIRM: ["is it", "is that", "are they", "does it", "does that", "do they", "has it", "has that", "have they"],
    Act.DENY: ["dont", "incorrect", "false", "not", "wrong"],
    Act.HELLO: ["hi", "hello", "halo", "good morning", "goodmorning", "good afternoon", "goodafternoon", "good evening", "goodevening"],
    Act.NEGATE: ["no"],
    Act.REPEAT: ["back", "repeat", "again"],
    Act.REQALTS: ["anything else", "anything different", "what about", "how about", "instead"],
    Act.REQMORE: ["more"],
    Act.REQUEST: ["whats", "what is", "could", "can", "what", "address", "phone number", "price", "type", "area", "neighborhood", "post code", "postal code"],
    Act.RESTART: ["reset", "start over", "start again"],
    Act.THANKYOU: ["thank you", "thanks", "thank"],
    Act.NULL: ["uh", "unintelligible", "um", "cough", "noise", "silence", "laughter", "oh", "sil", "sorry", "breathing", "tvnoise"]
  }

  def run(self, msg) -> Act:
    # Acts sorted in order of specificity
    for act in [Act.RESTART, Act.REPEAT, Act.REQALTS, Act.REQUEST, Act.REQMORE, Act.CONFIRM, Act.THANKYOU, Act.HELLO, Act.BYE, Act.AFFIRM, Act.DENY, Act.ACK, Act.NEGATE]:
      # Looking for separate words, so preceded by space or nothing and ending in space or nothing (so 'noise' is not classified as 'no')
      if any(re.search(f"(^|\\s){kw}(\\s|$)", msg) for kw in self.KEYWORDS[act]):
        return act

    # The 'null' keywords are exact message matches, so they don't need re.search
    if msg in self.KEYWORDS[Act.NULL]:
      return Act.NULL

    return Act.INFORM
  
  def __str__(self) -> str:
    return "RuleClassifier"


class LRClassifier(Classifier):
  """Logistic regression on bag-of-words features"""
  
  # Adaptation from https://gist.github.com/sebleier/554280
  STOP_WORDS = [
      "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
      "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
      "hers", "herself", "it", "itself", "they", "them", "their", "theirs", "themselves",
      "who", "whom", "this", "that", "these", "those", "am", "was",
      "were", "be", "been", "being", "has", "had", "having", "do", "does", "did", "doing"]

  vectorizer: CountVectorizer
  vocabulary: set[str]
  model: LogisticRegression

  def __init__(self):
    self.vectorizer = CountVectorizer(stop_words=self.STOP_WORDS)
    self.model = LogisticRegression(random_state=7)

  def _clean_msg(self, msg) -> str:
    return " ".join(word if word in self.vocabulary else "OOV" for word in msg.split())

  def fit(self, X_train, y_train, verbose=False):
    if verbose: print("Initializing LRClassifier...")
    self.vocabulary = set()
    for utterance in X_train:
      self.vocabulary = self.vocabulary.union(set(utterance.split()))

    X = self.vectorizer.fit_transform(np.concatenate((["OOV"], X_train)))
    self.model.fit(X, np.concatenate((["null"], y_train)))
    return self

  def run(self, msg) -> Act:
    vec = self.vectorizer.transform([self._clean_msg(msg)])
    return Act[self.model.predict(vec)[0].upper()]


class MLPClassifier(Classifier):
  """Multi-layer perceptron on bag-of-words features"""
  
  vectorizer: CountVectorizer
  vocabulary: set[str]
  model: SklearnMLPClassifier

  def __init__(self):
    self.vectorizer = CountVectorizer(stop_words=LRClassifier.STOP_WORDS)
    self.model = SklearnMLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=7)

  def _clean_msg(self, msg) -> str:
    """Replace words unseen during training with an 'OOV' placeholder"""
    return " ".join(word if word in self.vocabulary else "OOV" for word in msg.split())

  def fit(self, X_train, y_train, verbose=False):
    """Train the classifier on a list of utterances and their acts"""
    if verbose: print("Initializing MLPClassifier...")
    self.vocabulary = set()
    for utterance in X_train:
      self.vocabulary = self.vocabulary.union(set(utterance.split()))

    X = self.vectorizer.fit_transform(np.concatenate((["OOV"], X_train)))
    self.model.fit(X, np.concatenate((["null"], y_train)))
    return self

  def run(self, msg) -> Act:
    vec = self.vectorizer.transform([self._clean_msg(msg)])
    return Act[self.model.predict(vec)[0].upper()]
  

class FrozenEmbeddingEncoder:
  """Batched DistilBERT encoder, shared across classifiers with a cache
     Utterances are only ever encoded once. repeated calls for an already seen utterance
     are served from text_cache instead of re-running the model"""

  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    self.encoder = AutoModel.from_pretrained("distilbert-base-uncased")
    self.encoder.to(self.device)
    self.encoder.eval()

    self.text_cache = {}

  def encode(self, utterances, batch_size=64, description=None):
    #Return one embedding vector per utterance in the given order
    utterances = list(utterances)

    new_utterances = list(dict.fromkeys(u for u in utterances if u not in self.text_cache))

    with torch.no_grad():
      for i in range(0, len(new_utterances), batch_size):
        batch = new_utterances[i:i + batch_size]

        tokens = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        for key in tokens:
          tokens[key] = tokens[key].to(self.device)

        output = self.encoder(**tokens).last_hidden_state

        #Masked mean pooling. average token vectors, while ignoring padding.
        mask = tokens["attention_mask"].unsqueeze(-1)
        summed = (output * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        embeddings = summed / counts

        embeddings = embeddings.cpu().numpy()

        for utterance, embedding in zip(batch, embeddings):
          self.text_cache[utterance] = embedding

    return np.array([self.text_cache[u] for u in utterances])


class EmbeddedLRClassifier(Classifier):
  """Logistic regression on frozen DistilBERT embeddings"""

  def __init__(self, embedder: FrozenEmbeddingEncoder):
    self.embedder = embedder
    self.model = LogisticRegression(random_state=7, max_iter=1000)

  def fit(self, X_train, y_train, verbose=False):
    if verbose: print("Initializing EmbeddedLRClassifier...")
    #Train the classifier on a list of utterances and their acts
    embeddings = self.embedder.encode(X_train)
    self.model.fit(embeddings, y_train)
    return self

  def predict(self, utterances):
    #Predict dialog acts for a batch of utterances at once
    embeddings = self.embedder.encode(utterances)
    return self.model.predict(embeddings)

  def run(self, msg) -> Act:
    return Act[self.predict([msg])[0].upper()]


class EmbeddedMLPClassifier(Classifier):
  """Multi-layer perceptron on frozen DistilBERT embeddings"""

  def __init__(self, embedder: FrozenEmbeddingEncoder):
    self.embedder = embedder
    self.model = SklearnMLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=7)

  def fit(self, X_train, y_train, verbose=False):
    if verbose: print("Initializing EmbeddedMLPClassifier...")
    #Train the classifier on a list of utterances and their acts
    embeddings = self.embedder.encode(X_train)
    self.model.fit(embeddings, y_train)
    return self

  def predict(self, utterances):
    #Predict dialog acts for a batch of utterances at once
    embeddings = self.embedder.encode(utterances)
    return self.model.predict(embeddings)

  def run(self, msg) -> Act:
    return Act[self.predict([msg])[0].upper()]


def accuracy(classifier: Classifier, X_test, y_test, show_incorrect=0) -> float:
  """Measure accuracy of the given classifier in the usual way:
     accuracy = [amount correct] / [test set size]"""
  total = y_test.size
  correct = 0
  for x,y in zip(X_test, y_test):
    pred = classifier.run(x)
    if pred == y:
      correct += 1
    elif show_incorrect > 0:
      print(f"INCORRECT ({pred} should be {y}): {x}")
      show_incorrect -= 1

  return correct / total


# If `$ python ./classifiers.py` is ran, all models are trained
# and a basic accuracy comparison is done across the models
# (only regular accuracy implemented for now)
if __name__ == "__main__":
  from data import load_data, create_stratified_split, create_grouped_split

  data = load_data()
  train_data, test_data = create_stratified_split(data)
  clean_train_data, clean_test_data = create_grouped_split(data)

  frozen_embedding_encoder = FrozenEmbeddingEncoder()

  # 1 - Rule-based classifier
  rule_classifier = RuleClassifier()

  # 2a - Logistic regression Bag-of-Words classifier
  lr_classifier = LRClassifier()
  lr_classifier.fit(train_data["utterance"], train_data["act"])
  clean_lr_classifier = LRClassifier()
  clean_lr_classifier.fit(clean_train_data["utterance"], clean_train_data["act"])

  # 2b - Multi Layer Perceptron Bag-of-Words classifier
  mlp_classifier = MLPClassifier()
  mlp_classifier.fit(train_data["utterance"], train_data["act"])
  clean_mlp_classifier = MLPClassifier()
  clean_mlp_classifier.fit(clean_train_data["utterance"], clean_train_data["act"])

  # 3a - Logistic regression embedded word representations classifier
  embedded_lr_classifier = EmbeddedLRClassifier(frozen_embedding_encoder)
  embedded_lr_classifier.fit(train_data["utterance"], train_data["act"])
  clean_embedded_lr_classifier = EmbeddedLRClassifier(frozen_embedding_encoder)
  clean_embedded_lr_classifier.fit(clean_train_data["utterance"], clean_train_data["act"])

  # 3b - Multi Layer Perceptron embedded word representations classifier
  embedded_mlp_classifier = EmbeddedMLPClassifier(frozen_embedding_encoder)
  embedded_mlp_classifier.fit(train_data["utterance"], train_data["act"])
  clean_embedded_mlp_classifier = EmbeddedMLPClassifier(frozen_embedding_encoder)
  clean_embedded_mlp_classifier.fit(clean_train_data["utterance"], clean_train_data["act"])

  frozen_embedding_encoder.encode(test_data["utterance"])
  frozen_embedding_encoder.encode(clean_test_data["utterance"])

  # Measure and print all accuracy scores
  print("Rule baseline:", accuracy(rule_classifier, test_data["utterance"], test_data["act"]))
  print("BoW LR original:", accuracy(lr_classifier, test_data["utterance"], test_data["act"]))
  print("BoW LR grouped:", accuracy(clean_lr_classifier, clean_test_data["utterance"], clean_test_data["act"]))
  print("BoW MLP original:", accuracy(mlp_classifier, test_data["utterance"], test_data["act"]))
  print("BoW MLP grouped:", accuracy(clean_mlp_classifier, clean_test_data["utterance"], clean_test_data["act"]))
  print("Frozen LR original:", accuracy(embedded_lr_classifier, test_data["utterance"], test_data["act"]))
  print("Frozen LR grouped:", accuracy(clean_embedded_lr_classifier, clean_test_data["utterance"], clean_test_data["act"]))
  print("Frozen MLP original:", accuracy(embedded_mlp_classifier, test_data["utterance"], test_data["act"]))
  print("Frozen MLP grouped:", accuracy(clean_embedded_mlp_classifier, clean_test_data["utterance"], clean_test_data["act"]))
