"""Loading, splitting, and training utilities for the dialog act dataset.

Only imports from classifiers.py, never from evaluate.py or main.py, so
both of those can import this module freely without a circular import.
"""

import re

import gdown
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from classifiers import (
  Classifier,
  EmbeddedLRClassifier,
  EmbeddedMLPClassifier,
  FrozenEmbeddingEncoder,
  LRClassifier,
  MLPClassifier,
  RuleClassifier,
)

DATA_PATH = "dialog_acts.dat"
DATA_URL = "https://drive.google.com/uc?id=16GosT_JsqyijmdD6h6JdHH2e6i2GdCAk"


def load_data(path=DATA_PATH, download=True) -> pd.DataFrame:
  """Load and clean a dialog act .dat file (downloading it first if needed)."""
  if download:
    gdown.download(DATA_URL, path, quiet=False)

  raw_data = []
  with open(path) as f:
    for line in f.readlines():
      line = line.strip().lower()
      line = re.sub(r'[^a-zA-Z0-9\s]+', '', line)
      raw_data.append(line.split(maxsplit=1))

  data = pd.DataFrame(np.array(raw_data))
  return data.set_axis(["act", "utterance"], axis=1)


def save_data(path, data) -> None:
  """Write a (act, utterance) dataframe to disk in the original .dat format."""
  with open(path, "w") as f:
    for row in data.itertuples():
      f.write(f"{row[1]} {row[2]}\n")


def create_stratified_split(data, test_size=0.15):
  """Original 85/15 split, stratified by dialog act."""
  train_data, test_data = train_test_split(
    data, test_size=test_size, stratify=data['act'], random_state=7
  )
  save_data("dialog_acts_train.dat", train_data)
  save_data("dialog_acts_test.dat", test_data)
  return train_data, test_data


def create_grouped_split(data, test_size=0.15):
  """85/15 split that also keeps duplicate utterances in the same split,
  so identical utterances never leak between train and test."""
  acts = set(data['act'])
  train = {'act': [], 'utterance': []}
  test = {'act': [], 'utterance': []}
  acts_train = {act: 0 for act in acts}
  acts_test = {act: 0 for act in acts}

  grouped = data.groupby(['utterance', 'act']).size().sort_values(ascending=False)
  for (utterance, act), count in grouped.items():
    train_size, test_size_so_far = acts_train[act], acts_test[act]
    if train_size == 0 or test_size_so_far / (train_size + test_size_so_far) > test_size:
      train['utterance'] += [utterance] * count
      train['act'] += [act] * count
      acts_train[act] += count
    else:
      test['utterance'] += [utterance] * count
      test['act'] += [act] * count
      acts_test[act] += count

  return pd.DataFrame.from_dict(train), pd.DataFrame.from_dict(test)


def train_all(train_data, clean_train_data) -> dict[str, Classifier]:
  """Train every classifier on both splits and return them in a dict."""
  embedder = FrozenEmbeddingEncoder()

  classifiers = {
    "rule": RuleClassifier(),
    "lr": LRClassifier(),
    "clean_lr": LRClassifier(),
    "mlp": MLPClassifier(),
    "clean_mlp": MLPClassifier(),
    "embedded_lr": EmbeddedLRClassifier(embedder),
    "clean_embedded_lr": EmbeddedLRClassifier(embedder),
    "embedded_mlp": EmbeddedMLPClassifier(embedder),
    "clean_embedded_mlp": EmbeddedMLPClassifier(embedder),
  }

  classifiers["lr"].fit(train_data['utterance'], train_data['act'])
  classifiers["clean_lr"].fit(clean_train_data['utterance'], clean_train_data['act'])

  classifiers["mlp"].fit(train_data['utterance'], train_data['act'])
  classifiers["clean_mlp"].fit(clean_train_data['utterance'], clean_train_data['act'])

  classifiers["embedded_lr"].fit(train_data['utterance'], train_data['act'])
  classifiers["clean_embedded_lr"].fit(clean_train_data['utterance'], clean_train_data['act'])

  classifiers["embedded_mlp"].fit(train_data['utterance'], train_data['act'])
  classifiers["clean_embedded_mlp"].fit(clean_train_data['utterance'], clean_train_data['act'])

  return classifiers
