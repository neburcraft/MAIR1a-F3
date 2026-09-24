"""Loading, cleaning, and splitting the dialog act dataset."""

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "dialog_acts.dat"


#  Cleaning a single utterance

def clean_utterance(text: str) -> str:
  """Lowercase and strip an utterance of all punctuation."""
  text = text.strip().lower()
  return re.sub(r'[^a-zA-Z0-9\s]+', '', text)


#  Loading the raw data

def load_data(path=DATA_PATH) -> pd.DataFrame:
  """Load and clean a dialog act .dat file that is already present on disk,
  e.g. dialog_acts.dat, which is committed to the repository."""
  raw_data = []
  with open(path) as f:
    for line in f.readlines():
      line = clean_utterance(line)
      raw_data.append(line.split(maxsplit=1))

  data = pd.DataFrame(np.array(raw_data))
  return data.set_axis(["act", "utterance"], axis=1)


def save_data(path, data) -> None:
  """Write a (act, utterance) dataframe to disk in the original .dat format."""
  with open(path, "w") as f:
    for row in data.itertuples():
      f.write(f"{row[1]} {row[2]}\n")


#  Original stratified split

def create_stratified_split(data, test_size=0.15):
  """Original 85/15 split, stratified by dialog act."""
  train_data, test_data = train_test_split(
    data, test_size=test_size, stratify=data['act'], random_state=7
  )
  save_data("dialog_acts_train.dat", train_data)
  save_data("dialog_acts_test.dat", test_data)
  return train_data, test_data


#  Grouped split, avoids data leakage from duplicate utterances

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
