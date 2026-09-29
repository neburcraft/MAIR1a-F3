"""Interactive command-line entry point: trains all classifiers and lets
the user classify typed utterances until they type 'stop'. """

import argparse

from classifiers import MODEL_NAMES, Classifier, EmbeddedLRClassifier, EmbeddedMLPClassifier, FrozenEmbeddingEncoder, LRClassifier, MLPClassifier, RuleClassifier
from cli import run_prompt
from data import create_grouped_split, create_stratified_split, load_data


def train_all(train_data, clean_train_data, verbose=False) -> dict[str, Classifier]:
  """Train every classifier on both splits and return them in a dict.

  Relies on fit() returning self, so each classifier can be created and
  trained in one expression instead of two separate steps.
  """
  embedder = FrozenEmbeddingEncoder()
  utterance, act = train_data['utterance'], train_data['act']
  clean_utterance_col, clean_act = clean_train_data['utterance'], clean_train_data['act']

  return {
    "rule": RuleClassifier(),
    "lr": LRClassifier().fit(utterance, act, verbose),
    "clean_lr": LRClassifier().fit(clean_utterance_col, clean_act, verbose),
    "mlp": MLPClassifier().fit(utterance, act, verbose),
    "clean_mlp": MLPClassifier().fit(clean_utterance_col, clean_act, verbose),
    "embedded_lr": EmbeddedLRClassifier(embedder).fit(utterance, act, verbose),
    "clean_embedded_lr": EmbeddedLRClassifier(embedder).fit(clean_utterance_col, clean_act, verbose),
    "embedded_mlp": EmbeddedMLPClassifier(embedder).fit(utterance, act, verbose),
    "clean_embedded_mlp": EmbeddedMLPClassifier(embedder).fit(clean_utterance_col, clean_act, verbose),
  }


#  Entry point when ran as `$ python ./main.py`: parse args, load data, train, run the prompt
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--verbose", action="store_true", help="show every classifier's prediction instead of only one")
  parser.add_argument(
    "--model", default="mlp", choices=MODEL_NAMES,
    help="which classifier to use when --verbose is off (default: mlp); can also be changed at runtime with 'model <name>'",
  )
  args = parser.parse_args()

  data = load_data()
  train_data, test_data = create_stratified_split(data)
  clean_train_data, clean_test_data = create_grouped_split(data)

  if args.verbose: print("Starting training of the models")
  trained_classifiers = train_all(train_data, clean_train_data, verbose=args.verbose)
  run_prompt(trained_classifiers, args.model, args.verbose)
