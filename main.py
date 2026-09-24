"""Interactive command-line entry point: trains all classifiers and lets
the user classify typed utterances until they type 'stop'."""

import argparse

from classifiers import Classifier
from data import create_grouped_split, create_stratified_split, load_data, train_all


# 1. The interactive prompt

def run_prompt(trained_classifiers: dict[str, Classifier], model_name: str, verbose: bool) -> None:
  """Repeatedly ask for an utterance and print the predicted dialog act."""
  classifier = trained_classifiers[model_name]
  print(f"Using model: {model_name}")
  print("Type an utterance to classify it, or 'stop' to quit.")

  while True:
    inp = input("> ").strip().lower()
    if inp == "stop":
      return

    if verbose:
      # Show every classifier's prediction, so you can compare them.
      for name, other_classifier in trained_classifiers.items():
        print(f"  {name}: {other_classifier.run(inp)}")
    else:
      print(f"  {classifier.run(inp)}")


# 2. Entry point: parse args, load data, train, run the prompt

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--verbose", action="store_true", help="show every classifier's prediction instead of only one")
  parser.add_argument("--model", default="mlp", help="which classifier to use when --verbose is off (default: mlp)")
  args = parser.parse_args()

  data = load_data()
  train_data, test_data = create_stratified_split(data)
  clean_train_data, clean_test_data = create_grouped_split(data)

  trained_classifiers = train_all(train_data, clean_train_data)
  run_prompt(trained_classifiers, args.model, args.verbose)
