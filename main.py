"""Interactive command-line entry point: trains all classifiers and lets
the user classify typed utterances until they type 'stop'. """

import argparse

from classifiers import (
  Classifier,
  EmbeddedLRClassifier,
  EmbeddedMLPClassifier,
  FrozenEmbeddingEncoder,
  LRClassifier,
  MLPClassifier,
  RuleClassifier,
)
from data import clean_utterance, create_grouped_split, create_stratified_split, load_data


# Training all classifiers on both splits

def train_all(train_data, clean_train_data) -> dict[str, Classifier]:
  """Train every classifier on both splits and return them in a dict.

  Relies on fit() returning self, so each classifier can be created and
  trained in one expression instead of two separate steps.
  """
  embedder = FrozenEmbeddingEncoder()
  utterance, act = train_data['utterance'], train_data['act']
  clean_utterance_col, clean_act = clean_train_data['utterance'], clean_train_data['act']

  return {
    "rule": RuleClassifier(),
    "lr": LRClassifier().fit(utterance, act),
    "clean_lr": LRClassifier().fit(clean_utterance_col, clean_act),
    "mlp": MLPClassifier().fit(utterance, act),
    "clean_mlp": MLPClassifier().fit(clean_utterance_col, clean_act),
    "embedded_lr": EmbeddedLRClassifier(embedder).fit(utterance, act),
    "clean_embedded_lr": EmbeddedLRClassifier(embedder).fit(clean_utterance_col, clean_act),
    "embedded_mlp": EmbeddedMLPClassifier(embedder).fit(utterance, act),
    "clean_embedded_mlp": EmbeddedMLPClassifier(embedder).fit(clean_utterance_col, clean_act),
  }


MODEL_NAMES = [
  "rule", "lr", "clean_lr", "mlp", "clean_mlp",
  "embedded_lr", "clean_embedded_lr", "embedded_mlp", "clean_embedded_mlp",
]


# The interactive prompt

def run_prompt(trained_classifiers: dict[str, Classifier], model_name: str, verbose: bool) -> None:
  """Repeatedly ask for an utterance and print the predicted dialog act.

  Ask for utterances and print the predicted dialog act until the user types 'stop'. 
  Typing 'model <name>' switches to a different classifier
    
  """
  classifier = trained_classifiers[model_name]
  print(f"Using model: {model_name}")
  print("Type an utterance to classify it, 'model <name>' to switch models, or 'stop' to quit.")
  print(f"Available models: {', '.join(MODEL_NAMES)}")

  while True:
    raw_inp = input("> ").strip()

    if raw_inp.lower() == "stop":
      return

    if raw_inp.lower().startswith("model "):
      requested = raw_inp[len("model "):].strip()
      if requested in trained_classifiers:
        classifier = trained_classifiers[requested]
        print(f"Switched to model: {requested}")
      else:
        print(f"Unknown model '{requested}'. Available models: {', '.join(MODEL_NAMES)}")
      continue

    # Clean the same way the training data was cleaned, so live input
    # matches what the classifiers were trained on.
    inp = clean_utterance(raw_inp)

    if verbose:
      # Show every classifier's prediction, so you can compare them.
      for name, other_classifier in trained_classifiers.items():
        print(f"  {name}: {other_classifier.run(inp)}")
    else:
      print(f"  {classifier.run(inp)}")


#  Entry point: parse args, load data, train, run the prompt

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

  trained_classifiers = train_all(train_data, clean_train_data)
  run_prompt(trained_classifiers, args.model, args.verbose)
