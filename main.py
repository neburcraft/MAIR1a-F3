"""
Interactive command-line entry point with 3 modes:
Normal: Runs the full state manager from part 1b as defined in `manager.py`
--classify: Lets the user enter phrases to classify the speech-act for
--evaluate: Runs the automatic evaluation as defined in `evaluate.py`
"""

import argparse

from classifiers import (MODEL_NAMES, Classifier, EmbeddedLRClassifier, EmbeddedMLPClassifier,
                         FrozenEmbeddingEncoder, LRClassifier, MLPClassifier, RuleClassifier)
from data import clean_utterance, create_grouped_split, create_stratified_split, load_data
from slot_extraction import SlotExtract


class CLI:
  """The command-line interface for running the manager, classifier or evaluator"""

  def __init__(self, args: argparse.Namespace):
    self.model: str = args.model
    self.extractor: SlotExtract = SlotExtract(args.extractor)
    self.tts: bool = args.tts
    self.explain: bool = args.explain
    self.verbose: bool = args.verbose

    data = load_data()
    self.train_data, self.test_data = create_stratified_split(data)
    self.clean_train_data, self.clean_test_data = create_grouped_split(data)
    self.classifier = self.get_classifier(self.model)

  def get_classifier(self, name: str) -> Classifier:
    """Initializes and returns the classifier with the given `name`"""
    utterance, act = self.train_data['utterance'], self.train_data['act']
    clean_utterance_col = self.clean_train_data['utterance']
    clean_act = self.clean_train_data['act']

    match name:
      case "rule":
        return RuleClassifier()
      case "lr":
        return LRClassifier().fit(utterance, act, self.verbose)
      case "clean_lr":
        return LRClassifier().fit(clean_utterance_col, clean_act, self.verbose)
      case "mlp":
        return MLPClassifier().fit(utterance, act, self.verbose)
      case "clean_mlp":
        return MLPClassifier().fit(clean_utterance_col, clean_act, self.verbose)
      case "embedded_lr":
        embedder = FrozenEmbeddingEncoder()
        return EmbeddedLRClassifier(embedder).fit(utterance, act, self.verbose)
      case "clean_embedded_lr":
        embedder = FrozenEmbeddingEncoder()
        return EmbeddedLRClassifier(embedder).fit(clean_utterance_col, clean_act, self.verbose)
      case "embedded_mlp":
        embedder = FrozenEmbeddingEncoder()
        return EmbeddedMLPClassifier(embedder).fit(utterance, act, self.verbose)
      case "clean_embedded_mlp":
        embedder = FrozenEmbeddingEncoder()
        return EmbeddedMLPClassifier(embedder).fit(clean_utterance_col, clean_act, self.verbose)
      case _:
        if self.verbose:
          print("  \033[91mWARN: Unknown classifier, falling back to rule-based classifier.\033[0m")
        return RuleClassifier()

  def run_evaluate(self, path: str = ""):
    """
    Runs the evaluation suite defined in `evaluate.py`
    If a held-out test set is given with --evaluate path; use that for evaluation
    """
    from evaluate import (MODELS_PATH, load_models, save_models, evaluate_all,
                          evaluate_difficult_cases, evaluate_held_out)

    # Load cached models if possible, else initialize all classifiers
    if MODELS_PATH.exists():
      classifiers = load_models()
    else:
      classifiers = {name: self.get_classifier(name) for name in MODEL_NAMES}
      save_models(classifiers)

    if path:
      evaluate_held_out(classifiers, path)
    else:
      evaluate_all(classifiers, self.test_data, self.clean_test_data)
      evaluate_difficult_cases(classifiers["clean_embedded_mlp"], "clean_embedded_mlp")

  def run_classifier(self):
    """
    Repeatedly ask for an utterance and print the predicted dialog act

    Ask for utterances and print the predicted dialog act until the user types '!stop'

    Typing '!model <name>' switches to the `name` classifier
    """
    print(f"Using model: {self.model}")
    print("Type an utterance to classify it, '!model <name>' to switch models, or '!stop' to quit.")
    print(f"Available models: {', '.join(MODEL_NAMES)}")

    while True:
      # Graceful exit when EOF is encountered
      try:
        inp = input("> ").strip().lower()
      except EOFError:
        print("Stopping")
        return
      # Ignore empty input
      if len(inp) == 0:
        continue
      # Graceful exit when !stop command is issued
      if inp == "!stop":
        print("Stopping")
        return
      # Select model with given `name` when !model command is issued
      if inp.startswith("!model "):
        self.model = inp[len("!model "):].strip()
        self.classifier = self.get_classifier(self.model)
        continue

      # Clean the same way the training data was cleaned, so live input matches what the classifiers were trained on.
      inp = clean_utterance(inp)
      if len(inp) == 0:
        continue
      print(f"  Classified as: {self.classifier.run(inp)}")

  def run_manager(self):
    """Run the state manager from part 1b, with text-to-speech if specified with `--tts`"""
    from manager import Manager, State
    from tts import TTS

    manager = Manager(self.classifier, self.extractor, self.verbose, TTS() if self.tts else None, self.explain)
    while manager.state != State.FINISHED:
      manager.transition_state()
    manager.finish()


#  Entry point when ran as `$ python main.py`: parse args and run the CLI
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--model", default="clean_mlp", choices=MODEL_NAMES,
      help="Which classifier to use (default: clean_mlp); can also be changed at runtime with '!model <name>'"
  )
  parser.add_argument(
      "--extractor", default="levenshtein", choices=["levenshtein", "similarity"],
      help="Which slot extraction method to use (levenshtein distance or DistilBERT cosine similarity)"
  )
  parser.add_argument("--classify", action="store_true", help="Only run the classifier, not the entire chatbot")
  parser.add_argument(
      "--evaluate", default=None, const="", nargs="?", dest="path",
      help="Only run evaluation of all classifiers before exiting, optionally using a held-out dataset when [PATH] is given"
  )
  parser.add_argument("--tts", action="store_true", help="Use text-to-speech instead of text output")
  parser.add_argument("--explain", action="store_true", help="Explain why a restaurant is chosen")
  parser.add_argument("--verbose", action="store_true", help="Show more detailed reasoning information")
  args = parser.parse_args()

  cli = CLI(args)
  if args.classify:
    cli.run_classifier()
  elif args.path is not None:
    cli.run_evaluate(args.path)
  else:
    cli.run_manager()
