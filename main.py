# RUBEN: Cleanup code
# RUBEN: Add docstrings/comments
# RUBEN: Update README

"""Interactive command-line entry point: trains all classifiers and lets
the user classify typed utterances until they type 'stop'. """

import argparse

from classifiers import MODEL_NAMES, Classifier, EmbeddedLRClassifier, EmbeddedMLPClassifier, FrozenEmbeddingEncoder, LRClassifier, MLPClassifier, RuleClassifier
from data import clean_utterance, create_grouped_split, create_stratified_split, load_data
from manager import Manager, State
from slot_extraction import SlotExtract
from tts import TTS


class CLI:
  def __init__(self, args):
    self.model: str = args.model
    self.extractor: SlotExtract = SlotExtract(args.extractor)
    self.verbose: bool = args.verbose
    self.tts: bool = args.tts

    data = load_data()
    self.train_data, self.test_data = create_stratified_split(data)
    self.clean_train_data, self.clean_test_data = create_grouped_split(data)
    self.classifier = self.get_classifier()
    

  def get_classifier(self) -> Classifier:
    embedder = FrozenEmbeddingEncoder()
    utterance, act = self.train_data['utterance'], self.train_data['act']
    clean_utterance_col, clean_act = self.clean_train_data['utterance'], self.clean_train_data['act']

    match self.model:
      case "rule": return RuleClassifier()
      case "lr": return LRClassifier().fit(utterance, act, self.verbose)
      case "clean_lr": return LRClassifier().fit(clean_utterance_col, clean_act, self.verbose)
      case "mlp": return MLPClassifier().fit(utterance, act, self.verbose)
      case "clean_mlp": return MLPClassifier().fit(clean_utterance_col, clean_act, self.verbose)
      case "embedded_lr": return EmbeddedLRClassifier(embedder).fit(utterance, act, self.verbose)
      case "clean_embedded_lr": return EmbeddedLRClassifier(embedder).fit(clean_utterance_col, clean_act, self.verbose)
      case "embedded_mlp": return EmbeddedMLPClassifier(embedder).fit(utterance, act, self.verbose)
      case "clean_embedded_mlp": return EmbeddedMLPClassifier(embedder).fit(clean_utterance_col, clean_act, self.verbose)
      case _:
        print("  \033[91WARN: Unknown classifier, falling back to rule-based classifier.\033[0m")
        return RuleClassifier()
  

  def run_classifier(self):
    """Repeatedly ask for an utterance and print the predicted dialog act.

    Ask for utterances and print the predicted dialog act until the user types 'stop'. 
    Typing 'model <name>' switches to a different classifier
      
    """
    print(f"Using model: {self.model}")
    print("Type an utterance to classify it, '!model <name>' to switch models, or '!stop' to quit.")
    print(f"Available models: {', '.join(MODEL_NAMES)}")

    while True:
      try:
        inp = input("> ").strip().lower()
      except EOFError:
        print("Stopping")
        return

      if len(inp) == 0:
        continue
      if inp == "!stop":
        print("Stopping")
        return

      if inp.startswith("!model "):
        self.model = inp[len("!model "):].strip()
        self.classifier = self.get_classifier()

      # Clean the same way the training data was cleaned, so live input matches what the classifiers were trained on.
      inp = clean_utterance(inp)
      if len(inp) == 0:
        continue
      print(f"  Classified as: {self.classifier.run(inp)}")


  def run_manager(self):
    manager = Manager(self.classifier, self.extractor, self.verbose, TTS() if self.tts else None)
    while manager.state != State.FINISHED:
      manager.transition_state()
    manager.finish()


#  Entry point when ran as `$ python ./main.py`: parse args, load data, train, run the prompt
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model", default="mlp", choices=MODEL_NAMES,
    help="Which classifier to use (default: mlp); can also be changed at runtime with '!model <name>'"
  )
  parser.add_argument(
      "--extractor", default="levenshtein", choices=["levenshtein", "similarity"],
      help="Which slot extraction method to use (levenshtein distance or DistilBERT cosine similarity)"
  )
  parser.add_argument("--classify", action="store_true", help="Only run the classifier, not the entire chatbot")
  parser.add_argument("--tts", action="store_true", help="Use text-to-speech instead of text output")
  parser.add_argument("--verbose", action="store_true", help="Show more detailed reasoning information")
  args = parser.parse_args()

  cli = CLI(args)
  if args.classify:
    cli.run_classifier()
  else:
    cli.run_manager()