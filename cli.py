from classifiers import MODEL_NAMES, Classifier
from data import clean_utterance


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
    try:
      raw_inp = input("> ").strip()
    except EOFError:
      print("Stopping")
      return

    if len(raw_inp) == 0:
      continue

    if raw_inp.lower() == "stop":
      print("Stopping")
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
    if len(inp) == 0:
      continue

    if verbose:
      # Show every classifier's prediction, so you can compare them.
      for name, other_classifier in trained_classifiers.items():
        print(f"  {name}: {other_classifier.run(inp)}")
    else:
      print(f"  {classifier.run(inp)}")