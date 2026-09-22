from classifiers import Classifier

def accuracy(classifier: Classifier, X_test, y_test, show_incorrect=0) -> float:
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

def evaluate(classifier: Classifier, X_test, y_test, verbose=False):
  verbosity = 10 if verbose else 0
  print(classifier, "accuracy:", accuracy(classifier, X_test, y_test, show_incorrect=verbosity))