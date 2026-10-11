"""
Evaluate trained classifiers: on the original and grouped test splits, on a
held-out .dat file, and on hand-written difficult test cases.

Evaluation results (metrics, classification reports, confusion matrices,
predictions and errors) are saved locally under `EXPERIMENTS_DIR`.
"""

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, f1_score)

from classifiers import (Classifier, RuleClassifier, LRClassifier, MLPClassifier,
                         FrozenEmbeddingEncoder, EmbeddedLRClassifier, EmbeddedMLPClassifier)
from data import load_data

EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)
MODELS_PATH = Path("trained_models.joblib")

DIFFICULT_TEST_1 = pd.DataFrame([
    ("yeah, that sounds right", "affirm"),
    ("nope, not that one", "negate"),
    ("what's the address again?", "request"),
    ("any other options?", "reqalts"),
    ("actually, let's start over", "restart"),
    ("okay thanks, bye", "bye"),
    ("sth. more expensive plz.", "inform"),
    ("Repeat that plz.", "repeat"),
    ("Which direction is it in the city?", "request"),
    ("Repeat what you said please.", "repeat"),
    ("Chinese food.", "inform"), ("Hangzhoucai please.", "inform"),
    ("I love Sushi. Recommend.", "inform"),
    ("I wanna try something new.", "reqalts"),
    ("Repeat the phone number", "repeat"),
    ("Is there a double 0 in the phone number?", "confirm")],
    columns=("utterance", "act")
)

DIFFICULT_TEST_2 = pd.DataFrame([
    ("I do not want to try this one", "deny"),
    ("Thanx, and tell me the exact location please", "request"),
    ("Maybe could you tell me something else? I do not like this one", "reqalts"),
    ("I need this restaurant address and recommend me another restaurant", "request"),
    ("Recommend me three restaurants which are all with good food quality", "inform"),
    ("What time I can go to the restaurant that is not crowded", "request"),
    ("We need to stay from 17 to 20 o'clock, is it possible?", "confirm"),
    ("BTW, is the restaurant not crowded at 18?", "confirm"),
    ("We can to go the restaurananant you recommend. Thank you.", "thankyou"),
    ("That will be fine. Thank you so much.", "thankyou"),
    ("Would you please tell me the phone number of these two restaurant?", "request"),
    ("Just please give me three names that are Italian ones.", "inform"),
    ("One of my friends said he does not like Indian food, please let me try Thai food.", "inform"),
    ("We will go to the park in the south, would you please recommend me a restaurant in the south?",
     "inform"), ("Okay, we will try the first one.", "affirm"),
    ("Not south, north please.", "negate")],
    columns=("utterance", "act")
)


def predict_labels(classifier: Classifier, utterances: list[str]):
  """Classify the speech-act of a list of utterances, using batch processing where possible"""
  utterances = list(utterances)
  # Use batch processing where possible
  if hasattr(classifier, "predict"):
    return classifier.predict(utterances)

  return np.array([classifier.run(utt).value for utt in utterances])


def save_models(classifiers: dict[str, Classifier], path: Path = MODELS_PATH):
  """Save the trained classifiers to a file (MODELS_PATH) so the next evaluate does not retrain all models"""
  saved_models = {}
  for name, classifier in classifiers.items():
    if isinstance(classifier, RuleClassifier):
      saved_models[name] = {"type": "rule"}
    elif isinstance(classifier, LRClassifier):
      saved_models[name] = {
          "type": "lr",
          "model": classifier.model,
          "vectorizer": classifier.vectorizer,
          "vocabulary": classifier.vocabulary
      }
    elif isinstance(classifier, MLPClassifier):
      saved_models[name] = {
          "type": "mlp",
          "model": classifier.model,
          "vectorizer": classifier.vectorizer,
          "vocabulary": classifier.vocabulary
      }
    elif isinstance(classifier, EmbeddedLRClassifier):
      saved_models[name] = {"type": "embedded_lr", "model": classifier.model}
    elif isinstance(classifier, EmbeddedMLPClassifier):
      saved_models[name] = {"type": "embedded_mlp", "model": classifier.model}

  joblib.dump(saved_models, path)
  print("Saved trained models")


def load_models(path: Path = MODELS_PATH) -> dict[str, Classifier]:
  """Load saved models (pre-trained) from a file (MODELS_PATH) and returns them as a dictionary"""
  saved_models = joblib.load(path)
  classifiers = {}
  embedder = FrozenEmbeddingEncoder()

  for name, saved in saved_models.items():
    if saved["type"] == "rule":
      classifier = RuleClassifier()
    elif saved["type"] == "lr":
      classifier = LRClassifier()
      classifier.model = saved["model"]
      classifier.vectorizer = saved["vectorizer"]
      classifier.vocabulary = saved["vocabulary"]
    elif saved["type"] == "mlp":
      classifier = MLPClassifier()
      classifier.model = saved["model"]
      classifier.vectorizer = saved["vectorizer"]
      classifier.vocabulary = saved["vocabulary"]
    elif saved["type"] == "embedded_lr":
      classifier = EmbeddedLRClassifier(embedder)
      classifier.model = saved["model"]
    elif saved["type"] == "embedded_mlp":
      classifier = EmbeddedMLPClassifier(embedder)
      classifier.model = saved["model"]
    classifiers[name] = classifier

  print("Loaded trained models")
  return classifiers


def evaluate(experiment_name: str, display_name: str, classifier: Classifier,
             data: pd.DataFrame) -> dict[str, Any]:
  """
  Evaluate a single classifier on the given test data, print the accuracy results to the console and
  save the results of the evaluation to `EXPERIMENTS_DIR/experiment_name`
  """
  experiment_dir = EXPERIMENTS_DIR / experiment_name
  experiment_dir.mkdir(parents=True, exist_ok=True)

  utterances = data["utterance"].astype(str).tolist()
  correct = data["act"].astype(str).to_numpy()
  predicted = predict_labels(classifier, utterances)

  accuracy = accuracy_score(correct, predicted)
  balanced_accuracy = balanced_accuracy_score(correct, predicted)
  macro_f1 = f1_score(correct, predicted, average="macro", zero_division=0)
  weighted_f1 = f1_score(correct, predicted, average="weighted", zero_division=0)

  metrics = {
      "experiment": experiment_name,
      "model": display_name,
      "examples": len(data),
      "accuracy": accuracy,
      "balanced_accuracy": balanced_accuracy,
      "macro_f1": macro_f1,
      "weighted_f1": weighted_f1
  }

  # Save metrics to a csv file
  pd.DataFrame([metrics]).to_csv(experiment_dir / "metrics.csv", index=False)

  # Generate a scikit-learn classification report for all labels and save to a file
  # `correct` and `predicted` are both included as some acts may be included but not predicted or vice versa
  labels = sorted(set(correct) | set(predicted))
  report = classification_report(correct,
                                 predicted,
                                 labels=labels,
                                 output_dict=True,
                                 zero_division=0)
  pd.DataFrame(report).T.to_csv(experiment_dir / "classification_report.csv")

  # Save all predictions to `predictions.csv` and all incorrect predictions to `errors.csv`
  results = data[["utterance", "act"]].copy().reset_index(drop=True)
  results["prediction"] = predicted
  results["correct"] = results["act"] == results["prediction"]
  results.to_csv(experiment_dir / "predictions.csv", index=False)
  results[results["correct"] == False].to_csv(experiment_dir / "errors.csv", index=False)

  # Generate a scikit-learn confusion matrix between the correct and predicted speech-act labels
  # Save the figure to a file `confusion_matrix.png`
  matrix = confusion_matrix(correct, predicted, labels=labels)
  display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

  _, ax = plt.subplots(figsize=(11, 9))
  display.plot(ax=ax, xticks_rotation=45, colorbar=False)
  plt.title(f"Confusion matrix: {display_name}")
  plt.tight_layout()
  plt.savefig(experiment_dir / "confusion_matrix.png", dpi=180)
  plt.close()

  # Print summary of evaluation to console and return it for final `evaluate_all` summary
  print(
      f"{display_name} - accuracy: {round(accuracy, 4) } - balanced accuracy: {round(balanced_accuracy, 4)} - macro F1: {round(macro_f1, 4)}"
  )
  return metrics


def evaluate_all(classifiers: dict[str, Classifier], test_data: pd.DataFrame,
                 clean_test_data: pd.DataFrame) -> pd.DataFrame:
  """
  Evaluate all given classifiers with the test data, save the results to separate folders in `EXPERIMENTS_DIR` and print summary to console
  
  Classifiers use `test_data` when trained on the normal stratified split, `clean_test_data` for the grouped split models
  """
  display_names = {
      "rule": "Rule classifier",
      "lr": "BoW Logistic Regression - original split",
      "clean_lr": "BoW Logistic Regression - grouped split",
      "mlp": "BoW MLP - original split",
      "clean_mlp": "BoW MLP - grouped split",
      "embedded_lr": "Frozen embeddings + LR - original split",
      "clean_embedded_lr": "Frozen embeddings + LR - grouped split",
      "embedded_mlp": "Frozen embeddings + MLP - original split",
      "clean_embedded_mlp": "Frozen embeddings + MLP - grouped split"
  }

  results = []
  for name, classifier in classifiers.items():
    # Use clean test data for the models that used grouped split training data
    test = clean_test_data if name.startswith("clean") else test_data
    metrics = evaluate(name, display_names[name], classifier, test)
    results.append(metrics)

  # Save results summary to `all_experiments_results.csv`
  results = pd.DataFrame(results)
  results = results.sort_values(["macro_f1", "accuracy"], ascending=False)
  results.to_csv(EXPERIMENTS_DIR / "all_experiment_results.csv", index=False)

  print(results.reset_index(drop=True))
  return results


def evaluate_held_out(classifiers: dict[str, Classifier], path: str) -> pd.DataFrame:
  """Evaluate the models on a held-out test data set stored at `path`"""
  held_out_data = load_data(path)
  return evaluate_all(classifiers, held_out_data, held_out_data)


def evaluate_difficult_cases(classifier: Classifier, classifier_name: str):
  """Run evalutation of difficult test cases `DIFFICULT_TEST_1` and `DIFFICULT_TEST_2`"""
  evaluate(f"{classifier_name}_difficult_test_1", f"{classifier_name} on difficult test 1",
           classifier, DIFFICULT_TEST_1)
  evaluate(f"{classifier_name}_difficult_test_2", f"{classifier_name} on difficult test 2",
           classifier, DIFFICULT_TEST_2)
