"""Evaluate trained classifiers: on the original and grouped test splits, on a
held-out .dat file, and on hand-written difficult test cases.

Evaluation results (metrics, classification reports, confusion matrices,
predictions and errors) are saved locally under EXPERIMENTS_DIR. Re-running
evaluation for an unchanged classifier/dataset combination reuses the saved
result instead of recomputing it.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

from classifiers import RuleClassifier, LRClassifier, MLPClassifier, FrozenEmbeddingEncoder, EmbeddedLRClassifier, EmbeddedMLPClassifier
from data import load_data, create_stratified_split, create_grouped_split


EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(exist_ok=True)

MODELS_PATH = Path("trained_models.joblib")

CLEAN_CLASSIFIER_NAMES = {"clean_lr", "clean_mlp", "clean_embedded_lr", "clean_embedded_mlp"}


def predict_labels(classifier, utterances):
  utterances = list(utterances)

  if hasattr(classifier, "predict"):
    return classifier.predict(utterances)


  predictions = []
  for utterance in utterances:
    predictions.append(classifier.run(utterance).value)

  return np.array(predictions)




def save_models(classifiers):
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

  joblib.dump(saved_models, MODELS_PATH)
  print("Saved trained models")



def load_models():
  saved_models = joblib.load(MODELS_PATH)

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




def evaluate(experiment_name, display_name, classifier, data):
  experiment_dir = EXPERIMENTS_DIR / experiment_name
  experiment_dir.mkdir(parents=True, exist_ok=True)

  utterances = data["utterance"].astype(str).tolist()
  gold = data["act"].astype(str).to_numpy()
  predicted = predict_labels(classifier, utterances)

  accuracy = accuracy_score(gold, predicted)
  balanced_accuracy = balanced_accuracy_score(gold, predicted)
  macro_f1 = f1_score(gold, predicted, average="macro", zero_division=0)
  weighted_f1 = f1_score(gold, predicted, average="weighted", zero_division=0)

  metrics = {
    "experiment": experiment_name,
    "model": display_name,
    "examples": len(data),
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1
  }

  pd.DataFrame([metrics]).to_csv(experiment_dir / "metrics.csv", index=False)

  labels = sorted(set(gold) | set(predicted))
  report = classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0)
  pd.DataFrame(report).T.to_csv(experiment_dir / "classification_report.csv")

  results = data[["utterance", "act"]].copy().reset_index(drop=True)
  results["prediction"] = predicted
  results["correct"] = results["act"] == results["prediction"]

  results.to_csv(experiment_dir / "predictions.csv", index=False)
  results[results["correct"] == False].to_csv(experiment_dir / "errors.csv", index=False)

  matrix = confusion_matrix(gold, predicted, labels=labels)
  display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)

  fig, ax = plt.subplots(figsize=(11, 9))
  display.plot(ax=ax, xticks_rotation=45, colorbar=False)
  plt.title(f"Confusion matrix: {display_name}")
  plt.tight_layout()
  plt.savefig(experiment_dir / "confusion_matrix.png", dpi=180)
  plt.close()

  print(display_name, "- accuracy:", round(accuracy, 4), "- balanced accuracy:", round(balanced_accuracy, 4), "- macro F1:", round(macro_f1, 4))

  return metrics


def evaluate_all(classifiers, test_data, clean_test_data):
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
    if name in CLEAN_CLASSIFIER_NAMES:
      test = clean_test_data
    else:
      test = test_data

    metrics = evaluate(name, display_names[name], classifier, test)
    results.append(metrics)

  results = pd.DataFrame(results)
  results = results.sort_values(["macro_f1", "accuracy"], ascending=False)
  results.to_csv(EXPERIMENTS_DIR / "all_experiment_results.csv", index=False)

  print(results.reset_index(drop=True))
  return results


def evaluate_held_out(classifiers, path):
  held_out_data = load_data(path)
  results = []

  for name, classifier in classifiers.items():
    metrics = evaluate(name + "_held_out", name + " (held-out)", classifier, held_out_data)
    results.append(metrics)

  results = pd.DataFrame(results)
  results = results.sort_values(["macro_f1", "accuracy"], ascending=False)
  results.to_csv(EXPERIMENTS_DIR / "held_out_results.csv", index=False)

  print(results.reset_index(drop=True))
  return results


DIFFICULT_TEST_1 = pd.DataFrame({
  "utterance": [
    "yeah, that sounds right",
    "nope, not that one",
    "what's the address again?",
    "any other options?",
    "actually, let's start over",
    "okay thanks, bye",
    "sth. more expensive plz.",
    "Repeat that plz.",
    "Which direction is it in the city?",
    "Repeat what you said please.",
    "Chinese food.",
    "Hangzhoucai please.",
    "I love Sushi. Recommend.",
    "I wanna try something new.",
    "Repeat the phone number",
    "Is there a double 0 in the phone number?"
  ],
  "act": [
    "affirm", "negate", "request", "reqalts",
    "restart", "bye", "inform", "repeat",
    "request", "repeat", "inform", "inform",
    "inform", "reqalts", "repeat", "confirm"
  ]
})


DIFFICULT_TEST_2 = pd.DataFrame({
  "utterance": [
    "I do not want to try this one",
    "Thanx, and tell me the exact location please",
    "Maybe could you tell me something else? I do not like this one",
    "I need this restaurant address and recommend me another restaurant",
    "Recommend me three restaurants which are all with good food quality",
    "What time I can go to the restaurant that is not crowded",
    "We need to stay from 17 to 20 o'clock, is it possible?",
    "BTW, is the restaurant not crowded at 18?",
    "We can to go the restaurananant you recommend. Thank you.",
    "That will be fine. Thank you so much.",
    "Would you please tell me the phone number of these two restaurant?",
    "Just please give me three names that are Italian ones.",
    "One of my friends said he does not like Indian food, please let me try Thai food.",
    "We will go to the park in the south, would you please recommend me a restaurant in the south?",
    "Okay, we will try the first one.",
    "Not south, north please."
  ],
  "act": [
    "deny", "request", "reqalts", "reqalts",
    "reqalts", "request", "confirm", "confirm",
    "thankyou", "thankyou", "request", "reqalts",
    "inform", "inform", "affirm", "inform"
  ]
})


def evaluate_difficult_cases(classifier, classifier_name):
  evaluate(classifier_name + "_difficult_test_1", classifier_name + " on difficult test 1", classifier, DIFFICULT_TEST_1)
  evaluate(classifier_name + "_difficult_test_2", classifier_name + " on difficult test 2", classifier, DIFFICULT_TEST_2)


if __name__ == "__main__":
  data = load_data()
  train_data, test_data = create_stratified_split(data)
  clean_train_data, clean_test_data = create_grouped_split(data)

  if MODELS_PATH.exists():
    classifiers = load_models()
  else:
    embedder = FrozenEmbeddingEncoder()
    utterance, act = train_data['utterance'], train_data['act']
    clean_utterance_col, clean_act = clean_train_data['utterance'], clean_train_data['act']
    classifiers = {
      "rule": RuleClassifier(),
      "lr": LRClassifier().fit(utterance, act),
      "clean_lr": LRClassifier().fit(clean_utterance_col, clean_act),
      "mlp": MLPClassifier().fit(utterance, act),
      "clean_mlp": MLPClassifier().fit(clean_utterance_col, clean_act),
      "embedded_lr": EmbeddedLRClassifier(embedder).fit(utterance, act),
      "clean_embedded_lr": EmbeddedLRClassifier(embedder).fit(clean_utterance_col, clean_act),
      "embedded_mlp": EmbeddedMLPClassifier(embedder).fit(utterance, act),
      "clean_embedded_mlp": EmbeddedMLPClassifier(embedder).fit(clean_utterance_col, clean_act)
    }
    save_models(classifiers)

  evaluate_all(classifiers, test_data, clean_test_data)
  evaluate_difficult_cases(classifiers["clean_embedded_mlp"], "clean_embedded_mlp")

  if len(sys.argv) > 1:
    evaluate_held_out(classifiers, sys.argv[1])