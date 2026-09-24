"""Evaluate trained classifiers: on the original and grouped test splits, on a
held-out .dat file, and on hand-written difficult test cases.

Evaluation results (metrics, classification reports, confusion matrices,
predictions and errors) are saved locally under EXPERIMENTS_DIR. Re-running
evaluation for an unchanged classifier/dataset combination reuses the saved
result instead of recomputing it.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
)

from classifiers import Classifier, FrozenEmbeddingEncoder
from data import create_grouped_split, create_stratified_split, load_data
from main import train_all

# Local folder for all saved evaluation results.
EXPERIMENTS_DIR = Path("experiments")
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_PIPELINE_VERSION = "evaluation-v2-correct-splits-2026-09-23"
REUSE_SAVED_EVALUATIONS = True

# Classifiers trained on the grouped split are evaluated on the grouped
# test set; all others are evaluated on the original test set.
CLEAN_CLASSIFIER_NAMES = {"clean_lr", "clean_mlp", "clean_embedded_lr", "clean_embedded_mlp"}


# Helper functions used to save and reuse evaluation results

def _evaluation_name(name: str) -> str:
  """Turn a display name into a filesystem-safe folder name."""
  return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()


def _dataframe_fingerprint(data: pd.DataFrame) -> str:
  """Hash of a dataframe's content, used to detect if the test data changed."""
  ordered = data[["act", "utterance"]].astype(str)
  values = pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
  return hashlib.sha256(values).hexdigest()[:16]


def _classifier_fingerprint(classifier: Classifier) -> str:
  """Hash of a classifier's fitted state, used to detect if it was retrained."""
  parts = [classifier.__class__.__name__]
  for attribute in ("model", "vectorizer", "vocabulary", "KEYWORDS"):
    if hasattr(classifier, attribute):
      parts.append(joblib.hash(getattr(classifier, attribute)))
  return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _predict_labels(classifier: Classifier, utterances) -> np.ndarray:
  """Predict a whole batch at once if the classifier supports it, else one by one."""
  utterances = list(utterances)
  if hasattr(classifier, "predict"):
    predictions = classifier.predict(utterances)
    if len(predictions) == len(utterances):
      return np.asarray(predictions)
  return np.asarray([classifier.run(text) for text in utterances])


def _persist_classifier(experiment_dir: Path, classifier: Classifier) -> None:
  """Save a classifier's fitted sklearn parts, skipping the shared DistilBERT encoder."""
  if not hasattr(classifier, "model") and not hasattr(classifier, "vectorizer"):
    return
  model_path = experiment_dir / "classifier.joblib"
  if model_path.exists():
    return
  payload = {
    "class_name": classifier.__class__.__name__,
    "model": getattr(classifier, "model", None),
    "vectorizer": getattr(classifier, "vectorizer", None),
    "vocabulary": getattr(classifier, "vocabulary", None),
    "uses_shared_frozen_encoder": hasattr(classifier, "embedder"),
  }
  joblib.dump(payload, model_path, compress=3)


def persist_frozen_embedding_cache(embedder: FrozenEmbeddingEncoder) -> None:
  """Save the shared DistilBERT embedding cache so it can be reused later."""
  if not embedder.text_cache:
    return
  cache_dir = EXPERIMENTS_DIR / "frozen_embedding_cache_distilbert_base_uncased"
  cache_dir.mkdir(parents=True, exist_ok=True)
  texts = np.asarray(list(embedder.text_cache.keys()), dtype=str)
  embeddings = np.stack([embedder.text_cache[text] for text in texts]).astype(np.float32)
  np.savez_compressed(cache_dir / "embeddings.npz", texts=texts, embeddings=embeddings)
  (cache_dir / "config.json").write_text(
    json.dumps({
      "model_name": "distilbert-base-uncased",
      "pooling": "attention-mask-aware mean pooling",
      "count": len(texts),
    }, indent=2),
    encoding="utf-8",
  )


# Evaluating and storing results for a single classifier/dataset pair 

def evaluate_and_store(experiment_name: str, display_name: str, classifier: Classifier, data: pd.DataFrame) -> dict:
  """Evaluate one classifier on one dataset and save the full report to disk.

  Saves metrics.json, classification_report.csv, predictions.csv, errors.csv
  and a confusion matrix image. Reuses a previous result if the classifier
  and data have not changed since it was last saved.
  """
  experiment_dir = EXPERIMENTS_DIR / _evaluation_name(experiment_name)
  experiment_dir.mkdir(parents=True, exist_ok=True)
  config = {
    "evaluation_pipeline_version": EVALUATION_PIPELINE_VERSION,
    "display_name": display_name,
    "data_fingerprint": _dataframe_fingerprint(data),
    "classifier_fingerprint": _classifier_fingerprint(classifier),
    "examples": int(len(data)),
  }
  config_path = experiment_dir / "evaluation_config.json"
  metrics_path = experiment_dir / "metrics.json"

  if REUSE_SAVED_EVALUATIONS and config_path.exists() and metrics_path.exists():
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if saved_config == config:
      print(f"Loaded saved evaluation: {display_name}")
      return json.loads(metrics_path.read_text(encoding="utf-8"))

  utterances = data["utterance"].astype(str).tolist()
  gold = data["act"].astype(str).to_numpy()
  predicted = _predict_labels(classifier, utterances)
  labels = sorted(set(gold) | set(predicted))

  metrics = {
    "experiment": experiment_name,
    "model": display_name,
    "examples": int(len(gold)),
    "accuracy": float(accuracy_score(gold, predicted)),
    "balanced_accuracy": float(balanced_accuracy_score(gold, predicted)),
    "macro_f1": float(f1_score(gold, predicted, average="macro", zero_division=0)),
    "weighted_f1": float(f1_score(gold, predicted, average="weighted", zero_division=0)),
  }

  report = pd.DataFrame(
    classification_report(gold, predicted, labels=labels, output_dict=True, zero_division=0)
  ).T
  details = data[["utterance", "act"]].copy().reset_index(drop=True)
  details["prediction"] = predicted
  details["correct"] = details["act"].to_numpy() == predicted
  errors = details.loc[~details["correct"]]
  matrix = confusion_matrix(gold, predicted, labels=labels)

  metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
  config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
  report.to_csv(experiment_dir / "classification_report.csv")
  details.to_csv(experiment_dir / "predictions.csv", index=False)
  errors.to_csv(experiment_dir / "errors.csv", index=False)

  plt.figure(figsize=(11, 9))
  sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
  plt.title(f"Confusion matrix: {display_name}")
  plt.xlabel("Predicted label")
  plt.ylabel("True label")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  plt.savefig(experiment_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
  plt.close()

  _persist_classifier(experiment_dir, classifier)

  print(f"Saved evaluation: {display_name} -> {experiment_dir}")
  return metrics


# Evaluating all classifiers at once

def evaluate_all(classifiers: dict[str, Classifier], test_data: pd.DataFrame, clean_test_data: pd.DataFrame) -> pd.DataFrame:
  """Evaluate every classifier on the test split it was trained for, save a
  combined summary CSV, and return that summary as a dataframe."""
  display_names = {
    "rule": "Rule classifier",
    "lr": "BoW Logistic Regression - original split",
    "clean_lr": "BoW Logistic Regression - grouped split",
    "mlp": "BoW MLP - original split",
    "clean_mlp": "BoW MLP - grouped split",
    "embedded_lr": "Frozen embeddings + LR - original split",
    "clean_embedded_lr": "Frozen embeddings + LR - grouped split",
    "embedded_mlp": "Frozen embeddings + MLP - original split",
    "clean_embedded_mlp": "Frozen embeddings + MLP - grouped split",
  }

  rows = []
  for name, classifier in classifiers.items():
    data = clean_test_data if name in CLEAN_CLASSIFIER_NAMES else test_data
    display_name = display_names.get(name, name)
    rows.append(evaluate_and_store(name, display_name, classifier, data))

  summary = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
  summary.to_csv(EXPERIMENTS_DIR / "all_experiment_results.csv", index=False)
  print(summary.reset_index(drop=True))
  return summary


# Evaluating on a held-out test file

def evaluate_held_out(classifiers: dict[str, Classifier], path) -> pd.DataFrame:
  """Load a held-out .dat file (same format as dialog_acts.dat, already on
  disk, no download) and evaluate every classifier on it."""
  held_out_data = load_data(path)
  rows = []
  for name, classifier in classifiers.items():
    rows.append(evaluate_and_store(f"{name}_held_out", f"{name} (held-out)", classifier, held_out_data))
  summary = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
  summary.to_csv(EXPERIMENTS_DIR / "held_out_results.csv", index=False)
  print(summary.reset_index(drop=True))
  return summary


#  Hand-written difficult test cases

# Shorter, informal utterances: spelling variation, abbreviations, rare wording.
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
    "Is there a double 0 in the phone number?",
  ],
  "act": [
    "affirm", "negate", "request", "reqalts", "restart", "bye",
    "inform", "repeat", "request", "repeat", "inform", "inform",
    "inform", "reqalts", "repeat", "confirm",
  ],
})

# Longer sentences, some with two intentions (one dominant gold label is
# assigned, since the course dataset uses one label per utterance).
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
    "Not south, north please.",
  ],
  "act": [
    "deny", "request", "reqalts", "reqalts", "reqalts", "request",
    "confirm", "confirm", "thankyou", "thankyou", "request", "reqalts",
    "inform", "inform", "affirm", "inform",
  ],
})


def evaluate_difficult_cases(classifier: Classifier, classifier_name: str) -> None:
  """Run one classifier on both hand-written difficult test sets and save results."""
  for test_name, test_data in [("difficult_test_1", DIFFICULT_TEST_1), ("difficult_test_2", DIFFICULT_TEST_2)]:
    evaluate_and_store(
      f"{classifier_name}_{test_name}",
      f"{classifier_name} on {test_name}",
      classifier,
      test_data,
    )


if __name__ == "__main__":
  data = load_data()
  train_data, test_data = create_stratified_split(data)
  clean_train_data, clean_test_data = create_grouped_split(data)

  classifiers = train_all(train_data, clean_train_data)
  evaluate_all(classifiers, test_data, clean_test_data)

  # Try the difficult test cases on the best-performing model (adjust the
  # key here if a different classifier turns out to perform best).
  evaluate_difficult_cases(classifiers["clean_embedded_mlp"], "clean_embedded_mlp")

  persist_frozen_embedding_cache(classifiers["embedded_lr"].embedder)

  if len(sys.argv) > 1:
    evaluate_held_out(classifiers, sys.argv[1])
