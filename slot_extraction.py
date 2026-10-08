"""Extract restaurant preferences from a user utterance.

The dialog manager can call :func:`slot_extraction` after classifying an
utterance.  Exact keyword matching is deliberately kept separate from the
fallback methods so it is easy to compare the approaches later.
"""

import re
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from Levenshtein import distance as levenshtein_distance
from classifiers import Act
from restaurant import RestaurantInfo


@dataclass
class Suggestion:
  """A possible correction that still needs confirmation from the user."""

  inp: str
  suggest: str

  def __bool__(self) -> bool:
    return bool(self.suggest)

  def __iter__(self):
    # Allows PROMPTS["ask_confirm"].format(*suggestion) in the manager.
    return iter((self.inp, self.suggest))


class SlotExtract():
  # RUBEN: implement dontcare, address, phone, ZIP code and food keywords
  AREA_VALUES = ["north", "east", "south", "west", "centre"]
  PRICE_VALUES = ["cheap", "moderate", "expensive"]

  # Common words are not useful typo candidates.  Ignoring them also prevents a
  # short word such as "the" from being mapped to an unrelated cuisine.
  STOPWORDS = {
    "a", "an", "and", "any", "at", "be", "food", "for", "i", "in", "is",
    "it", "looking", "me", "of", "please", "restaurant",
    "serving", "something", "the", "to", "want", "with", "would",
  }

  ontology: dict[str, list[str]]
  exact: RestaurantInfo
  embedder: SentenceTransformer | None = None

  def __init__(self, mode: Literal["levenshtein", "similarity"] = "levenshtein"):
    self.load_ontology()
    self.exact = RestaurantInfo()
    if mode == "similarity":
      self.embedder = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")

  def load_ontology(self, path: str = "restaurant_info_extended.csv"):
    """Load the valid values for the three preference slots."""
    restaurants = pd.read_csv(path)
    # Sort by length - longer food names first, e.g. "modern european" before "european"
    food_values = sorted(restaurants["food"].dropna().str.lower().unique(), key=len, reverse=True)
    self.ontology = {"area": self.AREA_VALUES, "food": food_values, "pricerange": self.PRICE_VALUES}

  def _contains_phrase(self, text: str, phrase: str) -> bool:
    """Match a complete word or phrase instead of a substring."""
    return re.search(f"(^|\\s){phrase}(\\s|$)", text) is not None

  def extract_keywords(self, utterance: str) -> RestaurantInfo:
    """Return preferences that can be identified with exact keyword matching."""
    text = utterance.lower().strip()
    result = RestaurantInfo()

    for area in self.ontology["area"]:
      if self._contains_phrase(text, area):
        result["area"] = area
        break
    for price in self.ontology["pricerange"]:
      if self._contains_phrase(text, price):
        result["pricerange"] = price
        break
    for food in self.ontology["food"]:
      if self._contains_phrase(text, food):
        result["food"] = food
        break

    return result


  def _phrases(self, text: str, size: int) -> list[str]:
    """Return candidate phrases containing ``size`` non-stopwords."""
    words = list(filter(lambda w: w not in self.STOPWORDS, text.split()))
    result = []
    for start in range(len(words) - size + 1):
      phrase_words = words[start:start + size]
      result.append(" ".join(phrase_words))
    return result


  def _acceptable_distance(self, candidate: str, value: str, distance: int) -> bool:
    """Not more than 34% of the word can be misspelled"""
    longest = max(len(candidate), len(value))
    return distance / longest <= 0.34


  def _closest_value(self, utterance: str, values: list[str]) -> Suggestion | None:
    """Find the best plausible typo correction for one slot."""
    best: tuple[int, str, str] | None = None
    for value in values:
      # NOTE: This doesn't work with typing errors such as 'moderneuropean' where they missed a space
      candidates = self._phrases(utterance, 1) + self._phrases(utterance, 2) + self._phrases(utterance, 3)
      for candidate in candidates:
        distance = levenshtein_distance(candidate, value)
        if not self._acceptable_distance(candidate, value, distance):
          continue
        if best is None or distance < best[0]:
          best = (distance, candidate, value)

    if best is None:
      return None
    return Suggestion(best[1], best[2])


  def extract_with_levenshtein(self, utterance: str) -> dict[str, Suggestion]:
    """Run exact matching first, then suggest close spellings for missing slots."""
    suggestions: dict[str, Suggestion] = {}

    for slot in ("area", "food", "pricerange"):
      if slot in self.exact:
        continue
      suggestion = self._closest_value(utterance, self.ontology[slot])
      if suggestion is not None:
        suggestions[slot] = suggestion

    return suggestions


  def slot_extraction(self, act: Act, utterance: str) -> tuple[RestaurantInfo, dict[str, Suggestion]]:
    """Manager-compatible wrapper for keyword-based slot extraction."""
    self.exact = self.extract_keywords(utterance)
    if self.embedder:
      return self.exact, self.extract_with_semantic(utterance)
    else:
      return self.exact, self.extract_with_levenshtein(utterance)


  def closest_semantic_value(self, utterance: str, options: list[str], threshold: float) -> Suggestion | None:
    candidates = self._phrases(utterance, 1) + self._phrases(utterance, 2) + self._phrases(utterance, 3)
    if len(candidates) == 0:
      return None
    enc_candidates = self.embedder.encode(candidates)
    enc_options = self.embedder.encode(options)
    similarities = cosine_similarity(enc_candidates, enc_options)
    if similarities.max() < threshold:
      return
    argmax = np.unravel_index(similarities.argmax(), similarities.shape)
    return Suggestion(candidates[argmax[0]], options[argmax[1]])


  def extract_with_semantic(self, utterance, threshold: float = 0.5) -> dict[str, Suggestion]:
    suggestions: dict[str, Suggestion] = {}
    for slot in ("area", "food", "pricerange"):
      if slot in self.exact:
        continue
      suggestion = self.closest_semantic_value(utterance, self.ontology[slot], threshold)
      if suggestion is not None:
        suggestions[slot] = suggestion
    return suggestions


if __name__ == "__main__":
  extractor1 = SlotExtract("levenshtein")
  extractor2 = SlotExtract("similarity")

  test_sentences = [
    "i want somewhere fancy",
    "i want something inexpensive",
    "i want chinese food",
    "i want a restaurant in the center",
    "i want somewhere ceap",
  ]

  for sentence in test_sentences:
    print("\nSentence:", sentence)
    exact, suggestions = extractor1.slot_extraction(Act.INFORM, sentence)
    print("Levenshtein exact:", exact)
    print("Levenshtein suggestions:", suggestions)
    exact, suggestions = extractor2.slot_extraction(Act.INFORM, sentence)
    print("DistilBERT exact:", exact)
    print("DistilBERT suggestions:", suggestions)