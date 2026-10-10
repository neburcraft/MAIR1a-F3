"""Extract restaurant preferences and information requests from utterances."""

import re
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from data import clean_utterance
from restaurant import RestaurantInfo
from classifiers import Act


@dataclass
class Suggestion:
  """A possible correction that still needs confirmation from the user."""

  inp: str
  suggest: str

  def __bool__(self) -> bool:
    return bool(self.suggest)

  def __iter__(self):
    # Allows using the splat operator in str.format(*suggestion)
    return iter((self.inp, self.suggest))


class SlotExtract:
  AREA_VALUES = ["north", "east", "south", "west", "centre"]
  PRICE_VALUES = ["cheap", "moderate", "expensive"]
  ADDITION_VALUES = ["touristic", "assigned seats", "children", "romantic"]
  INFO_KEYWORDS = {
    "addr": ("address", "location", "located"),
    "phone": ("phone", "telephone", "phone number"),
    "postcode": ("postcode", "post code", "postal code", "zip code", "post"),
    "food": ("food", "cuisine", "serves"),
  }
  DONTCARE_PHRASES = [
    "dontcare", "don't care", "do not care", "doesn't matter", "anywhere", "anything",
    "does not matter", "no preference", "anything is fine", "any is fine", "whatever",
    "i don't mind", "i do not mind", "any area", "any food", "any price"
  ]
  SLOT_HINTS = {
    "area": ("area", "part of town", "location"),
    "food": ("food", "cuisine"),
    "pricerange": ("price", "cost", "priced"),
  }
  STOPWORDS = {
    "a", "an", "and", "any", "at", "be", "food", "for", "i", "in", "is",
    "it", "looking", "me", "of", "please", "restaurant", "serving",
    "something", "the", "to", "want", "with", "would",
  }

  ontology: dict[str, list[str]]
  exact: RestaurantInfo
  embedder: SentenceTransformer | None = None

  def __init__(self, mode: Literal["levenshtein", "similarity"] = "levenshtein"):
    self.load_ontology()
    self.exact = RestaurantInfo()
    self.embedder = None
    if mode == "similarity":
      self.embedder = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1")

  def load_ontology(self, path: str = "restaurant_info_extended.csv"):
    restaurants = pd.read_csv(path)
    food_values = sorted(
      restaurants["food"].dropna().str.lower().unique(), key=len, reverse=True
    )
    self.ontology = {
      "area": self.AREA_VALUES,
      "food": food_values,
      "pricerange": self.PRICE_VALUES,
      "additions": self.ADDITION_VALUES
    }

  def contains_phrase(self, text: str, phrase: str) -> bool:
    phrase = clean_utterance(phrase)
    return re.search(f"(^|\\s){phrase}(\\s|$)", text) is not None

  def extract_keywords(self, utterance: str, expected_slot: str | None = None) -> RestaurantInfo:
    """Extract exact slot values, additions and a slot-specific dontcare value."""
    result = RestaurantInfo()

    if any(self.contains_phrase(utterance, phrase) for phrase in self.DONTCARE_PHRASES) and expected_slot:
      result[expected_slot] = "dontcare"
    
    for area in self.ontology["area"]:
      if self.contains_phrase(utterance, area):
        result["area"] = area
        break
    for price in self.ontology["pricerange"]:
      if self.contains_phrase(utterance, price):
        result["pricerange"] = price
        break
    for food in self.ontology["food"]:
      if self.contains_phrase(utterance, food):
        result["food"] = food
        break
    for addition in self.ADDITION_VALUES:
      if self.contains_phrase(utterance, addition):
        result["additions"] = addition
        break

    return result

  def extract_requested_fields(self, utterance: str) -> list[str]:
    """Return restaurant details explicitly requested by the user."""
    result = []
    for field, keywords in self.INFO_KEYWORDS.items():
      if any(self.contains_phrase(utterance, key) for key in keywords):
        result.append(field)
    return result

  def _phrases(self, text: str, size: int) -> list[str]:
    """Return candidate phrases containing ``size`` non-stopwords."""
    words = [w for w in text.split() if w not in self.STOPWORDS]
    result = []
    for start in range(len(words) - size + 1):
      phrase_words = words[start:start + size]
      result.append(" ".join(phrase_words))
    return result

  def _acceptable_distance(self, candidate: str, value: str, distance: int) -> bool:
    longest = max(len(candidate), len(value))
    return distance / longest <= 0.34

  def _closest_value(self, utterance: str, values: list[str]) -> Suggestion | None:
    """Find the best plausible typo correction for one slot."""
    best: tuple[int, str, str] | None = None
    candidates = self._phrases(utterance, 1) + self._phrases(utterance, 2) + self._phrases(utterance, 3)
    for value in values:
      for candidate in candidates:
        distance = levenshtein_distance(candidate, value)
        if not self._acceptable_distance(candidate, value, distance):
          continue
        if best is None or distance < best[0]:
          best = (distance, candidate, value)
    if best:
      return Suggestion(best[1], best[2])
    return None

  def extract_with_levenshtein(self, utterance: str) -> dict[str, Suggestion]:
    """Run exact matching first, then suggest close spellings for missing slots."""
    suggestions: dict[str, Suggestion] = {}
    for slot in ("area", "food", "pricerange", "additions"):
      if slot in self.exact:
        continue
      suggestion = self._closest_value(utterance, self.ontology[slot])
      if suggestion:
        suggestions[slot] = suggestion
    return suggestions

  def closest_semantic_value(self, utterance: str, options: list[str], threshold: float) -> Suggestion | None:
    candidates = self._phrases(utterance, 1) + self._phrases(utterance, 2) + self._phrases(utterance, 3)
    if len(candidates) == 0:
      return None
    similarities = cosine_similarity(self.embedder.encode(candidates), self.embedder.encode(options))
    if similarities.max() < threshold:
      return None
    row, column = np.unravel_index(similarities.argmax(), similarities.shape)
    return Suggestion(candidates[row], options[column])

  def extract_with_semantic(self, utterance: str, threshold: float = 0.5) -> dict[str, Suggestion]:
    suggestions = {}
    for slot in ("area", "food", "pricerange"):
      if slot in self.exact:
        continue
      suggestion = self.closest_semantic_value(utterance, self.ontology[slot], threshold)
      if suggestion is not None:
        suggestions[slot] = suggestion
    return suggestions

  def slot_extraction(
      self, act: Act, utterance: str, expected_slot: str | None = None,
  ) -> tuple[RestaurantInfo, dict[str, Suggestion]]:
    self.exact = self.extract_keywords(utterance, expected_slot)
    if self.embedder:
      return self.exact, self.extract_with_semantic(utterance)
    return self.exact, self.extract_with_levenshtein(utterance)
