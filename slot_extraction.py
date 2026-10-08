"""Extract restaurant preferences and information requests from utterances."""

import re
from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import pandas as pd
from restaurant import RestaurantInfo

if TYPE_CHECKING:
  from classifiers import Act
else:
  Act = Any

try:
  from Levenshtein import distance as levenshtein_distance
except ImportError:
  # Small fallback so keyword extraction still works before dependencies are installed.
  def levenshtein_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for row, first_char in enumerate(first, start=1):
      current = [row]
      for column, second_char in enumerate(second, start=1):
        current.append(min(
          current[-1] + 1,
          previous[column] + 1,
          previous[column - 1] + (first_char != second_char),
        ))
      previous = current
    return previous[-1]


@dataclass
class Suggestion:
  """A possible correction that still needs confirmation from the user."""

  inp: str
  suggest: str

  def __bool__(self) -> bool:
    return bool(self.suggest)

  def __iter__(self):
    return iter((self.inp, self.suggest))


class SlotExtract:
  AREA_VALUES = ["north", "east", "south", "west", "centre"]
  PRICE_VALUES = ["cheap", "moderate", "expensive"]
  ADDITION_PHRASES = {
    "touristic": ("touristic", "touristy"),
    "assigned seats": ("assigned seats", "assigned seating"),
    "children": ("children", "child", "kids", "kid"),
    "romantic": ("romantic", "romance"),
  }
  INFO_KEYWORDS = {
    "addr": ("address", "location", "located"),
    "phone": ("phone", "telephone", "phone number"),
    "postcode": ("postcode", "post code", "postal code", "zip code"),
    "food": ("food", "cuisine", "serves"),
  }
  DONTCARE_PHRASES = (
    "dontcare", "don't care", "do not care", "doesn't matter",
    "does not matter", "no preference", "anything is fine", "any is fine",
  )
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

  def __init__(self, mode: Literal["levenshtein", "similarity"] = "levenshtein"):
    self.mode = mode
    self.load_ontology()
    self.exact = RestaurantInfo()
    self.embedder = None
    if mode == "similarity":
      # Loading DistilBERT is expensive, so only do it when that mode is used.
      from sentence_transformers import SentenceTransformer
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
    }

  def _contains_phrase(self, text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

  def _has_dontcare(self, text: str) -> bool:
    return any(self._contains_phrase(text, phrase) for phrase in self.DONTCARE_PHRASES)

  def extract_keywords(self, utterance: str, expected_slot: str | None = None) -> RestaurantInfo:
    """Extract exact slot values, additions and a slot-specific dontcare value."""
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

    for addition, phrases in self.ADDITION_PHRASES.items():
      if any(self._contains_phrase(text, phrase) for phrase in phrases):
        result["additions"] = addition
        break

    if self._has_dontcare(text):
      slot = expected_slot
      if slot is None:
        for name, hints in self.SLOT_HINTS.items():
          if any(self._contains_phrase(text, hint) for hint in hints):
            slot = name
            break
      if slot in ("area", "food", "pricerange"):
        result[slot] = "dontcare"

    return result

  def extract_requested_fields(self, utterance: str) -> list[str]:
    """Return restaurant details explicitly requested by the user."""
    text = utterance.lower().strip()
    return [
      field for field, keywords in self.INFO_KEYWORDS.items()
      if any(self._contains_phrase(text, keyword) for keyword in keywords)
    ]

  def _phrases(self, text: str, size: int) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower())
    words = [word for word in words if word not in self.STOPWORDS]
    return [" ".join(words[start:start + size]) for start in range(len(words) - size + 1)]

  def _acceptable_distance(self, candidate: str, value: str, distance: int) -> bool:
    longest = max(len(candidate), len(value))
    return longest > 0 and distance / longest <= 0.34

  def _closest_value(self, utterance: str, values: list[str]) -> Suggestion | None:
    best: tuple[int, str, str] | None = None
    candidates = self._phrases(utterance, 1) + self._phrases(utterance, 2) + self._phrases(utterance, 3)
    for value in values:
      for candidate in candidates:
        distance = levenshtein_distance(candidate, value)
        if not self._acceptable_distance(candidate, value, distance):
          continue
        if best is None or distance < best[0]:
          best = (distance, candidate, value)
    return None if best is None else Suggestion(best[1], best[2])

  def extract_with_levenshtein(self, utterance: str) -> dict[str, Suggestion]:
    suggestions = {}
    for slot in ("area", "food", "pricerange"):
      if slot in self.exact:
        continue
      suggestion = self._closest_value(utterance, self.ontology[slot])
      if suggestion is not None:
        suggestions[slot] = suggestion
    return suggestions

  def closest_semantic_value(self, utterance: str, options: list[str], threshold: float) -> Suggestion | None:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    candidates = self._phrases(utterance, 1) + self._phrases(utterance, 2) + self._phrases(utterance, 3)
    if not candidates:
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
    if self.mode == "similarity":
      return self.exact, self.extract_with_semantic(utterance)
    return self.exact, self.extract_with_levenshtein(utterance)


_default_extractor: SlotExtract | None = None


def get_default_extractor() -> SlotExtract:
  global _default_extractor
  if _default_extractor is None:
    _default_extractor = SlotExtract("levenshtein")
  return _default_extractor


def slot_extraction(
  act: Act, utterance: str, expected_slot: str | None = None,
) -> tuple[RestaurantInfo, dict[str, Suggestion]]:
  """Compatibility wrapper used by the dialog manager."""
  return get_default_extractor().slot_extraction(act, utterance, expected_slot)


def extract_requested_fields(utterance: str) -> list[str]:
  return get_default_extractor().extract_requested_fields(utterance)
