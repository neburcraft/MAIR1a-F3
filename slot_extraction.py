"""Extract restaurant preferences from a user utterance.

The dialog manager can call :func:`slot_extraction` after classifying an
utterance.  Exact keyword matching is deliberately kept separate from the
fallback methods so it is easy to compare the approaches later.
"""

from dataclasses import dataclass
import re
from pathlib import Path

import pandas as pd
from Levenshtein import distance as levenshtein_distance

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


AREA_VALUES = ("north", "east", "south", "west", "centre")
PRICE_VALUES = ("cheap", "moderate", "expensive")

# NOTE: See notes from albert-slot-keyword as well
AREA_ALIASES = {"center": "centre", "central": "centre"}
PRICE_ALIASES = {
  "moderately priced": "moderate",
  "moderate price": "moderate",
  "mid priced": "moderate",
}

# Common words are not useful typo candidates.  Ignoring them also prevents a
# short word such as "the" from being mapped to an unrelated cuisine.
# NOTE: the word pricey (expensive) might also be ignored if 'priced'
#       is included in this list, which we don't want?
STOPWORDS = {
  "a", "an", "and", "any", "at", "be", "food", "for", "i", "in", "is",
  "it", "like", "looking", "me", "of", "please", "priced", "restaurant",
  "serving", "something", "the", "to", "want", "with", "would",
}


def load_ontology(path: str = "restaurant_info_extended.csv") -> dict[str, tuple[str, ...]]:
  """Load the valid values for the three preference slots."""
  data_path = Path(path)
  if not data_path.exists():
    data_path = Path(__file__).with_name(path)

  restaurants = pd.read_csv(data_path)
  food_values = tuple(
    sorted(restaurants["food"].dropna().str.lower().unique(), key=len, reverse=True)
  )
  return {
    "area": AREA_VALUES,
    "food": food_values,
    "pricerange": PRICE_VALUES,
  }


def _contains_phrase(text: str, phrase: str) -> bool:
  """Match a complete word or phrase instead of a substring."""
  return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def extract_keywords(
    utterance: str,
    ontology: dict[str, tuple[str, ...]] | None = None,
) -> RestaurantInfo:
  """Return preferences that can be identified with exact keyword matching."""
  text = utterance.lower().strip()
  values = ontology or load_ontology()
  result: RestaurantInfo = {}

  for area in values["area"]:
    if _contains_phrase(text, area):
      result["area"] = area
      break
  if "area" not in result:
    for alias, area in AREA_ALIASES.items():
      if _contains_phrase(text, alias):
        result["area"] = area
        break

  for price in values["pricerange"]:
    if _contains_phrase(text, price):
      result["pricerange"] = price
      break
  if "pricerange" not in result:
    for alias, price in PRICE_ALIASES.items():
      if _contains_phrase(text, alias):
        result["pricerange"] = price
        break

  # Check longer food names first, e.g. "modern european" before "european".
  for food in values["food"]:
    if _contains_phrase(text, food):
      result["food"] = food
      break

  return result


def _phrases(text: str, size: int) -> list[str]:
  """Return candidate phrases containing ``size`` words."""
  # NOTE: This is overkill. Considering the text is already cleaned, just do
  #       `words = text.split()`
  words = re.findall(r"[a-z]+", text.lower())
  result = []
  for start in range(len(words) - size + 1):
    phrase_words = words[start:start + size]
    if all(word in STOPWORDS for word in phrase_words):
      continue
    result.append(" ".join(phrase_words))
  return result


def _acceptable_distance(candidate: str, value: str, distance: int) -> bool:
  """Use a small length-dependent threshold to avoid random corrections."""
  longest = max(len(candidate), len(value))
  # NOTE: This 'and' seems a bit double, would just `longest // distance >= 4` not be enough?
  max_edits = 1 if longest <= 5 else 2 if longest <= 10 else 3
  return distance <= max_edits and distance / longest <= 0.34


def _closest_value(
    utterance: str,
    values: tuple[str, ...],
) -> Suggestion | None:
  """Find the best plausible typo correction for one slot."""
  best: tuple[int, str, str] | None = None
  for value in values:
    # NOTE: This doesn't work with typing errors such as 'moderneuropean' where they missed a space
    word_count = len(value.split())
    for candidate in _phrases(utterance, word_count):
      # NOTE: This will never pass, because `extract_with_levenshtein` already
      #       checks for exact matches first, right?
      if candidate == value:
        continue
      distance = levenshtein_distance(candidate, value)
      if not _acceptable_distance(candidate, value, distance):
        continue
      if best is None or distance < best[0]:
        best = (distance, candidate, value)

  if best is None:
    return None
  return Suggestion(best[1], best[2])


def extract_with_levenshtein(
    utterance: str,
    ontology: dict[str, tuple[str, ...]] | None = None,
) -> tuple[RestaurantInfo, dict[str, Suggestion]]:
  """Run exact matching first, then suggest close spellings for missing slots."""
  values = ontology or load_ontology()
  exact = extract_keywords(utterance, values)
  suggestions: dict[str, Suggestion] = {}

  for slot in ("area", "food", "pricerange"):
    if slot in exact:
      continue
    suggestion = _closest_value(utterance, values[slot])
    if suggestion is not None:
      suggestions[slot] = suggestion

  return exact, suggestions


def slot_extraction(
    act,
    utterance: str,
    ontology: dict[str, tuple[str, ...]] | None = None,
) -> tuple[RestaurantInfo, dict[str, Suggestion]]:
  """Manager-compatible wrapper for keyword-based slot extraction.

  ``act`` is accepted for compatibility with the state manager.  The text is
  still scanned for preferences because acts such as ``reqalts`` and ``deny``
  can also contain a new value.
  """
  del act
  return extract_with_levenshtein(utterance, ontology)
