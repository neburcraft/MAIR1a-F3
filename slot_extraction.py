"""Extract restaurant preferences from a user utterance.

The dialog manager can call :func:`slot_extraction` after classifying an
utterance.  Exact keyword matching is deliberately kept separate from the
fallback methods so it is easy to compare the approaches later.
"""

from dataclasses import dataclass
import re
from pathlib import Path

import pandas as pd

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

# NOTE: Aliases are not allowed here. Lehvenstein distance and DistilBERT implementations
#       will be used to handle these cases.
AREA_ALIASES = {"center": "centre", "central": "centre"}
PRICE_ALIASES = {
  "moderately priced": "moderate",
  "moderate price": "moderate",
  "mid priced": "moderate",
}


def load_ontology(path: str = "restaurant_info_extended.csv") -> dict[str, tuple[str, ...]]:
  """Load the valid values for the three preference slots."""
  data_path = Path(path)
  if not data_path.exists():
    data_path = Path(__file__).with_name(path)

  restaurants = pd.read_csv(data_path)
  # NOTE: Explain here why you sort this way instead of down in line 105
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
  # NOTE: I made another regex for that in classifiers.py:52 which is more easily understandable
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
      # NOTE: While this makes sense for keyword matching, you are missing the
      #       pattern matching on variable keywords the description refers to.
      #       To be honest, I have no idea why they want this (except that it maybe helps
      #       for the Lehvenstein and DistilBERT implementation).
      #       We should ask in the class whether we need to do regex rules for this, because
      #       it seems like a bad solution at best and an impossible task at worst.
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

  # NOTE: Put explanation in load_ontology function instead
  # Check longer food names first, e.g. "modern european" before "european".
  for food in values["food"]:
    if _contains_phrase(text, food):
      result["food"] = food
      break

  return result


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
  del act # NOTE: This does not seem necessary to me
  return extract_keywords(utterance, ontology), {}
