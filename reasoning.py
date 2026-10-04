"""Small rule-based reasoning component for restaurant recommendations."""

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class Inference:
  rule_id: int
  property: str
  value: bool
  reason: str


def _value(restaurant: dict, key: str) -> str:
  return str(restaurant.get(key, "")).strip().lower()


def apply_rules(restaurant: dict) -> list[Inference]:
  """Apply all six assignment rules to one restaurant."""
  results = []

  if (_value(restaurant, "pricerange") == "cheap"
      and _value(restaurant, "food quality") == "good"):
    results.append(Inference(1, "touristic", True, "it is cheap and has good food"))
  if _value(restaurant, "food") == "romanian":
    results.append(Inference(2, "touristic", False, "it serves Romanian food"))
  if _value(restaurant, "crowdedness") == "busy":
    results.append(Inference(3, "assigned seats", True, "it is busy"))
  if _value(restaurant, "length of stay") == "long":
    results.append(Inference(4, "children", False, "it allows a long stay"))
  if _value(restaurant, "crowdedness") == "busy":
    results.append(Inference(5, "romantic", False, "it is busy"))
  if _value(restaurant, "length of stay") == "long":
    results.append(Inference(6, "romantic", True, "it allows a long stay"))

  return results


def derived_values(inferences: list[Inference]) -> dict[str, set[bool]]:
  """Keep all conclusions so contradictions remain visible."""
  values: dict[str, set[bool]] = {}
  for inference in inferences:
    values.setdefault(inference.property, set()).add(inference.value)
  return values


def matches_requirements(restaurant: dict, requirements: dict[str, bool]) -> bool:
  """Check extra requirements using a conservative contradiction policy.

  When rules derive both True and False for the same property, the restaurant
  is not used for that requirement because the system cannot make a reliable
  claim to the user.
  """
  values = derived_values(apply_rules(restaurant))
  for property_name, requested_value in requirements.items():
    if values.get(property_name) != {requested_value}:
      return False
  return True


def filter_candidates(
    restaurants: pd.DataFrame,
    requirements: dict[str, bool],
) -> pd.DataFrame:
  """Filter lookup results by the user's additional requirements."""
  if not requirements:
    return restaurants.copy()
  keep = [
    matches_requirements(row.to_dict(), requirements)
    for _, row in restaurants.iterrows()
  ]
  return restaurants.loc[keep].copy()


PROPERTY_PHRASES = {
  "touristic": ("touristic", "touristy"),
  "assigned seats": ("assigned seats", "assigned seat", "assigned seating"),
  "children": ("children", "child", "kids", "kid"),
  "romantic": ("romantic", "romance"),
}


def extract_additional_requirements(utterance: str) -> dict[str, bool]:
  """Extract simple positive or negative additional requirements."""
  text = utterance.lower()
  result = {}
  for property_name, phrases in PROPERTY_PHRASES.items():
    for phrase in phrases:
      if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text):
        negative = re.search(
          rf"\b(no|not|without)\b(?:\s+\w+){{0,2}}\s+{re.escape(phrase)}\b",
          text,
        )
        result[property_name] = negative is None
        break
  return result


def explain_recommendation(
    restaurant: dict,
    requirements: dict[str, bool],
) -> str:
  """Return a short explanation for the first relevant rule."""
  if not requirements:
    return ""
  for inference in apply_rules(restaurant):
    if requirements.get(inference.property) == inference.value:
      claims = {
        ("touristic", True): "It is touristic",
        ("touristic", False): "It is not touristic",
        ("assigned seats", True): "It has assigned seats",
        ("children", False): "It is not suitable for children",
        ("romantic", True): "It is romantic",
        ("romantic", False): "It is not romantic",
      }
      claim = claims[(inference.property, inference.value)]
      return f"{claim} because {inference.reason}."
  return ""
