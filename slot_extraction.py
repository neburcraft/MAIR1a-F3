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
import numpy as np
from classifiers import FrozenEmbeddingEncoder
from sentence_transformers import SentenceTransformer


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
    method: str = "levenshtein",
    ontology: dict[str, tuple[str, ...]] | None = None,
) -> tuple[RestaurantInfo, dict[str, Suggestion]]:
  """Manager-compatible wrapper for keyword-based slot extraction.

  ``act`` is accepted for compatibility with the state manager.  The text is
  still scanned for preferences because acts such as ``reqalts`` and ``deny``
  can also contain a new value.
  """
  del act
  if method == "semantic":
    if embedder is None:
      embedder = FrozenEmbeddingEncoder()

    return extract_with_semantic(utterance, embedder, ontology)
  
  else:
    return extract_with_levenshtein(utterance, ontology)





def cosine_similarity(candidate_embedding, ontology_embedding):
    dot_product = np.dot(candidate_embedding, ontology_embedding)

    #magnitude of vectors
    candidate_length = np.linalg.norm(candidate_embedding)
    ontology_length = np.linalg.norm(ontology_embedding)

    #creates it in degrees
    cosine_similarity = dot_product / (candidate_length * ontology_length)

    return cosine_similarity





def semantic_candidates(utterance):
  words = utterance.lower().strip().split()
  candidates = []

  for phrase_size in (1, 2, 3):
    number_of_phrases = len(words) - phrase_size + 1

    for start in range(number_of_phrases):
      end = start + phrase_size
      phrase_words = words[start:end]

      if all(word in STOPWORDS for word in phrase_words):
        continue

      phrase = " ".join(phrase_words)
      candidates.append(phrase)

  return candidates



def closest_semantic_value(utterance, ontology_values, embedder, threshold: float = 0.7):

  candidates = semantic_candidates(utterance)

  #little safeguard
  if len(candidates) == 0:
    return None
  
  ontology_values_list = list(ontology_values)

  #combine the candiates and ontology values to send everything through distilbert in one call
  texts_to_encode = candidates + ontology_values_list
  embeddings = embedder.encode(texts_to_encode)

  number_of_candidates = len(candidates)

  #retrieve candidate and ontology embeddings
  candidate_embeddings = embeddings[:number_of_candidates]
  ontology_embeddings = embeddings[number_of_candidates:]

  #current best, not possible best
  best_similarity = -1
  best_candidate = None
  best_ontology_value = None

  for candidate_text, candidate_embedding in zip(candidates, candidate_embeddings):
    for ontology_value, ontology_embedding in zip(ontology_values_list, ontology_embeddings):

      similarity = cosine_similarity(candidate_embedding, ontology_embedding)

      if ontology_value in ["cheap","moderate","expensive"]:
        print(f"candidate text: {candidate_text}, ontology value: {ontology_value}, similarity: {similarity}")

      if similarity > best_similarity:
        best_similarity = similarity
        best_candidate = candidate_text
        best_ontology_value = ontology_value
  print(f"best candidate: {best_candidate}, best ontology value: {best_ontology_value} best similarity: {best_similarity}")
  
  if best_similarity < threshold:
    print("best similarity rejected")
    return None

  return Suggestion(best_candidate, best_ontology_value)




def extract_with_semantic(utterance, embedder, ontology: dict[str, tuple[str, ...]] | None = None, threshold: float = 0.7) -> tuple[RestaurantInfo, dict[str, Suggestion]]:

  values = ontology or load_ontology()
  keyword_matches = extract_keywords(utterance, values)

  suggestions: dict[str, Suggestion] = {}

  for slot in ("area", "food", "pricerange"):
    #If exact keyword matching already found the slot, skip the semantic matching
    if slot in keyword_matches:
      continue

    suggestion = closest_semantic_value(utterance, values[slot], embedder, threshold)

    #if the suggestion is not below the threshold
    if suggestion is not None:
      suggestions[slot] = suggestion

  return keyword_matches, suggestions


class SemanticEncoder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, texts):
        return self.model.encode(texts)


if __name__ == "__main__":
  embedder = SemanticEncoder()

  test_sentences = [
    "I want somewhere fancy",
    "I want something inexpensive",
    "I want chinese food",
    "I want a restaurant in the centre",
    "I want somewhere cheap",
  ]

#   test_sentences = [
#     "fancy",
#     "inexpensive",
# ]

  for sentence in test_sentences:
    exact, suggestions = extract_with_semantic(sentence, embedder)

    print("\nSentence:", sentence)
    print("Exact:", exact)
    print("Suggestions:", suggestions)