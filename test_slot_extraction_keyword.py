# NOTE: See note on test files from albert-slot-keyword
import unittest

from slot_extraction import extract_keywords, load_ontology, slot_extraction


class KeywordExtractionTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ontology = load_ontology()

  def test_extracts_three_preferences_from_one_utterance(self):
    result = extract_keywords(
      "I want a cheap Italian restaurant in the west", self.ontology
    )
    self.assertEqual(result, {
      "area": "west",
      "food": "italian",
      "pricerange": "cheap",
    })

  def test_supports_common_area_and_price_phrases(self):
    result = extract_keywords(
      "A moderately priced restaurant in the city center please", self.ontology
    )
    self.assertEqual(result, {"area": "centre", "pricerange": "moderate"})

  def test_matches_multiword_food_type(self):
    result = extract_keywords("I would like modern European food", self.ontology)
    self.assertEqual(result, {"food": "modern european"})

  def test_does_not_match_substrings_or_unrelated_words(self):
    result = extract_keywords("I want something new near the station", self.ontology)
    self.assertEqual(result, {})

  def test_manager_wrapper_returns_no_confirmation_for_exact_match(self):
    result, suggestions = slot_extraction(
      "inform", "Chinese food in the north", self.ontology
    )
    self.assertEqual(result, {"area": "north", "food": "chinese"})
    self.assertEqual(suggestions, {})


if __name__ == "__main__":
  unittest.main()
