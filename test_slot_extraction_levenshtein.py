# NOTE: See note on test files from albert-slot-keyword
import unittest

from slot_extraction import extract_with_levenshtein, load_ontology


class LevenshteinExtractionTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ontology = load_ontology()

  def test_exact_keywords_are_still_used_first(self):
    result, suggestions = extract_with_levenshtein(
      "cheap Chinese food in the north", self.ontology
    )
    self.assertEqual(result, {
      "area": "north",
      "food": "chinese",
      "pricerange": "cheap",
    })
    self.assertEqual(suggestions, {})

  def test_suggests_food_and_area_typos(self):
    result, suggestions = extract_with_levenshtein(
      "I want Spenish food in the nort", self.ontology
    )
    self.assertEqual(result, {})
    self.assertEqual(suggestions["food"].inp, "spenish")
    self.assertEqual(suggestions["food"].suggest, "spanish")
    self.assertEqual(suggestions["area"].suggest, "north")

  def test_suggestion_is_not_committed_before_confirmation(self):
    result, suggestions = extract_with_levenshtein(
      "A moderat restaurant please", self.ontology
    )
    self.assertNotIn("pricerange", result)
    self.assertEqual(suggestions["pricerange"].suggest, "moderate")

  def test_rejects_distant_words(self):
    result, suggestions = extract_with_levenshtein(
      "I want something fancy near the station", self.ontology
    )
    self.assertEqual(result, {})
    self.assertEqual(suggestions, {})


if __name__ == "__main__":
  unittest.main()
