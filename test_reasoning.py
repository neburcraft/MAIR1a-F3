import unittest

import pandas as pd

from reasoning import (
  apply_rules,
  derived_values,
  explain_recommendation,
  extract_additional_requirements,
  filter_candidates,
)


class ReasoningTests(unittest.TestCase):
  def test_all_relevant_rules_fire(self):
    restaurant = {
      "restaurantname": "example",
      "pricerange": "cheap",
      "food quality": "good",
      "food": "romanian",
      "crowdedness": "busy",
      "length of stay": "long",
    }
    rules = apply_rules(restaurant)
    self.assertEqual({rule.rule_id for rule in rules}, {1, 2, 3, 4, 5, 6})
    self.assertEqual(derived_values(rules)["romantic"], {True, False})

  def test_contradictory_candidate_is_excluded(self):
    candidates = pd.DataFrame([
      {
        "restaurantname": "contradictory",
        "crowdedness": "busy",
        "length of stay": "long",
      },
      {
        "restaurantname": "romantic option",
        "crowdedness": "not busy",
        "length of stay": "long",
      },
    ])
    result = filter_candidates(candidates, {"romantic": True})
    self.assertEqual(result["restaurantname"].tolist(), ["romantic option"])

  def test_extracts_positive_and_negative_requirements(self):
    result = extract_additional_requirements(
      "I want somewhere romantic but without children"
    )
    self.assertEqual(result, {"children": False, "romantic": True})

  def test_explanation_mentions_reason(self):
    restaurant = {
      "restaurantname": "example",
      "crowdedness": "not busy",
      "length of stay": "long",
    }
    explanation = explain_recommendation(restaurant, {"romantic": True})
    self.assertIn("romantic", explanation)
    self.assertIn("long stay", explanation)


if __name__ == "__main__":
  unittest.main()
