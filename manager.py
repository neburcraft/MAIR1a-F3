from enum import Enum
from typing import Optional
import argparse

import pandas as pd

from classifiers import Act, Classifier, RuleClassifier
from data import clean_utterance
from reasoning import extract_additional_requirements, explain_recommendation, filter_candidates
from responses import Prompts
from restaurant import RestaurantInfo, load_restaurants, lookup_restaurant
from slot_extraction import Suggestion, extract_requested_fields, slot_extraction
from tts import TTS


class State(Enum):
  WELCOME = 1
  AREA_ASK = 2
  AREA_CONFIRM = 3
  FOOD_ASK = 4
  FOOD_CONFIRM = 5
  PRICE_ASK = 6
  PRICE_CONFIRM = 7
  ADDITIONS_ASK = 8
  ADDITIONS_CONFIRM = 9
  SUGGEST_REST = 10
  NO_REST = 11
  INFORM_REST = 12
  FINISHED = 13


class Manager:
  """Manage the state of one restaurant recommendation dialogue."""

  def __init__(self, classifier: Classifier, tts=None):
    self.classifier = classifier
    self.state = State.WELCOME
    self.preferences = RestaurantInfo()
    self.suggestions: dict[str, Suggestion] = {}
    self.additional_requirements: dict[str, bool] = {}
    self.requested_fields: list[str] = []
    self.prompt = ""
    self.candidates = load_restaurants()
    self.explanation = ""
    self.current: Optional[RestaurantInfo] = None
    self.shown: set[str] = set()
    self.max_dist = 0
    self.tts = tts

  def _send_pompt(self, prompt) -> tuple[Act, str]:
    self.prompt = prompt
    if self.tts:
      self.tts.say(prompt)
    else:
      print(prompt)
    clean_input = clean_utterance(input("> "))
    return self.classifier.run(clean_input), clean_input

  def _expected_slot(self) -> str | None:
    if self.state in (State.AREA_ASK, State.AREA_CONFIRM):
      return "area"
    if self.state in (State.FOOD_ASK, State.FOOD_CONFIRM):
      return "food"
    if self.state in (State.PRICE_ASK, State.PRICE_CONFIRM):
      return "pricerange"
    return None

  def _update_preferences(self, act: Act, utterance: str, suggest: bool = True):
    """Update normal slots, additional requirements and requested information."""
    new_preferences, new_suggestions = slot_extraction(
      act, utterance, expected_slot=self._expected_slot()
    )
    self.preferences.update(new_preferences)
    self.additional_requirements.update(extract_additional_requirements(utterance))
    self.requested_fields = extract_requested_fields(utterance)
    if suggest:
      self.suggestions.update(new_suggestions)

  def _remove_choice(self, source: str, key: str):
    if source == "preference":
      self.preferences.pop(key, None)
    else:
      self.additional_requirements.pop(key, None)

  def _resolve_conflict(
    self,
    first: tuple[str, str, str],
    second: tuple[str, str, str],
    fallback: State,
  ) -> Optional[State]:
    """Ask which of two conflicting requirements should be kept."""
    first_source, first_key, first_label = first
    second_source, second_key, second_label = second
    conflict_prompt = (
      f"You asked for both {first_label} and {second_label}, but these "
      "requirements conflict. Which one is more important?"
    )
    _, answer = self._send_pompt(conflict_prompt)

    first_words = set(first_label.split())
    second_words = set(second_label.split())
    answer_words = set(answer.split())
    if first_words & answer_words:
      self._remove_choice(second_source, second_key)
      return None
    if second_words & answer_words:
      self._remove_choice(first_source, first_key)
      return None

    # The answer was unclear. Remove both so the normal state can ask again.
    self._remove_choice(first_source, first_key)
    self._remove_choice(second_source, second_key)
    return fallback

  def _apply_reasoning(self) -> Optional[State]:
    """Apply the six inference rules and resolve incompatible preferences."""
    conflicts = []
    if self.additional_requirements.get("touristic") is True:
      price = self.preferences.get("pricerange")
      if price not in (None, "cheap", "dontcare"):
        conflicts.append((
          ("preference", "pricerange", price),
          ("additional", "touristic", "touristic"),
          State.PRICE_ASK,
        ))
      if self.preferences.get("food") == "romanian":
        conflicts.append((
          ("preference", "food", "Romanian food"),
          ("additional", "touristic", "touristic"),
          State.FOOD_ASK,
        ))
    if self.additional_requirements.get("romantic") is True:
      for other, label in (("children", "suitable for children"), ("assigned seats", "assigned seats")):
        if self.additional_requirements.get(other) is True:
          conflicts.append((
            ("additional", other, label),
            ("additional", "romantic", "romantic"),
            State.ADDITIONS_ASK,
          ))

    for first, second, fallback in conflicts:
      next_state = self._resolve_conflict(first, second, fallback)
      if next_state is not None:
        return next_state

    # Start again from all restaurants, then apply the derived requirements.
    self.candidates = filter_candidates(load_restaurants(), self.additional_requirements)
    self.explanation = ""
    return None

  def restart(self) -> State:
    self.__init__(self.classifier, self.tts)
    return State.WELCOME

  def _welcome(self) -> State:
    act, utterance = self._send_pompt(Prompts.welcome)
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return State.AREA_ASK

  def _area_ask(self, repeat=False) -> State:
    if self.preferences.get("area"): return State.FOOD_ASK
    if self.suggestions.get("area"): return State.AREA_CONFIRM
    act, utterance = self._send_pompt(Prompts.area_ask_invalid if repeat else Prompts.area_ask)
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._area_ask(repeat=True)

  def _area_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["area"]))
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance, suggest=False)
    if act in (Act.AFFIRM, Act.ACK, Act.CONFIRM) and "area" not in self.preferences:
      self.preferences["area"] = self.suggestions["area"].suggest
    self.suggestions.pop("area", None)
    return self._area_ask(repeat=True)

  def _food_ask(self, repeat=False) -> State:
    if self.preferences.get("food"): return State.PRICE_ASK
    if self.suggestions.get("food"): return State.FOOD_CONFIRM
    act, utterance = self._send_pompt(Prompts.food_ask_invalid if repeat else Prompts.food_ask)
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._food_ask(repeat=True)

  def _food_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["food"]))
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance, suggest=False)
    if act in (Act.AFFIRM, Act.ACK, Act.CONFIRM) and "food" not in self.preferences:
      self.preferences["food"] = self.suggestions["food"].suggest
    self.suggestions.pop("food", None)
    return self._food_ask(repeat=True)

  def _price_ask(self, repeat=False) -> State:
    if self.preferences.get("pricerange"): return State.ADDITIONS_ASK
    if self.suggestions.get("pricerange"): return State.PRICE_CONFIRM
    act, utterance = self._send_pompt(Prompts.price_ask_invalid if repeat else Prompts.price_ask)
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._price_ask(repeat=True)

  def _price_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["pricerange"]))
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance, suggest=False)
    if act in (Act.AFFIRM, Act.ACK, Act.CONFIRM) and "pricerange" not in self.preferences:
      self.preferences["pricerange"] = self.suggestions["pricerange"].suggest
    self.suggestions.pop("pricerange", None)
    return self._price_ask(repeat=True)

  def _additions_ask(self, repeat=False) -> State:
    if self.additional_requirements:
      next_state = self._apply_reasoning()
      return next_state or State.SUGGEST_REST
    if self.suggestions.get("additions"): return State.ADDITIONS_CONFIRM
    act, utterance = self._send_pompt(Prompts.additions_ask_invalid if repeat else Prompts.additions_ask)
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    if act in (Act.DENY, Act.NEGATE) or utterance in ("no", "none", "no preference"):
      self.candidates = load_restaurants()
      return State.SUGGEST_REST
    return self._additions_ask(repeat=True)

  def _additions_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["additions"]))
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance, suggest=False)
    if act in (Act.AFFIRM, Act.ACK, Act.CONFIRM):
      addition = self.suggestions["additions"].suggest
      self.preferences["additions"] = addition
      self.additional_requirements[addition] = True
    self.suggestions.pop("additions", None)
    return self._additions_ask(repeat=True)

  def _suggest_rest(self, new=False) -> State:
    if self.current is None or new:
      normal_preferences = {
        key: value for key, value in self.preferences.items() if key != "additions"
      }
      candidates = lookup_restaurant(self.candidates, normal_preferences, self.max_dist)
      if len(candidates) == 0: return State.NO_REST
      self.current = candidates.sample(1).iloc[0].to_dict()
      self.candidates = self.candidates[
        self.candidates["restaurantname"] != self.current["restaurantname"]
      ]
      self.explanation = explain_recommendation(self.current, self.additional_requirements)
    prompt = Prompts.suggest_new if new else Prompts.suggest
    if self.explanation:
      prompt += " " + self.explanation
    act, utterance = self._send_pompt(prompt.format(self.current["restaurantname"]))
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act in (Act.THANKYOU, Act.BYE): return State.FINISHED
    if act == Act.REQALTS: return self._suggest_rest(new=True)
    if act == Act.REQUEST:
      self._update_preferences(act, utterance, suggest=False)
      return State.INFORM_REST
    return State.SUGGEST_REST

  def _information_prompt(self) -> str:
    fields = self.requested_fields or ["addr", "phone", "postcode", "food"]
    labels = {"addr": "address", "phone": "phone number", "postcode": "postcode", "food": "food"}
    parts = []
    for field in fields:
      value = self.current.get(field) if self.current else None
      if value is not None and not pd.isna(value):
        parts.append(f"the {labels[field]} is {value}")
    if not parts:
      return "Sorry, that information is not available."
    sentence = ", and ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."

  def _inform_rest(self) -> State:
    act, utterance = self._send_pompt(Prompts.inform.format(self._information_prompt()))
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act in (Act.THANKYOU, Act.BYE): return State.FINISHED
    if act == Act.REQUEST:
      self._update_preferences(act, utterance, suggest=False)
    return State.INFORM_REST

  def _no_rest(self) -> State:
    new_preferences = []
    act, utterance = self._send_pompt(Prompts.no_restaurant)
    if act in (Act.REPEAT, Act.NULL): return self.state
    if act == Act.RESTART: return self.restart()
    if act in (Act.THANKYOU, Act.BYE): return State.FINISHED
    if act in (Act.INFORM, Act.REQALTS):
      new_preferences, new_suggestions = slot_extraction(
        act, utterance, expected_slot=self._expected_slot()
      )
      self.preferences.update(new_preferences)
      for slot, state in (
        ("area", State.AREA_CONFIRM), ("food", State.FOOD_CONFIRM),
        ("pricerange", State.PRICE_CONFIRM), ("additions", State.ADDITIONS_CONFIRM),
      ):
        if slot in new_suggestions:
          self.suggestions.update(new_suggestions)
          return state
    if len(new_preferences) == 0:
      self.max_dist += 1
    return State.SUGGEST_REST

  def finish(self):
    print(Prompts.goodbye)

  def transition_state(self):
    match self.state:
      case State.WELCOME: self.state = self._welcome()
      case State.AREA_ASK: self.state = self._area_ask()
      case State.AREA_CONFIRM: self.state = self._area_confirm()
      case State.FOOD_ASK: self.state = self._food_ask()
      case State.FOOD_CONFIRM: self.state = self._food_confirm()
      case State.PRICE_ASK: self.state = self._price_ask()
      case State.PRICE_CONFIRM: self.state = self._price_confirm()
      case State.ADDITIONS_ASK: self.state = self._additions_ask()
      case State.ADDITIONS_CONFIRM: self.state = self._additions_confirm()
      case State.SUGGEST_REST: self.state = self._suggest_rest()
      case State.INFORM_REST: self.state = self._inform_rest()
      case State.NO_REST: self.state = self._no_rest()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tts", action="store_true", help="Use text-to-speech instead of text output")
  args = parser.parse_args()
  manager = Manager(RuleClassifier(), TTS() if args.tts else None)
  while manager.state != State.FINISHED:
    manager.transition_state()
  manager.finish()
