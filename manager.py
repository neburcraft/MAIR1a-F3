from enum import Enum
from typing import Optional
import pandas as pd

from classifiers import Act, Classifier, RuleClassifier
from data import clean_utterance
from reasoning import extract_additional_requirements, explain_recommendation, filter_candidates
from responses import Prompts
from restaurant import RestaurantInfo, load_restaurants, lookup_restaurant


class State(Enum):
  # Added confirm states to check if a Suggestion is what the user intended
  # See their state manager functions for more information
  # ERROR = 0
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


class Suggestion:
  inp: str
  suggest: str

  def __init__(self, inp: str, suggest: str):
    self.inp = inp
    self.suggest = suggest

  def __str__(self) -> str:
    return f"({self.inp}, {self.suggest})"

  def __iter__(self):
    # Allows using the splat operator in str.format(*suggestion)
    return iter([self.inp, self.suggest])


# TODO: replace with import to real slot extraction
# This version asks for the input to the slots literally (like a form) after the real user input
def slot_extraction(act: Act, utterance: str) -> tuple[RestaurantInfo, dict[str, Suggestion]]:
  errors: dict[str, Suggestion] = {}
  result: RestaurantInfo = {}
  # Area
  area = input("area: ").strip().lower()
  # Slot extraction should place anything that is certainly correct in result
  if area in ["north", "east", "south", "west", "centre"]:
    result["area"] = area
  # Anything that is misspelled and corrected is added as a Suggestion(original, suggestion)
  elif area == "center":
    errors["area"] = Suggestion(area, "centre")
  # Food
  food = input("food: ").strip().lower()
  if len(food) > 0:
    result["food"] = food
  # Pricerange
  price = input("price: ").strip().lower()
  if price in ["cheap", "moderate", "expensive"]:
    result["pricerange"] = price
  elif price == "ceap":
    errors["pricerange"] = Suggestion(price, "cheap")
  # Additional requirements
  additions = input("additional requirements: ").strip().lower()
  if additions in ["touristic", "assigned seats", "children", "romantic"]:
    result["additions"] = additions
  # Return definitely correct results and errors/suggestions
  return result, errors


class Manager:
  # Manages the entire dialog flow, started with manager.run()
  classifier: Classifier
  state: State
  preferences: RestaurantInfo
  suggestions: dict[str, Suggestion]
  prompt: str
  candidates: pd.DataFrame
  explanation: str
  current: Optional[RestaurantInfo]
  shown: set[str]
  max_dist: int

  def __init__(self, classifier: Classifier):
    self.classifier = classifier
    self.state = State.WELCOME
    self.preferences = RestaurantInfo()
    self.suggestions = {}
    self.additional_requirements = {}
    self.prompt = ""
    self.candidates = load_restaurants()
    self.explanation = ""
    self.current = None
    self.shown = set()
    self.max_dist = 0

  def _send_pompt(self, prompt) -> tuple[Act, str]:
      self.prompt = prompt
      inp = input(self.prompt + "\n> ")
      clean_inp = clean_utterance(inp)
      act = self.classifier.run(clean_inp)
      return act, clean_inp
  
  def _update_preferences(self, act: Act, utterance: str, suggest: bool = True):
    # Run slot extraction with the user's last utterance
    # Update preferences and suggestions (where applicable)
    new_prefs, new_suggests = slot_extraction(act, utterance)
    self.preferences.update(new_prefs)
    if suggest:
      self.suggestions.update(new_suggests)

  def _apply_reasoning(self):
    # TODO: update self.preferences according to additional requirements
    # Also deal with conflicting preferences - this means removing both and self.explanation and going back to the thing the user prefers
    # so, if price=expensive and additions=touristic, ask which is more important and apply the newly entered one
    # i.e. if touristic filter on pricerange==cheap, quality==good, food!=romanian
    # self.candidates = self.candidates[self.candidates["pricerange"] == "cheap"]
    # Should also set self.explanation to the explanation
    pass

  def _welcome(self) -> State:
    act, utterance = self._send_pompt(Prompts.welcome)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return State.AREA_ASK

  def _area_ask(self, repeat=False) -> State:
    # TODO: for each preference, handle 'no preference' case
    if self.preferences.get("area"):
      return State.FOOD_ASK
    if self.suggestions.get("area"):
      return State.AREA_CONFIRM
    act, utterance = self._send_pompt(Prompts.area_ask_invalid if repeat else Prompts.area_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._area_ask(repeat=True)

  def _area_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["area"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance, suggest=False)
    if act in [Act.AFFIRM, Act.ACK] and "area" not in self.preferences:
      self.preferences["area"] = self.suggestions["area"].suggest
    return self._area_ask(repeat=True)

  def _food_ask(self, repeat=False) -> State:
    if self.preferences.get("food"):
      return State.PRICE_ASK
    if self.suggestions.get("food"):
      return State.FOOD_CONFIRM
    act, utterance = self._send_pompt(Prompts.food_ask_invalid if repeat else Prompts.food_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._food_ask(repeat=True)

  def _food_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["food"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance, suggest=False)
    if act in [Act.AFFIRM, Act.ACK] and "food" not in self.preferences:
      self.preferences["food"] = self.suggestions["food"].suggest
    return self._food_ask(repeat=True)

  def _price_ask(self, repeat=False) -> State:
    if self.preferences.get("pricerange"):
      return State.ADDITIONS_ASK
    if self.suggestions.get("pricerange"):
      return State.PRICE_CONFIRM
    act, utterance = self._send_pompt(Prompts.price_ask_invalid if repeat else Prompts.price_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._price_ask(repeat=True)
  
  def _price_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["pricerange"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance, suggest=False)
    if act in [Act.AFFIRM, Act.ACK] and "pricerange" not in self.preferences:
      self.preferences["pricerange"] = self.suggestions["pricerange"].suggest
    return self._price_ask(repeat=True)

  def _additions_ask(self, repeat=False) -> State:
    if self.preferences.get("additions"):
      self._apply_reasoning()
      return State.SUGGEST_REST
    if self.suggestions.get("additions"):
      return State.ADDITIONS_CONFIRM
    act, utterance = self._send_pompt(Prompts.additions_ask_invalid if repeat else Prompts.additions_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    if act in [Act.DENY, Act.NEGATE]:
      return State.SUGGEST_REST
    return self._additions_ask(repeat=True)

  def _additions_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["additions"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance, suggest=False)
    if act in [Act.AFFIRM, Act.ACK] and "additions" not in self.preferences:
      self.preferences["additions"] = self.suggestions["additions"].suggest
    return self._additions_ask(repeat=True)

  def _suggest_rest(self, new=False) -> State:
    if self.current is None or new:
      candidates = lookup_restaurant(self.candidates, self.preferences, self.max_dist)
      if len(candidates) == 0:
        return State.NO_REST
      self.current = candidates.sample(1).iloc[0].to_dict()
      # Remove chosen restaurant from the list so the next time, the user gets a new choice
      self.candidates = self.candidates[self.candidates["restaurantname"] != self.current["restaurantname"]]
    prompt = Prompts.suggest_new if new else Prompts.suggest
    if self.explanation:
      prompt += " " + self.explanation
    act, utterance = self._send_pompt(prompt.format(self.current["restaurantname"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    # TODO: Deal with when the user wants to change something 'I want a cheap restaurant instead"
    if act == Act.REQALTS:
      return self._suggest_rest(new=True)
    if act == Act.REQUEST:
      return State.INFORM_REST
    return State.SUGGEST_REST

  def _inform_rest(self) -> State:
    # TODO: add recognition for 'address', 'phone', 'post', 'food' in slot extraction
    # TODO: add correct language formatting for prompt
    prompt = Prompts.inform.format(self.current)
    act, utterance = self._send_pompt(prompt)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    return State.INFORM_REST

  def _no_rest(self) -> State:
    new_prefs = []
    act, utterance = self._send_pompt(Prompts.no_restaurant)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    # User changes their preferences
    if act in [Act.INFORM, Act.REQALTS]:
      new_prefs, new_suggests = slot_extraction(act, utterance)
      self.preferences.update(new_prefs)
      if "area" in new_suggests:
        return State.AREA_CONFIRM
      if "food" in new_suggests:
        return State.FOOD_CONFIRM
      if "pricerange" in new_suggests:
        return State.PRICE_CONFIRM
      if "additions" in new_suggests:
        return State.ADDITIONS_CONFIRM
    # If user did not change their preferences, increase allowed distance
    if len(new_prefs) == 0:
      self.max_dist += 1
    return State.SUGGEST_REST

  def finish(self):
    print(Prompts.goodbye)
    exit()
  
  def transition_state(self):
    # Main state transition function
    print(f"  \033[92mDEBUG: preferences{self.preferences}\033[0m")

    # Request preferences. TODO: Implement the rest of the states and think about
    # speech-acts that influence the flow in ways that are not yet accounted for
    # (This match can be replaced with fancy python if we want to, but the
    #  professors may not like it)
    # Either have the state enum values be the functions and then `self.state(act, utterance)`
    # Or `getattr(self, f"_from_{self.state.name.lower()}")(act, utterance)`
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

    print(f"  \033[93mDEBUG: new preferences{self.preferences}\033[0m")


if __name__ == "__main__":
  # Quick testing function ran with `python manager2.py`
  classifier = RuleClassifier()
  manager = Manager(classifier)
  while manager.state != State.FINISHED:
    manager.transition_state()
  manager.finish()
