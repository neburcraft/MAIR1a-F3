from enum import Enum
from typing import Optional
import pandas as pd

from classifiers import Act, Classifier
from data import clean_utterance
from responses import Prompts, EXPLANATION
from restaurant import RestaurantInfo, load_restaurants, lookup_restaurant
from slot_extraction import Suggestion, SlotExtract
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
  classifier: Classifier
  extractor: SlotExtract
  verbose: bool
  state: State
  preferences: RestaurantInfo
  suggestions: dict[str, Suggestion]
  requested_fields: list[str]
  prompt: str
  candidates: pd.DataFrame
  explanation: str
  current: Optional[RestaurantInfo]
  shown: set[str]
  max_dist: int
  tts: TTS | None

  def __init__(self, classifier: Classifier, extractor: SlotExtract, verbose: bool = False, tts=None, logger=None):
    self.classifier = classifier
    self.extractor = extractor
    self.verbose = verbose
    self.state = State.WELCOME
    self.preferences = RestaurantInfo()
    self.suggestions= {}
    self.requested_fields = []
    self.prompt = ""
    self.candidates = load_restaurants()
    self.explanation = ""
    self.current = None
    self.shown = set()
    self.max_dist = 1
    self.tts = tts
    self.logger = logger

  def _send_pompt(self, prompt) -> tuple[Act, str]:
    self.prompt = prompt
    if self.tts:
      self.tts.say(prompt)
    else:
      print(prompt)
    clean_input = clean_utterance(input("> "))
    act = self.classifier.run(clean_input)
    if self.logger:
      restaurant = self.current["restaurantname"] if self.current else ""
      self.logger.log_turn(self.state.name, prompt, clean_input, act.name, dict(self.preferences), restaurant
    if self.verbose:
      print(f"  \033[92mINFO (manager.py): Input classified as {act.name}\033[0m")
    return act, clean_input

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
    new_preferences, new_suggestions = self.extractor.slot_extraction(
        act, utterance, self._expected_slot()
    )
    self.preferences.update(new_preferences)
    self.requested_fields = self.extractor.extract_requested_fields(utterance)
    if suggest:
      self.suggestions.update(new_suggestions)
    if self.verbose:
      print("\033[92m", end="")
      if self.preferences: print(f"  INFO (manager.py): {self.preferences=}")
      if self.suggestions: print(f"  INFO (manager.py): {self.suggestions=}")
      if self.requested_fields: print(f"  INFO (manager.py): {self.requested_fields=}")
      print("\033[0m", end="")

  def filter_candidates(self, restaurants: pd.DataFrame, addition: str | None) -> pd.DataFrame:
    """Filter lookup results by the user's additional requirements."""
    match addition:
      case "touristic": # Cheap, good food that is not Romanian --> touristic
        cheap = restaurants[restaurants["pricerange"] == "cheap"]
        positive = cheap[cheap["food quality"] == "good"]
        return positive[positive["food"] != "romanian"]
      case "assigned seats": # Crowded restaurant --> assigned seats
        return restaurants[restaurants["crowdedness"] == "busy"]
      case "children": # Short stay --> suitable for children
        return restaurants[restaurants["length of stay"] != "long"]
      case "romantic": # Not busy and long stay --> romantic
        not_busy = restaurants[restaurants["crowdedness"] != "busy"]
        return not_busy[not_busy["length of stay"] == "long"]
      case _: # No additional requirements
        return restaurants

  def _resolve_conflict(
    self,
    key_a: str, label_a: str,
    key_b: str, label_b: str
  ) -> bool:
    """Ask which of two conflicting requirements should be kept. Returns True if succeeded, False when it deleted both requirements."""
    _, answer = self._send_pompt(Prompts.conflict.format(label_a, label_b))
    if self.extractor.contains_phrase(answer, label_a):
      del self.preferences[key_b]
      return True
    if self.extractor.contains_phrase(answer, label_b):
      del self.preferences[key_a]
      return True

    # The answer was unclear. Remove both so the normal state can ask again.
    del self.preferences[key_a]
    del self.preferences[key_b]
    return False

  def _apply_reasoning(self) -> Optional[State]:
    """Apply the six inference rules and resolve incompatible preferences."""
    if self.preferences.get("additions") == "touristic":
      if self.preferences.get("food") == "romanian":
        if self.verbose:
          print("  \033[92mINFO (manager.py): 'touristic' not compatible with 'romanian'\033[0m")
        if not self._resolve_conflict("food", "Romanian food", "additions", "touristic"):
          return State.FOOD_ASK
    
    if self.preferences.get("additions") == "touristic":
      price = self.preferences.get("pricerange", "dontcare")
      if price not in ["cheap", "dontcare"]:
        if self.verbose:
          print(f"  \033[92mINFO (manager.py): 'touristic' not compatible with '{price}'\033[0m")
        if not self._resolve_conflict("pricerange", price, "additions", "touristic"):
          return State.PRICE_ASK
    # You cannot have a conflict with the other additions, as they could only conflict
    # with each other and the user can only choose one addition at most.

    # Start again from all restaurants, then apply the derived requirements.
    self.candidates = self.filter_candidates(load_restaurants(), self.preferences.get("additions", None))
    self.explanation = EXPLANATION[self.preferences.get("additions", "")]
    return None

  def restart(self) -> State:
    self.__init__(self.classifier, self.extractor, self.verbose, self.tts, self.logger)
    if self.verbose:
      print(f"  \033[92mINFO (manager.py): {self.preferences=}")
      print(f"  INFO (manager.py): {self.suggestions=}")
      print(f"  INFO (manager.py): {self.requested_fields=}\033[0m")
    return State.WELCOME

  def _welcome(self) -> State:
    act, utterance = self._send_pompt(Prompts.welcome)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return State.AREA_ASK

  def _area_ask(self, repeat=False) -> State:
    if self.preferences.get("area"): return State.FOOD_ASK
    if self.suggestions.get("area"): return State.AREA_CONFIRM
    act, utterance = self._send_pompt(Prompts.area_ask_invalid if repeat else Prompts.area_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._area_ask(repeat=True)

  def _area_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["area"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.AFFIRM, Act.ACK, Act.CONFIRM] and "area" not in self.preferences:
      self.preferences["area"] = self.suggestions["area"].suggest
    del self.suggestions["area"]
    self._update_preferences(act, utterance, suggest=False)
    return self._area_ask(repeat=True)

  def _food_ask(self, repeat=False) -> State:
    if self.preferences.get("food"): return State.PRICE_ASK
    if self.suggestions.get("food"): return State.FOOD_CONFIRM
    act, utterance = self._send_pompt(Prompts.food_ask_invalid if repeat else Prompts.food_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._food_ask(repeat=True)

  def _food_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["food"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.AFFIRM, Act.ACK, Act.CONFIRM] and "food" not in self.preferences:
      self.preferences["food"] = self.suggestions["food"].suggest
    del self.suggestions["food"]
    self._update_preferences(act, utterance, suggest=False)
    return self._food_ask(repeat=True)

  def _price_ask(self, repeat=False) -> State:
    if self.preferences.get("pricerange"): return State.ADDITIONS_ASK
    if self.suggestions.get("pricerange"): return State.PRICE_CONFIRM
    act, utterance = self._send_pompt(Prompts.price_ask_invalid if repeat else Prompts.price_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._price_ask(repeat=True)

  def _price_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["pricerange"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.AFFIRM, Act.ACK, Act.CONFIRM] and "pricerange" not in self.preferences:
      self.preferences["pricerange"] = self.suggestions["pricerange"].suggest
    del self.suggestions["pricerange"]
    self._update_preferences(act, utterance, suggest=False)
    return self._price_ask(repeat=True)

  def _additions_ask(self, repeat=False) -> State:
    if self.preferences.get("additions", ""):
      return self._apply_reasoning() or State.SUGGEST_REST
    if self.suggestions.get("additions"): return State.ADDITIONS_CONFIRM
    act, utterance = self._send_pompt(Prompts.additions_ask_invalid if repeat else Prompts.additions_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.DENY, Act.NEGATE]:
      return State.SUGGEST_REST
    self._update_preferences(act, utterance)
    return self._additions_ask(repeat=True)

  def _additions_confirm(self) -> State:
    act, utterance = self._send_pompt(Prompts.ask_confirm.format(*self.suggestions["additions"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.AFFIRM, Act.ACK, Act.CONFIRM]:
      addition = self.suggestions["additions"].suggest
      self.preferences["additions"] = addition
    del self.suggestions["additions"]
    self._update_preferences(act, utterance, suggest=False)
    return self._additions_ask(repeat=True)

  def _suggest_rest(self, new=False) -> State:
    if self.current is None or new:
      candidates = lookup_restaurant(self.candidates, self.preferences, self.max_dist)
      if len(candidates) == 0: return State.NO_REST
      self.current = candidates.sample(1).iloc[0].to_dict()
      self.candidates = self.candidates[
        self.candidates["restaurantname"] != self.current["restaurantname"]
      ]
    self.explanation = EXPLANATION[self.preferences.get("additions", "")]
    prompt = Prompts.suggest_new if new else Prompts.suggest
    if self.explanation:
      prompt += " " + self.explanation
    act, utterance = self._send_pompt(prompt.format(self.current["restaurantname"].capitalize()))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    if act == Act.REQALTS: return self._suggest_rest(new=True)
    if act == Act.REQUEST:
      self._update_preferences(act, utterance, suggest=False)
      return State.INFORM_REST
    return State.SUGGEST_REST

  def _information_prompt(self) -> str:
    if not self.requested_fields:
      return Prompts.inform(self.current)
    fields = self.requested_fields or ["area", "addr", "phone", "postcode", "quality", "food", "pricerange"]
    labels = {"area": "location", "addr": "address", "phone": "phone number", "postcode": "ZIP code", "food quality": "food quality", "food": "cuisine", "pricerange": "price"}
    parts = []
    for field in fields:
      value = self.current.get(field) if self.current else None
      if value is not None and not pd.isna(value):
        parts.append(Prompts.inform_part.format(labels[field], value))
    if not parts:
      return Prompts.inform_none
    return ", ".join(parts).capitalize() + "."

  def _inform_rest(self) -> State:
    act, utterance = self._send_pompt(self._information_prompt())
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    if act == Act.REQALTS: return self._suggest_rest(new=True)
    if act == Act.REQUEST:
      self._update_preferences(act, utterance, suggest=False)
    return State.INFORM_REST

  def _no_rest(self) -> State:
    new_prefs = []
    act, utterance = self._send_pompt(Prompts.no_restaurant)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    # User changes their preferences
    if act in [Act.INFORM, Act.REQALTS]:
      new_prefs, new_suggests = self.extractor.slot_extraction(act, utterance)
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
    if self.verbose:
      print(f"  \033[92mINFO (manager.py): {self.state=}\033[0m")
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
