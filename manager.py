"""Full state manager implementation for part 1b of the assignment"""

from enum import Enum
from typing import Literal, Optional
import pandas as pd

from classifiers import Act, Classifier
from data import clean_utterance
from responses import Prompts
from restaurant import RestaurantInfo, load_restaurants, lookup_restaurant
from slot_extraction import Suggestion, SlotExtract
from tts import TTS
from logger import InteractionLogger


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
  """Manage the state of one restaurant recommendation dialogue"""
  classifier: Classifier
  extractor: SlotExtract
  verbose: bool
  prompts: Prompts
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
  explain: bool
  logger: InteractionLogger | None

  def __init__(self,
               classifier: Classifier,
               extractor: SlotExtract,
               verbose: bool = False,
               tts: TTS | None = None,
               explain: bool = False,
               logger: InteractionLogger | None = None,
               age_group: Literal["old", "young"] = "old"):
    self.classifier = classifier
    self.extractor = extractor
    self.verbose = verbose
    self.prompts = Prompts(age_group)
    self.state = State.WELCOME
    self.preferences = RestaurantInfo()
    self.suggestions = {}
    self.requested_fields = []
    self.prompt = ""
    self.candidates = load_restaurants()
    self.explanation = ""
    self.current = None
    self.shown = set()
    self.max_dist = 0
    self.tts = tts
    self.explain = explain
    self.logger = logger

  def _send_pompt(self, prompt: str) -> tuple[Act, str]:
    """
    Prints `prompt` to the console and waits for user input

    Then classify the cleaned input and return `(act, utterance)`
    """
    self.prompt = prompt
    # Use text-to-speech when `--tts` was used, otherwise print to console
    if self.tts:
      self.tts.say(prompt)
    else:
      print(prompt)

    try:
      clean_input = clean_utterance(input("> "))
    except EOFError:
      return Act.BYE, ""
    act = self.classifier.run(clean_input)
    if self.logger:
      restaurant = self.current["restaurantname"] if self.current else ""
      self.logger.log_turn(self.state.name, prompt, clean_input, act.name, dict(self.preferences), restaurant)
    if self.verbose:
      print(f"  \033[92mINFO (manager.py): Input classified as {act.name}\033[0m")
    return act, clean_input

  def _expected_slot(self) -> str | None:
    """Helper function to decide what `preferences` slot a 'dontcare' response should occupy"""
    if self.state in (State.AREA_ASK, State.AREA_CONFIRM):
      return "area"
    if self.state in (State.FOOD_ASK, State.FOOD_CONFIRM):
      return "food"
    if self.state in (State.PRICE_ASK, State.PRICE_CONFIRM):
      return "pricerange"
    return None

  def _update_preferences(self, act: Act, utterance: str, suggest: bool = True):
    """Update `preferences`, `suggestions` and `requested_fields` based on the `utterance`"""
    new_preferences, new_suggestions = self.extractor.slot_extraction(utterance, self._expected_slot())
    self.preferences.update(new_preferences)
    self.requested_fields = self.extractor.extract_requested_fields(utterance)
    if suggest:
      self.suggestions.update(new_suggestions)
    # Print debug output about new preferences if `--verbose` was used
    if self.verbose:
      print("\033[92m", end="")
      if self.preferences: print(f"  INFO (manager.py): {self.preferences=}")
      if self.suggestions: print(f"  INFO (manager.py): {self.suggestions=}")
      if self.requested_fields: print(f"  INFO (manager.py): {self.requested_fields=}")
      print("\033[0m", end="")

  def filter_candidates(self, restaurants: pd.DataFrame, addition: str | None) -> pd.DataFrame:
    """Filter lookup results by the user's additional requirements"""
    match addition:
      case "touristic":  # Cheap, good food that is not Romanian --> touristic
        cheap = restaurants[restaurants["pricerange"] == "cheap"]
        positive = cheap[cheap["food quality"] == "good"]
        return positive[positive["food"] != "romanian"]
      case "assigned seats":  # Crowded restaurant --> assigned seats
        return restaurants[restaurants["crowdedness"] == "busy"]
      case "children":  # Short stay --> suitable for children
        return restaurants[restaurants["length of stay"] != "long"]
      case "romantic":  # Not busy and long stay --> romantic
        not_busy = restaurants[restaurants["crowdedness"] != "busy"]
        return not_busy[not_busy["length of stay"] == "long"]
      case _:  # No additional requirements
        return restaurants

  def _resolve_conflict(self, key_a: str, label_a: str, key_b: str, label_b: str) -> bool:
    """
    Ask which of conflicting requirements should be kept; `label_a` or `label_b`

    Returns True if succeeded, False when it deleted both requirements
    """
    _, answer = self._send_pompt(self.prompts.conflict.format(label_a, label_b))
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
    """Resolve incompatible preferences and apply the inference rules"""
    # 'touristic' and 'romanian' are incompatible, try to resolve conflict, fall back to `FOOD_ASK`` otherwise
    if self.preferences.get("additions") == "touristic":
      if self.preferences.get("food") == "romanian":
        if self.verbose:
          print("  \033[92mINFO (manager.py): 'touristic' not compatible with 'romanian'\033[0m")
        if not self._resolve_conflict("food", "Romanian food", "additions", "touristic"):
          return State.FOOD_ASK

    # 'touristic' may no longer be in preferences after a `_resolve_conflict`, so repeat if statement
    # 'touristic' and 'expensive'/'moderate' are incompatible, try to resolve conflict, fall back to `PRICE_ASK`` otherwise
    if self.preferences.get("additions") == "touristic":
      price = self.preferences.get("pricerange", "dontcare")
      if price not in ["cheap", "dontcare"]:
        if self.verbose:
          print(f"  \033[92mINFO (manager.py): 'touristic' not compatible with '{price}'\033[0m")
        if not self._resolve_conflict("pricerange", price, "additions", "touristic"):
          return State.PRICE_ASK

    # You cannot have a conflict with the other additions, as they could only conflict
    # with each other and the user can only choose one addition at most
    # Start again from all restaurants, then apply the newly derived requirements
    self.candidates = self.filter_candidates(load_restaurants(),
                                             self.preferences.get("additions", None))
    self.explanation = self.prompts.explanations[self.preferences.get("additions", "")]
    return None

  def restart(self) -> State:
    """Fully re-initialize the manager to start over cleanly"""
    self.__init__(self.classifier, self.extractor, self.verbose, self.tts, self.explain, self.logger)
    if self.verbose:
      print(f"  \033[92mINFO (manager.py): {self.preferences=}")
      print(f"  INFO (manager.py): {self.suggestions=}")
      print(f"  INFO (manager.py): {self.requested_fields=}\033[0m")
    return State.WELCOME

  def _welcome(self) -> State:
    """
    Every state has its own `_state()` function that is run when the given state is active

    Structure is usually as follows:
    1. (Optional) preliminary checks for moving on automatically when information is already known
    2. (Optional) collecting any information necessary to generate the user prompt
    3. Send the user prompt and classify its speech-act
    4. (Optional) update user preferences, suggestions and requested fields
    5. Return the next state based on e.g. speech act
    """
    act, utterance = self._send_pompt(self.prompts.welcome)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return State.AREA_ASK

  def _area_ask(self, repeat=False) -> State:
    """
    Ask for preferred area if not already given, or go to confirm if area preference was unclear

    Note that the structure of all `_preference_ask()` and `_preference_confirm()` functions is very similar
    so they could easily have been combined, but we chose to leave them separate to more exactly match
    the state diagram.
    """
    # Area already known certainly (exact match)
    if self.preferences.get("area"): return State.FOOD_ASK
    # Area likely known, confirm with user (match with levenshtein or cosine similarity)
    if self.suggestions.get("area"): return State.AREA_CONFIRM
    # A repeated ask has a different phrasing (adds "I don't recognize that...")
    act, utterance = self._send_pompt(self.prompts.area_ask_invalid if repeat else self.prompts.area_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    # Recursive call until area is known, setting repeat to True to use the different phrasing if needed
    return self._area_ask(repeat=True)

  def _area_confirm(self) -> State:
    """Ask if the suggestion generated by levenshtein or similarity is what the user intended"""
    # Splat (*) operator used as a shorthand for `(suggestion.inp, suggestion.suggest)`
    act, utterance = self._send_pompt(self.prompts.ask_confirm.format(*self.suggestions["area"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.AFFIRM, Act.ACK] and "area" not in self.preferences:
      self.preferences["area"] = self.suggestions["area"].suggest
    # Suggestion is no longer necessary (either accepted or rejected), so we delete it
    del self.suggestions["area"]
    self._update_preferences(act, utterance, suggest=False)
    # Go back to the `_area_ask` execution path with repeat phrasing if needed
    self.state = State.AREA_ASK
    return self._area_ask(repeat=True)

  def _food_ask(self, repeat=False) -> State:
    """Ask for preferred food type if not already given, or go to confirm if food preference was unclear"""
    if self.preferences.get("food"): return State.PRICE_ASK
    if self.suggestions.get("food"): return State.FOOD_CONFIRM
    act, utterance = self._send_pompt(self.prompts.food_ask_invalid if repeat else self.prompts.food_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._food_ask(repeat=True)

  def _food_confirm(self) -> State:
    """Ask if the suggestion generated by levenshtein or similarity is what the user intended"""
    act, utterance = self._send_pompt(self.prompts.ask_confirm.format(*self.suggestions["food"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.AFFIRM, Act.ACK] and "food" not in self.preferences:
      self.preferences["food"] = self.suggestions["food"].suggest
    del self.suggestions["food"]
    self._update_preferences(act, utterance, suggest=False)
    self.state = State.FOOD_ASK
    return self._food_ask(repeat=True)

  def _price_ask(self, repeat=False) -> State:
    """Ask for preferred price range if not already given, or go to confirm if price preference was unclear"""
    if self.preferences.get("pricerange"): return State.ADDITIONS_ASK
    if self.suggestions.get("pricerange"): return State.PRICE_CONFIRM
    act, utterance = self._send_pompt(self.prompts.price_ask_invalid if repeat else self.prompts.price_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    self._update_preferences(act, utterance)
    return self._price_ask(repeat=True)

  def _price_confirm(self) -> State:
    """Ask if the suggestion generated by levenshtein or similarity is what the user intended"""
    act, utterance = self._send_pompt(self.prompts.ask_confirm.format(*self.suggestions["pricerange"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.AFFIRM, Act.ACK] and "pricerange" not in self.preferences:
      self.preferences["pricerange"] = self.suggestions["pricerange"].suggest
    del self.suggestions["pricerange"]
    self._update_preferences(act, utterance, suggest=False)
    self.state = State.PRICE_ASK
    return self._price_ask(repeat=True)

  def _additions_ask(self, repeat=False) -> State:
    """Ask for preferred additions if not already given, or go to confirm if additions preference was unclear"""
    if self.preferences.get("additions", ""):
      return self._apply_reasoning() or State.SUGGEST_REST
    if self.suggestions.get("additions"): return State.ADDITIONS_CONFIRM
    act, utterance = self._send_pompt(
        self.prompts.additions_ask_invalid if repeat else self.prompts.additions_ask)
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.DENY, Act.NEGATE]:
      return State.SUGGEST_REST
    self._update_preferences(act, utterance)
    return self._additions_ask(repeat=True)

  def _additions_confirm(self) -> State:
    """Ask if the suggestion generated by levenshtein or similarity is what the user intended"""
    act, utterance = self._send_pompt(self.prompts.ask_confirm.format(*self.suggestions["additions"]))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act == Act.BYE: return State.FINISHED
    if act in [Act.AFFIRM, Act.ACK]:
      addition = self.suggestions["additions"].suggest
      self.preferences["additions"] = addition
    del self.suggestions["additions"]
    self._update_preferences(act, utterance, suggest=False)
    self.state = State.ADDITIONS_ASK
    return self._additions_ask(repeat=True)

  def _restaurant_reqalts(self, utterance: str) -> tuple[State | None, bool]:
    """Update preferences when another restaurant is requested"""
    new_prefs, new_suggests = self.extractor.slot_extraction(utterance)
    self.preferences.update(new_prefs)
    if "additions" in new_prefs:
      return self._additions_ask(), True
    if "area" in new_suggests:
      return State.AREA_CONFIRM, True
    if "food" in new_suggests:
      return State.FOOD_CONFIRM, True
    if "pricerange" in new_suggests:
      return State.PRICE_CONFIRM, True
    if "additions" in new_suggests:
      return State.ADDITIONS_CONFIRM, True
    return None, bool(new_prefs)

  def _suggest_rest(self, new=False) -> State:
    """All preferences have been collected, suggest a restaurant matching the preferences"""
    if self.current is None or new:
      # Pick a new restaurant matching the preferences with maximally `max_dist` alterations
      candidates = lookup_restaurant(self.candidates, self.preferences, self.max_dist)
      # Go to `NO_REST` state if no options were found, else select a random one that was not shown already
      if len(candidates) == 0: return State.NO_REST
      self.current = candidates.sample(1).iloc[0].to_dict()
      self.shown.add(self.current["restaurantname"])
      self.candidates = self.candidates[~self.candidates["restaurantname"].isin(self.shown)]

    # Tell the user about the (new) restaurant with an explanation if an additional requirement was requested
    self.explanation = self.prompts.explanations[self.preferences.get("additions", "")]
    prompt = self.prompts.suggest_new if new else self.prompts.suggest
    if self.current["dist"] > 0:
      prompt = self.prompts.imperfect_prefix + prompt.lower()
    prompt = prompt.format(
        self.current["restaurantname"].capitalize(),
        self.current["pricerange"],
        self.current["food"],
        self.current["area"]
    )
    if self.explanation and self.explain:
      prompt += " " + self.explanation
    act, utterance = self._send_pompt(prompt)

    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    # User changes their preferences
    if act in [Act.INFORM, Act.REQALTS]:
      confirm, changed = self._restaurant_reqalts(utterance)
      if changed: self.current = None
      if confirm: return confirm
      if act == Act.REQALTS: return self._suggest_rest(new=True)
    if act == Act.REQUEST:
      # `_update_preferences` used here to update `requested_fields` used by `_information_prompt`
      self._update_preferences(act, utterance, suggest=False)
      return State.INFORM_REST
    return State.SUGGEST_REST

  def _information_prompt(self) -> str:
    """Generate a message containing the information the user requested (stored in `requested_fields`)"""
    # If not specified, put information in a nicely formatted message
    if not self.requested_fields:
      return self.prompts.inform(self.current)

    # Generate a message with the requested information otherwise
    fields = self.requested_fields or [
        "area", "addr", "phone", "postcode", "quality", "food", "pricerange"
    ]
    labels = {
        "area": "location",
        "addr": "address",
        "phone": "phone number",
        "postcode": "ZIP code",
        "food quality": "food quality",
        "food": "cuisine",
        "pricerange": "price"
    }
    parts = []
    for field in fields:
      value = self.current.get(field) if self.current else None
      # Skip unknown values
      if value is not None and not pd.isna(value):
        parts.append(self.prompts.inform_part.format(labels[field], value))
    if not parts:
      return self.prompts.inform_none
    return ", ".join(parts).capitalize() + "."

  def _inform_rest(self) -> State:
    """Gives the requested information to the user, or all if not specified"""
    act, utterance = self._send_pompt(self._information_prompt())
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    # User changes their preferences
    if act in [Act.INFORM, Act.REQALTS]:
      confirm, changed = self._restaurant_reqalts(utterance)
      if changed: self.current = None
      if confirm: return confirm
      if act == Act.REQALTS:
        self.state = State.SUGGEST_REST
        return self._suggest_rest(new=True)
    if act == Act.REQUEST:
      # `_update_preferences` used to update `requested_fields` again to answer the new question
      self._update_preferences(act, utterance, suggest=False)
    return State.INFORM_REST

  def _no_rest(self) -> State:
    """Tell the user no restaurant could be found and let them change preferences"""
    prefs = ", ".join(f"{key}={value}" for key,value in self.preferences.items())
    act, utterance = self._send_pompt(self.prompts.no_restaurant.format(prefs))
    if act in [Act.REPEAT, Act.NULL]: return self.state
    if act == Act.RESTART: return self.restart()
    if act in [Act.THANKYOU, Act.BYE]: return State.FINISHED
    self.current = None
    changed = False
    # User changes their preferences
    if act in [Act.INFORM, Act.REQALTS]:
      confirm, changed = self._restaurant_reqalts(utterance)
      if confirm: return confirm
    # If user did not change their preferences, increase allowed distance
    if not changed:
      self.max_dist += 1
    return State.SUGGEST_REST

  def finish(self):
    """Print the goodbye message"""
    if self.tts:
      self.tts.say(self.prompts.goodbye)
    else:
      print(self.prompts.goodbye)

  def transition_state(self):
    if self.verbose:
      print(f"  \033[92mINFO (manager.py): {self.state=}\033[0m")
    match self.state:
      case State.WELCOME:
        self.state = self._welcome()
      case State.AREA_ASK:
        self.state = self._area_ask()
      case State.AREA_CONFIRM:
        self.state = self._area_confirm()
      case State.FOOD_ASK:
        self.state = self._food_ask()
      case State.FOOD_CONFIRM:
        self.state = self._food_confirm()
      case State.PRICE_ASK:
        self.state = self._price_ask()
      case State.PRICE_CONFIRM:
        self.state = self._price_confirm()
      case State.ADDITIONS_ASK:
        self.state = self._additions_ask()
      case State.ADDITIONS_CONFIRM:
        self.state = self._additions_confirm()
      case State.SUGGEST_REST:
        self.state = self._suggest_rest()
      case State.INFORM_REST:
        self.state = self._inform_rest()
      case State.NO_REST:
        self.state = self._no_rest()
