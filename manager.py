from enum import Enum

from classifiers import Act, Classifier, RuleClassifier
from data import clean_utterance
from responses import PROMPTS
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
  SUGGEST_REST = 8
  NO_REST = 9
  INFORM_REST = 10
  FINISHED = 11


class Suggestion:
  inp: str
  suggest: str

  def __init__(self, inp: str, suggest: str):
    self.inp = inp
    self.suggest = suggest

  def __str__(self) -> str:
    return f"({self.inp}, {self.suggest})"

  def __bool__(self) -> bool:
    # If Suggestion contains no suggest, it is 'failed' and should be falsy
    return self.suggest != ""

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
  # Anything that cannot be corrected is added as a failed Suggestion(original, "")
  else:
    errors["area"] = Suggestion(area, "")
  # Food
  food = input("food: ").strip().lower()
  if len(food) > 0:
    result["food"] = food
  else:
    errors["food"] = Suggestion(food, "")
  # Pricerange
  price = input("price: ").strip().lower()
  if price in ["cheap", "moderate", "expensive"]:
    result["pricerange"] = price
  elif price == "ceap":
    errors["pricerange"] = Suggestion(price, "cheap")
  else:
    errors["pricerange"] = Suggestion(price, "")
  # Return definitely correct results and errors/suggestions
  return result, errors


class Manager:
  # Manages the entire dialog flow, started with manager.run()
  classifier: Classifier
  state: State
  preferences: RestaurantInfo
  errors: dict[str, Suggestion]
  prompt: str
  current = None  # the restaurant we are talking about
  shown = None  # names we already suggested in this search

  def __init__(self, classifier: Classifier):
    self.classifier = classifier
    self.state = State.WELCOME
    self.preferences = RestaurantInfo()
    self.errors = {}
    self.prompt = ""
    

  def _update_preferences(self, act: Act, utterance: str):
    # Run slot extraction with the user's last utterance
    # Update preferences and suggestions (where applicable)
    new_prefs, errors = slot_extraction(act, utterance)
    self.preferences.update(new_prefs)
    for key, err in errors.items():
      if err:
        self.errors[key] = err

  def _from_welcome(self, act: Act, utterance: str) -> tuple[State, str]:
    # Each state (rectangle in the diagram) has a _from_<state> function
    # that takes the speech act and utterance as parameters
    # and returns the next state and the message to the user

    # If the user did not provide us with area information, ask for that
    if not (self.preferences.get("area") or self.errors.get("area")):
      return State.AREA_ASK, PROMPTS["area_ask"]
    # If they did, (this is the triangle pointing down), continue with the next check
    return self._from_area_ask(act, utterance)

  def _from_area_ask(self, act: Act, utterance: str) -> tuple[State, str]:
    # Asking area did not yield a result with 100% certainty - check with user or ask again
    if not self.preferences.get("area"):
      if self.errors.get("area"):
        # Got a suggestion
        print(f"  \033[92mDEBUG: errors={repr(self.errors)}\033[0m")
        return State.AREA_CONFIRM, PROMPTS["ask_confirm"].format(*self.errors["area"])
      # Got no suggesion
      return State.AREA_ASK, PROMPTS["area_ask_invalid"].format(*self.errors["area"])
    # If the user did not provide us with food information, ask for that
    if not (self.preferences.get("food") or self.errors.get("food")):
      return State.FOOD_ASK, PROMPTS["food_ask"]
    # If they did, continue with the next check
    return self._from_food_ask(act, utterance)

  def _from_area_confirm(self, act: Act, utterance: str) -> tuple[State, str]:
    if act in [Act.AFFIRM, Act.ACK]:
      # User agrees with the change
      self.preferences["area"] = self.errors["area"].suggest
    elif not (self.preferences.get("area") or self.errors.get("area")):
      # User did not agree and did not provide an alternative
      return State.AREA_ASK, PROMPTS["area_ask"]
    # User did not agree but have provided an alternative
    return self._from_area_ask(act, utterance)

  def _from_food_ask(self, act: Act, utterance: str) -> tuple[State, str]:
    # Functions identically to area_ask, but for food type
    if not self.preferences.get("food"):
      if self.errors.get("food"):
        return State.FOOD_CONFIRM, PROMPTS["ask_confirm"].format(*self.errors["food"])
      return State.FOOD_ASK, PROMPTS["food_ask_invalid"].format(*self.errors["food"])
    if not (self.preferences.get("pricerange") or self.errors.get("pricerange")):
      return State.PRICE_ASK, PROMPTS["price_ask"]
    return self._from_price_ask(act, utterance)

  def _from_food_confirm(self, act: Act, utterance: str) -> tuple[State, str]:
    # Functions identically to area_confirm, but for food type
    if act in [Act.AFFIRM, Act.ACK]:
      self.preferences["food"] = self.errors["food"].suggest
    elif not (self.preferences.get("food") or self.errors.get("food")):
      return State.FOOD_ASK, PROMPTS["food_ask"]
    return self._from_price_ask(act, utterance)

  def _from_price_ask(self, act: Act, utterance: str) -> tuple[State, str]:
    # If no valid pricerange is entered
    if not self.preferences.get("pricerange"):
      if self.errors.get("pricerange"):
        return State.PRICE_CONFIRM, PROMPTS["ask_confirm"].format(*self.errors["pricerange"])
      return State.PRICE_ASK, PROMPTS["price_ask_invalid"].format(*self.errors["pricerange"])
    # All preference information gathered, find suitable restaurants
    rest = load_restaurants()
    options = lookup_restaurant(rest, self.preferences, max_dist=-1)
    print(f"\033[93m{options}\033[0m")

    #aangepast want stel er wordt niks gevonden dan
    # dan valt er ook niks voor te stellen
    if len(options) == 0:
      self.options = []
      return State.NO_REST, PROMPTS["no_rest"]
    # eenmaal schudden en de hele lijst bewaren, reqalts loopt hier later doorheen
    self.options = options.sample(frac=1).to_dict("records")
    self.option_index = 0
    response = PROMPTS["suggest"].format(self.options[0]["restaurantname"])
    return State.SUGGEST_REST, response
    

  def _from_price_confirm(self, act: Act, utterance: str) -> tuple[State, str]:
    # Functions identically to area_confirm, but for pricerange
    if act in [Act.AFFIRM, Act.ACK]:
      self.preferences["pricerange"] = self.errors["pricerange"].suggest
    elif not (self.preferences.get("pricerange") or self.errors.get("pricerange")):
      return State.PRICE_ASK, PROMPTS["price_ask"]
    return self._from_price_ask(act, utterance)

  def _lookup(self) -> list[RestaurantInfo]:
    # the same search as in _from_price_ask, but as a plain list we can walk through
    options = lookup_restaurant(load_restaurants(), self.preferences, max_dist=-1)
    return options.to_dict("records")

  def _start_search(self, act: Act, utterance: str) -> tuple[State, str]:
    # forget what we showed and let _from_price_ask suggest a restaurant again
    self.current = None
    self.shown = None
    if not self._lookup():
      return State.NO_REST, PROMPTS["no_rest"]
    return self._from_price_ask(act, utterance)

  def _current_restaurant(self) -> RestaurantInfo:
    # _from_price_ask does not store which restaurant it picked, but its name
    # is in the last thing we said, so find it there the first time. This looks
    # at all restaurants, because the preferences may already have changed
    if self.current is None:
      everyone = load_restaurants().to_dict("records")
      named = [r for r in everyone if r["restaurantname"] in self.prompt]
      self.current = max(named, key=lambda r: len(r["restaurantname"]))
      self.shown = {self.current["restaurantname"]}
    return self.current

  def _from_suggest_rest(self, act: Act, utterance: str) -> tuple[State, str]:
    current = self._current_restaurant()
    # user wants something else
    if act == Act.REQALTS:
      options = self._lookup()
      # the current one no longer fits, so the user changed a preference
      if not any(o["restaurantname"] == current["restaurantname"] for o in options):
        return self._start_search(act, utterance)
      # otherwise the next one we have not shown yet
      unseen = [o for o in options if o["restaurantname"] not in self.shown]
      if not unseen:
        return State.NO_REST, PROMPTS["no_more_rest"]
      self.current = unseen[0]
      self.shown.add(self.current["restaurantname"])
      return State.SUGGEST_REST, PROMPTS["suggest"].format(self.current["restaurantname"])
    # user asks for details
    if act == Act.REQUEST:
      return self._from_inform_rest(act, utterance)
    if act in [Act.THANKYOU, Act.BYE]:
      return State.FINISHED, PROMPTS["goodbye"]
    #  say the suggestion again
    return State.SUGGEST_REST, PROMPTS["suggest"].format(current["restaurantname"])

  def _from_inform_rest(self, act: Act, utterance: str) -> tuple[State, str]:
    rest = self._current_restaurant()
    name = rest["restaurantname"]
    if act in [Act.THANKYOU, Act.BYE]:
      return State.FINISHED, PROMPTS["goodbye"]
    # the temporary slot extraction cannot tell what was asked, so look for words
    answers = []
    if "address" in utterance:
      answers.append(PROMPTS["info_addr"].format(name, rest["addr"]))
    if "phone" in utterance or "number" in utterance:
      answers.append(PROMPTS["info_phone"].format(name, rest["phone"]))
    if "post" in utterance:
      answers.append(PROMPTS["info_postcode"].format(name, rest["postcode"]))
    if "food" in utterance or "type" in utterance:
      answers.append(PROMPTS["info_food"].format(name, rest["food"]))
    # nothing recognised, so ask what they want to know
    if not answers:
      return State.INFORM_REST, PROMPTS["info_ask"].format(name)
    return State.INFORM_REST, " ".join(answers)

  def _from_no_rest(self, act: Act, utterance: str) -> tuple[State, str]:
    if act in [Act.THANKYOU, Act.BYE]:
      return State.FINISHED, PROMPTS["goodbye"]
    # new information can make restaurants available that we have not shown yet
    if act in [Act.INFORM, Act.REQALTS]:
      unseen = [o for o in self._lookup() if o["restaurantname"] not in (self.shown or set())]
      if unseen:
        return self._start_search(act, utterance)
    # nothing new, repeat the last message
    return State.NO_REST, self.prompt

  
  def transition_state(self, act: Act, utterance: str) -> tuple[State, str]:
    # Main state transition function
    print(
        f"  \033[92mDEBUG: {self.state.name}(act={act.value}, {utterance=})\033[0m")
    print(f"  \033[92mDEBUG: preferences{self.preferences}\033[0m")
    self._update_preferences(act, utterance)

    # Repeat handling. Since this is the same for all states, its check happens before the
    # _from_<state>() functions
    if act in [Act.NULL, Act.REPEAT]:
      return self.state, self.prompt

    # bye kan altijd, het maakt niet uit in welke state we zitten, dus dan altijd finishen
    if act == Act.BYE:
      return State.FINISHED, PROMPTS["goodbye"]

    # Request preferences. TODO: Implement the rest of the states and think about
    # speech-acts that influence the flow in ways that are not yet accounted for
    # (This match can be replaced with fancy python if we want to, but the
    #  professors may not like it)
    # Either have the state enum values be the functions and then `self.state(act, utterance)`
    # Or `getattr(self, f"_from_{self.state.name.lower()}")(act, utterance)`
    match self.state:
      case State.WELCOME: state, prompt = self._from_welcome(act, utterance)
      case State.AREA_ASK: state, prompt = self._from_area_ask(act, utterance)
      case State.AREA_CONFIRM: state, prompt = self._from_area_confirm(act, utterance)
      case State.FOOD_ASK: state, prompt = self._from_food_ask(act, utterance)
      case State.FOOD_CONFIRM: state, prompt = self._from_food_confirm(act, utterance)
      case State.PRICE_ASK: state, prompt = self._from_price_ask(act, utterance)
      case State.PRICE_CONFIRM: state, prompt = self._from_price_confirm(act, utterance)
      case State.SUGGEST_REST: state, prompt = self._from_suggest_rest(act, utterance)
      case State.INFORM_REST: state, prompt = self._from_inform_rest(act, utterance)
      case State.NO_REST: state, prompt = self._from_no_rest(act, utterance)
      case _: raise NotImplementedError()

    print(f"  \033[93mDEBUG: new preferences{self.preferences}\033[0m")
    return state, prompt

  def run(self):
    # Start with a welcome message, classify user input,
    # then walk through the states until finished.
    self.prompt = PROMPTS["welcome"]
    while self.state != State.FINISHED:
      inp = input(self.prompt + "\n> ")
      clean_inp = clean_utterance(inp)
      act = self.classifier.run(clean_inp)
      self.state, self.prompt = self.transition_state(act, clean_inp)
    print(self.prompt)


if __name__ == "__main__":
  # Quick testing function ran with `python manager.py`
  classifier = RuleClassifier()
  manager = Manager(classifier)
  manager.run()
