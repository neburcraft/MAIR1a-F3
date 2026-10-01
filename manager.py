from enum import Enum

from classifiers import Act, Classifier, RuleClassifier
from data import clean_utterance
from responses import PROMPTS
from restaurant import RestaurantInfo, load_restaurants, lookup_restaurant


class State(Enum):
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


class Suggest:
  inp: str
  suggest: str

  def __init__(self, inp: str, suggest: str):
    self.inp = inp
    self.suggest = suggest

  def __bool__(self) -> bool:
    return self.suggest != ""

  def __iter__(self):
    return iter([self.inp, self.suggest])


# TODO: replace with import to real slot extraction
def slot_extraction(act: Act, utterance: str) -> tuple[RestaurantInfo, dict[str, Suggest]]:
  errors: dict[str, Suggest] = {}
  result: RestaurantInfo = {}
  # Area
  area = input("area: ").strip().lower()
  if area in ["north", "east", "south", "west", "centre"]:
    result["area"] = area
  elif area == "center":
    errors["area"] = Suggest(area, "centre")
  elif len(area) > 0:
    errors["area"] = Suggest(area, "")
  # Food
  food = input("food: ").strip().lower()
  if len(food) > 0:
    result["food"] = food
  elif len(food) > 0:
    errors["food"] = Suggest(food, "")
  # Pricerange
  price = input("price: ").strip().lower()
  if price in ["cheap", "moderate", "expensive"]:
    result["pricerange"] = price
  elif len(price) > 0:
    errors["pricerange"] = Suggest(price, "")
  # Return
  return result, errors


class Manager:
  classifier: Classifier
  state: State
  preferences: RestaurantInfo
  errors: dict[str, Suggest]
  prompt: str

  def __init__(self, classifier: Classifier):
    self.classifier = classifier
    self.state = State.WELCOME
    self.preferences = RestaurantInfo()
    self.errors = {}
    self.prompt = ""

  def _update_preferences(self, act: Act, utterance: str):
    new_prefs, errors = slot_extraction(act, utterance)
    self.preferences.update(new_prefs)
    self.errors.update(errors)

  def _from_welcome(self, act: Act, utterance: str) -> tuple[State, str]:
    if not (self.preferences.get("area") or self.errors.get("area")):
      return State.AREA_ASK, PROMPTS["area_ask"]
    return self._from_area_ask(act, utterance)

  def _from_area_ask(self, act: Act, utterance: str) -> tuple[State, str]:
    # Asking area did not yield a result - ask again.
    if not self.preferences.get("area"):
      if self.errors.get("area"):
        print(f"  DEBUG: errors={repr(self.errors)}")
        return State.AREA_CONFIRM, PROMPTS["ask_confirm"].format(*self.errors["area"])
      return State.AREA_ASK, PROMPTS["area_ask_invalid"].format(*self.errors["area"])
    if not self.preferences.get("food"):
      return State.FOOD_ASK, PROMPTS["food_ask"]
    return self._from_food_ask(act, utterance)

  def _from_area_confirm(self, act: Act, utterance: str) -> tuple[State, str]:
    if act in [Act.AFFIRM, Act.ACK]:
      self.preferences["area"] = self.errors["area"].suggest
    elif not self.preferences.get("area"):
      return State.AREA_ASK, PROMPTS["area_ask"]
    if not self.preferences.get("food"):
      return State.FOOD_ASK, PROMPTS["food_ask"]
    return self._from_food_ask(act, utterance)

  def _from_food_ask(self, act: Act, utterance: str) -> tuple[State, str]:
    if not self.preferences.get("food"):
      if self.errors.get("food"):
        return State.FOOD_CONFIRM, PROMPTS["ask_confirm"].format(*self.errors["food"])
      return State.FOOD_ASK, PROMPTS["food_ask_invalid"].format(*self.errors["food"])
    if not self.preferences.get("pricerange"):
      return State.PRICE_ASK, PROMPTS["price_ask"]
    return self._from_price_ask(act, utterance)

  def _from_food_confirm(self, act: Act, utterance: str) -> tuple[State, str]:
    if act in [Act.AFFIRM, Act.ACK]:
      self.preferences["food"] = self.errors["food"].suggest
    elif not self.preferences.get("food"):
      return State.FOOD_ASK, PROMPTS["food_ask"]
    if not self.preferences.get("pricerange"):
      return State.FOOD_ASK, PROMPTS["price_ask"]
    return self._from_price_ask(act, utterance)

  def _from_price_ask(self, act: Act, utterance: str) -> tuple[State, str]:
    if not self.preferences.get("pricerange"):
      if self.errors.get("pricerange"):
        return State.PRICE_CONFIRM, PROMPTS["ask_confirm"].format(*self.errors["pricerange"])
      return State.PRICE_ASK, PROMPTS["price_ask_invalid"].format(*self.errors["pricerange"])
    # All preference information gathered, find a suitable restaurant
    # TODO: allow max_dist to be -1 (no max dist) for this
    rest = load_restaurants()
    options = lookup_restaurant(rest, self.preferences, max_dist=99)
    suggested: RestaurantInfo = options.sample(1).iloc[0].to_dict()
    response = PROMPTS["suggest"].format(suggested["restaurantname"])
    return State.SUGGEST_REST, response

  def _from_price_confirm(self, act: Act, utterance: str) -> tuple[State, str]:
    if act in [Act.AFFIRM, Act.ACK]:
      self.preferences["pricerange"] = self.errors["pricerange"].suggest
    elif not self.preferences.get("pricerange"):
      return State.PRICE_ASK, PROMPTS["price_ask"]
    return self._from_price_ask(act, utterance)

  def transition_state(self, act: Act, utterance: str) -> tuple[State, str]:
    self._update_preferences(act, utterance)
    print(f"  \033[92mDEBUG: state={self.state.name}(act={act.value}, {utterance=})\033[0m")
    print(f"  \033[92mDEBUG: preferences{self.preferences}\033[0m")

    # Repeat handling
    if act in [Act.NULL, Act.REPEAT]:
      return self.state, self.prompt

    # Request preferences
    match self.state:
      case State.WELCOME: state,prompt = self._from_welcome(act, utterance)
      case State.AREA_ASK: state,prompt = self._from_area_ask(act, utterance)
      case State.AREA_CONFIRM: state,prompt = self._from_area_confirm(act, utterance)
      case State.FOOD_ASK: state,prompt = self._from_food_ask(act, utterance)
      case State.FOOD_CONFIRM: state,prompt = self._from_food_confirm(act, utterance)
      case State.PRICE_ASK: state,prompt = self._from_price_ask(act, utterance)
      case State.PRICE_CONFIRM: state,prompt = self._from_price_confirm(act, utterance)
      case _:
        raise NotImplementedError()\

    print(f"  \033[93mDEBUG: new preferences{self.preferences}\033[0m")
    return state, prompt

  def run(self):
    self.prompt = PROMPTS["welcome"]
    while self.state != State.FINISHED:
      inp = input(self.prompt + "\n> ")
      clean_inp = clean_utterance(inp)
      act = self.classifier.run(clean_inp)
      self.state, self.prompt = self.transition_state(act, clean_inp)


if __name__ == "__main__":
  classifier = RuleClassifier()
  manager = Manager(classifier)
  manager.run()
