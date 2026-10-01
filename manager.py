from enum import Enum

from classifiers import Act, Classifier, RuleClassifier
from data import clean_utterance
from responses import PROMPTS
from restaurant import RestaurantInfo, load_restaurants, lookup_restaurant


class State(Enum):
  ERROR = 0
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


# TODO: replace with import to real slot extraction
def slot_extraction(act: Act, utterance: str) -> tuple[RestaurantInfo, dict[str, tuple[str, str]]]:
  errors: dict[str, tuple[str, str]] = {}
  result: RestaurantInfo = {}
  # Area
  area = input("area: ").strip().lower()
  if area in ["north", "east", "south", "west", "centre"]:
    result["area"] = area
  elif area == "center":
    errors["area"] = (area, "centre")
  else:
    errors["area"] = (area, "")
  # Food
  food = input("food: ").strip().lower()
  if len(food) > 0:
    result["food"] = food
  else:
    errors["food"] = (food, "")
  # Pricerange
  price = input("price: ").strip().lower()
  if price in ["cheap", "moderate", "expensive"]:
    result["pricerange"] = price
  else:
    errors["pricerange"] = (price, "")
  # Return
  return result, errors


class Manager:
  classifier: Classifier
  state: State
  preferences: RestaurantInfo
  prompt: str

  def __init__(self, classifier: Classifier):
    self.classifier = classifier
    self.state = State.WELCOME
    self.preferences = RestaurantInfo()
    self.prompt = ""

  def _update_preferences(self, act: Act, utterance: str) -> dict[str, tuple[str, str]]:
    new_prefs, errors = slot_extraction(act, utterance)
    self.preferences.update(new_prefs)
    return errors

  def transition_state(self, act: Act, utterance: str) -> tuple[State, str]:
    print(f"  DEBUG: {act=}, {utterance=}, {self.state=}")

    # Repeat handling
    if self.state == State.WELCOME and act in [Act.NULL, Act.REPEAT, Act.REQMORE]:
      return State.WELCOME, PROMPTS["welcome"]
    if act in [Act.NULL, Act.REPEAT]:
      return self.state, self.prompt

    errors = self._update_preferences(act, utterance)
    print(f"  DEBUG: preferences{self.preferences}")
    # Request preferences
    if self.state == State.WELCOME and not self.preferences.get("area"):
      return State.AREA_ASK, PROMPTS["area_ask"]
    # Asking area did not yield a result - ask again.
    if self.state == State.AREA_ASK and not self.preferences.get("area"):
      if errors.get("area",('',''))[1]:
        print(repr(errors))
        return State.AREA_CONFIRM, PROMPTS["ask_confirm"].format(*errors["area"])
      return State.AREA_ASK, PROMPTS["area_ask_invalid"].format(*errors["area"])
    if self.state == State.AREA_CONFIRM:
      if act in [Act.AFFIRM, Act.ACK]:
        self.preferences["area"] = errors["area"][1]
      elif not self.preferences.get("area"):
        return State.AREA_ASK, PROMPTS["area_ask"]
    # Food unknown
    if self.state in [State.WELCOME, State.AREA_ASK] and not self.preferences.get("food"):
      return State.FOOD_ASK, PROMPTS["food_ask"]
    if self.state == State.FOOD_ASK and not self.preferences.get("food"):
      if errors.get("food",('',''))[1]:
        return State.FOOD_CONFIRM, PROMPTS["ask_confirm"].format(*errors["food"])
      return State.FOOD_ASK, PROMPTS["food_ask_invalid"].format(*errors["food"])
    if self.state == State.FOOD_CONFIRM:
      if act in [Act.AFFIRM, Act.ACK]:
        self.preferences["food"] = errors["food"][1]
      elif not self.preferences.get("food"):
        return State.FOOD_ASK, PROMPTS["food_ask"]
    # Pricerange unknown
    if self.state in [State.WELCOME, State.AREA_ASK, State.FOOD_ASK] and not self.preferences.get("pricerange"):
      return State.PRICE_ASK, PROMPTS["price_ask"]
    if self.state == State.PRICE_ASK and not self.preferences.get("pricerange"):
      if errors.get("pricerange",('',''))[1]:
        return State.PRICE_CONFIRM, PROMPTS["ask_confirm"].format(*errors["pricerange"])
      return State.PRICE_ASK, PROMPTS["price_ask_invalid"].format(*errors["pricerange"])
    if self.state == State.PRICE_CONFIRM:
      if act in [Act.AFFIRM, Act.ACK]:
        self.preferences["pricerange"] = errors["pricerange"][1]
      elif not self.preferences.get("pricerange"):
        return State.PRICE_ASK, PROMPTS["price_ask"]

    # All preferences are known
    if self.state in [State.WELCOME, State.AREA_ASK, State.AREA_CONFIRM, State.FOOD_ASK, State.FOOD_CONFIRM, State.PRICE_ASK, State.PRICE_CONFIRM]:
      # TODO: allow max_dist to be -1 (no max dist) for this
      options = lookup_restaurant(
          load_restaurants(), self.preferences, max_dist=99)
      suggested: RestaurantInfo = options.sample(
          1).iloc[0].to_dict()  # type: ignore
      response = PROMPTS["suggest"].format(
          suggested["restaurantname"])  # type: ignore
      return State.SUGGEST_REST, response

    return State.ERROR, f"State not implemented (from={self.state}, act={act})"

  def run(self):
    self.prompt = PROMPTS["welcome"]
    while self.state not in [State.ERROR, State.FINISHED]:
      inp = input(self.prompt + "\n> ")
      clean_inp = clean_utterance(inp)
      act = self.classifier.run(clean_inp)
      self.state, self.prompt = self.transition_state(act, clean_inp)


if __name__ == "__main__":
  classifier = RuleClassifier()
  manager = Manager(classifier)
  manager.run()
