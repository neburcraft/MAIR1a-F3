"""Definition of all prompts the system sends to the user"""

class Prompts:
  style: str

  def __init__(self, style: str):
    self.style = style

  """All prompt format strings to print to the user"""
  def __getattr__(self, attr):
    return Prompts.__dict__[f"{attr}_{self.style}"]

  welcome_old = "Welcome to MAIR - Manager of Awesome Interesting Restaurants! What kind of restaurant are you looking for?"
  welcome_young = "Welcome to MAIR - Manager of Awesome Interesting Restaurants! What kind of restaurant do you want?"
  ask_confirm_old = "I’m sorry, I’m not sure I understood that correctly. Did you perhaps mean ‘{1}’ instead of ‘{0}’?”"
  ask_confirm_young = "Did you mean '{1}' instead of '{0}'?"
  area_ask_old = "Could you tell me which part of town you would prefer?"
  area_ask_young = "Where should the restaurant be?"
  area_ask_invalid_old = "I don't recognize that as a valid area. Please choose north, east, south, west or centre. What area?"
  area_ask_invalid_young = "I don't know that area. Do you mean north, east, south, west or centre?"
  food_ask_old = "What kind of food would you like?"
  food_ask_young = "What food do you want?"
  food_ask_invalid_old = "I don't recognize that as a valid food type. What kind of food would you like?"
  food_ask_invalid_young = "I don't understand. What kind of food"
  price_ask_old = "What is the price range that you are aiming for?"
  price_ask_young = "What price do you want?"
  price_ask_invalid_old = "I don't recognize that as a valid price. Please choose cheap, moderate or expensive. What is the price range you are aiming for?"
  price_ask_invalid_young = "What do you mean? Cheap, moderate or expensive?"
  additions_ask_old = "Do you have an additional requirement: touristic, assigned seats, children or romantic? You can also say no."
  additions_ask_young = "Anything else, like touristic, assigned seats, children or romantic?"
  additions_ask_invalid_old = "I did not recognise an additional requirement. Please choose touristic, assigned seats, children or romantic, or say no."
  additions_ask_invalid_young = "What do you mean? Touristic, assigned seats, children, romantic or nothing?"
  imperfect_prefix_old = "I'm sorry, I couldn't find an exact match, but "
  imperfect_prefix_young = "It isn't perfect, but "
  suggest_old = "{0} is a nice {1} {2} restaurant in the {3}."
  suggest_young = "{0} is a nice {1} {2} restaurant in the {3}."
  suggest_new_old = "Another good option is {0}: a nice {1} {2} restaurant in the {3}."
  suggest_new_young = "Maybe {0}: a nice {1} {2} restaurant in the {3}."
  no_restaurant_old = "Sorry, there are no restaurants matching {}. Would you like to change something?"
  no_restaurant_young = "I failed to find anything good. Do you want to change something?"
  goodbye_old = "Thank you for using this system and goodbye!"
  goodbye_young = "Thanks for using MAIR and bye!"
  inform_part_old = "the {0} is {1}"
  inform_part_young = "the {0} is {1}"
  inform_none_old = "Sorry, that information is not available."
  inform_none_young = "I have no clue."
  inform_old = lambda r: f'{r["restaurantname"].capitalize()} is a{"n" if r["pricerange"] == "expensive" else ""} ' \
      f'{r["pricerange"]} {r["food"]} restaurant in the {r["area"]}, {r["addr"]}. ' \
      f'\nZIP code: {r["postcode"]}; phone: {r["phone"]}'.replace("nan", "unknown")
  inform_young = lambda r: f'{r["restaurantname"].capitalize()} is a{"n" if r["pricerange"] == "expensive" else ""} ' \
      f'{r["pricerange"]} {r["food"]} restaurant in the {r["area"]}, {r["addr"]}. ' \
      f'\nZIP code: {r["postcode"]}; phone: {r["phone"]}'.replace("nan", "unknown")
  conflict_old = "You asked for both {0} and {1}, but these requirements conflict. Which one is more important?"
  conflict_young = "You can't have both {0} and {1}. Which one is more important?"

  # Explanations for additional requirements from the user, (`"": ""` added to make indexing easier)
  explanations_old = {
      "touristic": "It is touristic because it has cheap and good food.",
      "assigned seats": "It has assigned seats because it is a busy restaurant.",
      "children": "It is suitable for children because it is not a long stay.",
      "romantic": "It is romantic because it is not busy and allows for a long stay.",
      "": ""
  }
  explanations_young = {
      "touristic": "It is touristic because it has cheap and good food.",
      "assigned seats": "It has assigned seats because it is a busy restaurant.",
      "children": "It is suitable for children because it is not a long stay.",
      "romantic": "It is romantic because it is not busy and allows for a long stay.",
      "": ""
  }
