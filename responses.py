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
  area_ask_invalid_old = "I’m afraid that isn’t one of the areas I know. Would you prefer north, east, south, west or the centre?
  area_ask_invalid_young = "I don't know that area. Do you mean north, east, south, west or centre?"
  food_ask_old = "What kind of food would you like?"
  food_ask_young = "What sort of food do you want?"
  food_ask_invalid_old = "I’m afraid that this foodtype isn't one i know. What kind of food would you prefer?"
  food_ask_invalid_young = "I don't understand. What kind of food do you mean?"
  price_ask_old = "Do you have a particular price range in mind?”
  price_ask_young = "What price do you want?"
  price_ask_invalid_old = "Sorry, I don’t recognise that price range. Could you choose between cheap, moderate or expensive?”
  price_ask_invalid_young = "What do you mean? Cheap, moderate or expensive?"
  additions_ask_old = "Is there anything else you would like the restaurant to have, such as being touristic, having assigned seats, being child-friendly or romantic? Feel free to say no if not."
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
  conflict_old = "I understand that you'd prefer both {0} and {1}. Unfortunately, these two preferences don't quite work together. May I ask which one matters more to you?"
  conflict_young = "Looks like {0} and {1} don't really go together. Which one would you rather go for?"

  # Explanations for additional requirements from the user, (`"": ""` added to make indexing easier)
  explanations_old = {
      "touristic": "It's a lovely place for visitors, offering good-quality food at reasonable prices.",
      "assigned seats": "Guests are kindly shown to their seats, especially during busy hours.",
      "children": "It's a family-friendly restaurant where guests can enjoy a pleasant meal without spending too much time.",
      "romantic": "It's a peaceful and charming restaurant, perfect for enjoying a relaxed dinner together.",
      "": ""
  }
  explanations_young = {
      "touristic": "It's a great spot for tourists because the food is tasty and affordable.",
      "assigned seats": "They assign seats because the place gets pretty busy.",
      "children": "It's a good place for families with kids because you can grab a quick bite.",
      "romantic": "It's a nice place for a date because it's quiet and you can take your time.",
      "": ""
  }
