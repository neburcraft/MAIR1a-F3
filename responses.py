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
  ask_confirm_old = "I'm sorry, I'm not sure I understood that correctly. Did you perhaps mean '{1}' instead of '{0}'?"
  ask_confirm_young = "Did you mean '{1}' instead of '{0}'?"
  area_ask_old = "Could you tell me which part of town you would prefer?"
  area_ask_young = "Where should the restaurant be?"
  area_ask_invalid_old = "I'm afraid that isn't one of the areas I know. Would you prefer north, east, south, west or the centre?"
  area_ask_invalid_young = "I don't know that area. Do you mean north, east, south, west or centre?"
  food_ask_old = "What kind of food are you looking for?"
  food_ask_young = "What sort of food do you want?"
  food_ask_invalid_old = "I'm afraid that this foodtype isn't one I know. What kind of food would you prefer?"
  food_ask_invalid_young = "I don't understand. What kind of food do you mean?"
  price_ask_old = "Do you have a particular price range in mind?"
  price_ask_young = "What price do you want?"
  price_ask_invalid_old = "I'm sorry, I didn't quite understand your preferred price range. Do you mind choosing from cheap, moderate, or expensive?"
  price_ask_invalid_young = "Oops, didn't quite catch your budget! Are you looking for something cheap, moderate, or expensive?"
  additions_ask_old = "Would you like the restaurant to have any additional features, such as being touristic, offering assigned seats, being suitable for children, or having a romantic atmosphere? It's perfectly fine if you have no other preferences."
  additions_ask_young = "Any other preferences? Maybe somewhere touristic, with assigned seats, good for children, or romantic? No worries if not!"
  additions_ask_invalid_old = "I'm sorry, I didn't quite understand your preference. Would you mind choosing from touristic, assigned seats, suitable for children, or romantic? You can also simply say 'no' if you don't have any additional preferences."
  additions_ask_invalid_young = "Oops, didn't quite catch that! Are you looking for somewhere touristic, with assigned seats, good for children, or romantic? Or just say 'no' if none of those matter!"
  imperfect_prefix_old = "I'm sorry, I couldn't find a restaurant that perfectly matches your preferences. However, I hope you might find this alternative suitable: "
  imperfect_prefix_young = "Couldn't find a perfect match, but here's something pretty close: "
  suggest_old = "I'd be happy to recommend {0}, a lovely {1} {2} restaurant in the {3}."
  suggest_young = "Check out {0}! It's a cool {1} {2} spot in the {3}."
  suggest_new_old = "If you'd like to consider another option, I would be happy to suggest {0}, a lovely {1} {2} restaurant in the {3}."
  suggest_new_young = "How about {0}? It's a pretty nice {1} {2} spot in the {3}!"
  no_restaurant_old = "I'm sorry, but I couldn't find any restaurants that match {}. Would you like to adjust your preferences so I can help you find another option?"
  no_restaurant_young = "No luck finding a match for {}! Wanna switch things up a bit?"
  goodbye_old = "Thank you very much for using our service. It has been a pleasure assisting you. Have a wonderful day!"
  goodbye_young = "Thanks for hanging out with MAIR! Hope you find somewhere awesome to eat. See ya!"
  inform_part_old = "the {0} is {1}"
  inform_part_young = "{0} is {1}"
  inform_none_old = "I'm very sorry, but I'm afraid that information isn't available at the moment."
  inform_none_young = "Oops, looks like I don't have that info right now!"
  
  inform_old = lambda r: (
    f"I'd be happy to share some details about {r['restaurantname'].capitalize()}. "
    f"It's a{'n' if r['pricerange'] == 'expensive' else ''} "
    f"{r['pricerange']} {r['food']} restaurant in the {r['area']}. "
    f"You can find it at {r['addr']}. "
    f"\nThe postcode is {r['postcode']}, and the phone number is {r['phone']}."
  ).replace("nan", "unknown")

  inform_young = lambda r: (
    f"Here's the scoop on {r['restaurantname'].capitalize()}! "
    f"It's a{'n' if r['pricerange'] == 'expensive' else ''} "
    f"{r['pricerange']} {r['food']} spot in the {r['area']}. "
    f"You'll find it at {r['addr']}. "
    f"\nPostcode: {r['postcode']} | Wanna get in touch? Call {r['phone']}!"
  ).replace("nan", "unknown")

  conflict_old = "I understand that you'd prefer both {0} and {1}. Unfortunately, these two preferences don't quite work together. May I ask which one matters more to you?"
  conflict_young = "Looks like {0} and {1} don't match. Which one would you rather go for?"

  # Explanations for additional requirements from the user, (`"": ""` added to make indexing easier)
  explanations_old = {
      "touristic": "It's a lovely place for visitors, offering quality food at reasonable prices.",
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
