# RUBEN: Improve
class Prompts:
  welcome = "Welcome to MAIR - Manager of Awesome Interesting Restaurants! What kind of restaurant are you looking for?"
  ask_confirm = "Did you mean '{1}' instead of '{0}'?"
  area_ask = "What area should the restaurant be located?"
  area_ask_invalid = "I don't recognize that as a valid area, choose north, east, south, west or centre. What area?"
  food_ask = "What kind of food would you like?"
  food_ask_invalid = "I don't recognize that as a valid food type. What kind of food would you like?"
  price_ask = "What is the price range that you are aiming for?"
  price_ask_invalid = "I don't recognize that as a valid price, choose cheap, moderate or expensive. What is the price range you are aiming for?"
  additions_ask = "Do you have an additional requirement: touristic, assigned seats, children or romantic? You can also say no."
  additions_ask_invalid = "I did not recognise an additional requirement. Please choose touristic, assigned seats, children or romantic, or say no."
  suggest = "I suggest {}."
  suggest_new = "Another good option is {}."
  no_restaurant = "There are no restaurants matching your preferences. Would you like to change something?"
  goodbye = "Thank you for using this system and goodbye!"
  inform_part = "the {0} is {1}"
  inform_none = "Sorry, that information is not available."
  inform = lambda r: f'{r["restaurantname"].capitalize()} is a{"n" if r["pricerange"] == "expensive" else ""} ' \
                     f'{r["pricerange"]} {r["food"]} restaurant in the {r["area"]}, {r["addr"]}. ' \
                     f'\nZIP code: {r["postcode"]}; phone: {r["phone"]}'.replace("nan", "unknown")
  conflict = "You asked for both {0} and {1}, but these requirements conflict. Which one is more important?" 

EXPLANATION = {
  "touristic": "It is touristic because it has cheap and good food.",
  "assigned seats": "It has assigned seats because it is a busy restaurant.",
  "children": "It is suitable for children because it is not a long stay.",
  "romantic": "It is romantic because it is not busy and allows for a long stay.",
  "": ""
}
