
PROMPTS = {
    "welcome": "Welcome to restaurantpicker9000! What kind of restaurant are you looking for?",

    # the three questions to find a restaurant
    "area_ask": "What part of town would you prefer?",
    "price_ask": "What price range are you looking for?",
    "food_ask": "What kind of food would you like?",
    
    # we can't yell what the user meant
    "area_ask_invalid": "'{0}' is not a valid area, choose north, east, south, west or centre. What area do you prefer?",
    "food_ask_invalid": "Sorry, I am not familiar with '{0}'. What kind of food would you like?",
    "price_ask_invalid": "'{0}' is not a valid price, choose cheap, moderate or expensive. What price range are you looking for?",
    "ask_confirm": "Sorry I don't understand. Did you mean '{1}' instead of '{0}'?",

    # the recommendation 
    "suggest": "I suggest {0}.",

    # no restaurant available
    "no_rest": "I could not find a restaurant that matches. Would you like to change the area, food or price range?",
    "no_more_rest": "I'm afraid that there are no other restaurants that match. Would you like to change the area, food or price range?",

    #give info about the restaurant
    "info_ask": "What would you like to know about {0}? I can give the address, phone number, post code or type of food.",
    "info_addr": "{0} is located at {1}.",
    "info_phone": "The phone number of {0} is {1}.",
    "info_postcode": "The post code of {0} is {1}.",
    "info_food": "{0} serves {1} food.",
    
    "goodbye": "Goodbye, enjoy your meal!",
}
