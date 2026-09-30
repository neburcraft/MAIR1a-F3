from restaurant import RestaurantInfo, load_restaurants, lookup_restaurant

if __name__=="__main__":
  query: RestaurantInfo = {"pricerange": "cheap", "area": "north", "food": "chinese", "length of stay": "long"}
  print(lookup_restaurant(load_restaurants(), query, max_dist=1)) 