from typing import TypedDict

import pandas as pd

RestaurantInfo = TypedDict('RestaurantInfo', {
      "restaurantname": str,
      "pricerange": str,
      "area": str,
      "food": str,
      "phone": str,
      "addr": str,
      "postcode": str,
      "food quality": str,
      "crowdedness": str,
      "length of stay": str,
      "additions": str
    }, total=False
)


def load_restaurants(path: str = "restaurant_info_extended.csv") -> pd.DataFrame:
  """Small helper function to load all restaurants defined in `path` to a pandas DataFrame"""
  return pd.read_csv(path)


def lookup_restaurant(restaurants: pd.DataFrame,
            requirements: RestaurantInfo,
            max_dist: int = 0) -> pd.DataFrame:
  """
  Return a DataFrame with all restaurants that match the requirements
  or differ on no more than `max_dist` keys (negative to allow any distance)
  """
  restaurants['dist'] = 0
  for key, req in requirements.items():
    if key == "additions" or req == "dontcare":
      continue
    # Add 1 (boolean True) to each restaurant that doesn't match perfectly
    restaurants['dist'] += restaurants[key] != req

  # A negative `max_dist` allows any distance
  if max_dist >= 0:
    restaurants = restaurants[restaurants['dist'] <= max_dist]
  # Return only the ones that match the best (all minimum distance results)
  return restaurants[restaurants['dist'] == restaurants['dist'].min()]
