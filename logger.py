"""
Logs the dialog to a csv file in `logs/`, so the results can be analysed in part 2

One file per run: `logs/<session>_<participant>.csv`, with one row for every turn
The columns are explained in `docs/logging.md`
"""

import csv
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")

# The slots the manager keeps in self.preferences
SLOTS = ["area", "food", "pricerange", "additions"]

TURN_FIELDS = [
    "session_id", "participant_id", "condition", "turn", "state", "system_utterance",
    "user_utterance", "act", "area", "food", "pricerange", "additions", "restaurant"
]


class InteractionLogger:
  """One logger per session, so make a new one every time the program starts"""

  def __init__(self, participant_id="anonymous", condition="default", log_dir=LOG_DIR):
    self.participant_id = participant_id
    self.condition = condition

    # The session id is just the start time, which is unique enough here
    self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    self.turn = 0

    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    self.path = log_dir / f"{self.session_id}_{participant_id}.csv"

    # Start the file with only the header, rows are added later
    with open(self.path, "w", newline="", encoding="utf-8") as f:
      csv.DictWriter(f, fieldnames=TURN_FIELDS).writeheader()

  def log_turn(self, state, system_utterance, user_utterance, act, preferences, restaurant=""):
    """Adds one row. It is written straight away, so a crash does not lose the log."""
    self.turn += 1

    row = {
        "session_id": self.session_id,
        "participant_id": self.participant_id,
        "condition": self.condition,
        "turn": self.turn,
        "state": state,
        "system_utterance": system_utterance,
        "user_utterance": user_utterance,
        "act": act,
        "restaurant": restaurant,
    }

    # One column per slot, and an empty cell when it is not known yet
    for slot in SLOTS:
      value = preferences.get(slot) if preferences else None
      row[slot] = "" if value is None else value

    with open(self.path, "a", newline="", encoding="utf-8") as f:
      csv.DictWriter(f, fieldnames=TURN_FIELDS).writerow(row)
