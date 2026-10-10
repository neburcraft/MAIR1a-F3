"""Logs the dialog to csv files in logs/, so the results can be analysed in part 2.

Two files per run:
  logs/<session>_<participant>.csv  one row for every turn
  logs/sessions.csv                 one row for every session
The columns are explained in docs/logging.md.
"""

import atexit
import csv
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")

# The slots the manager keeps in self.preferences
SLOTS = ["area", "food", "pricerange", "additions"]

TURN_FIELDS = [
  "session_id", "participant_id", "condition", "turn", "timestamp",
  "seconds_since_prev", "state", "system_utterance", "user_utterance",
  "act", "area", "food", "pricerange", "additions", "restaurant",
]

SESSION_FIELDS = [
  "session_id", "participant_id", "condition", "start", "turns",
  "duration_seconds", "got_recommendation", "restaurant",
]


class InteractionLogger:
  """One logger per session, so make a new one every time the program starts."""

  def __init__(self, participant_id="anonymous", condition="default", log_dir=LOG_DIR):
    self.participant_id = participant_id
    self.condition = condition

    # The session id is just the start time, which is unique enough here
    self.start_time = datetime.now()
    self.session_id = self.start_time.strftime("%Y%m%d-%H%M%S")
    self.start = time.time()
    self.prev = self.start

    self.turn = 0
    self.restaurant = ""
    self.closed = False

    self.dir = Path(log_dir)
    self.dir.mkdir(exist_ok=True)
    self.path = self.dir / f"{self.session_id}_{participant_id}.csv"

    # Start the turn file with only the header, rows are added later
    with open(self.path, "w", newline="", encoding="utf-8") as f:
      csv.DictWriter(f, fieldnames=TURN_FIELDS).writeheader()

    # Makes sure the session row is written, even if the user quits with ctrl-c
    atexit.register(self.close)

  def log_turn(self, state, system_utterance, user_utterance, act, preferences, restaurant=""):
    """Adds one row. It is written straight away, so a crash does not lose the log."""
    now = time.time()
    self.turn += 1
    if restaurant:
      self.restaurant = restaurant

    row = {
      "session_id": self.session_id,
      "participant_id": self.participant_id,
      "condition": self.condition,
      "turn": self.turn,
      "timestamp": datetime.now().isoformat(timespec="seconds"),
      "seconds_since_prev": round(now - self.prev, 1),
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

    self.prev = now
    with open(self.path, "a", newline="", encoding="utf-8") as f:
      csv.DictWriter(f, fieldnames=TURN_FIELDS).writerow(row)

  def close(self):
    """Adds the summary row of this session to sessions.csv. Only runs once."""
    if self.closed:
      return
    self.closed = True

    summary_path = self.dir / "sessions.csv"
    is_new = not summary_path.exists()

    row = {
      "session_id": self.session_id,
      "participant_id": self.participant_id,
      "condition": self.condition,
      "start": self.start_time.isoformat(timespec="seconds"),
      "turns": self.turn,
      "duration_seconds": round(time.time() - self.start, 1),
      "got_recommendation": self.restaurant != "",
      "restaurant": self.restaurant,
    }

    with open(summary_path, "a", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=SESSION_FIELDS)
      if is_new:
        writer.writeheader()
      writer.writerow(row)

