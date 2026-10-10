# Log format

Logging is switched on with a participant id:

    python main.py --participant p01 --condition baseline

Without `--participant` nothing is logged. `--condition` is a free text label for the version of the system, for example `baseline` or `tts`. 

Each run writes two csv files in the folder `logs/` (utf-8, comma separated, with a header row).

## 1. Turn log

File: `logs/<session_id>_<participant_id>.csv`. One row for every turn. A turn is one system prompt and the reply of the user.

| Column | Meaning |
|---|---|
| session_id | Start time of the run |
| participant_id | Value of `--participant` |
| condition | Value of `--condition` |
| turn | Turn number, starting at 1 |
| timestamp | Time of the reply |
| seconds_since_prev | Seconds since the previous reply |
| state | Dialog state of the system prompt (name in the `State` enum) |
| system_utterance | What the system said |
| user_utterance | What the user typed|
| act | Dialog act predicted by the classifier |
| area, food, pricerange, additions | Preferences known when the prompt was given. This is before the reply of this turn. Empty means not known yet |
| restaurant | Recommended restaurant, empty until one is suggested |

## 2. Session log

File: `logs/sessions.csv`. One row for every session, added when the program stops. All sessions share this file.

| Column | Meaning |
|---|---|
| session_id, participant_id, condition | Same as above |
| start | Start time (ISO 8601) |
| turns | Number of turns |
| duration_seconds | Length of the session |
| got_recommendation | True if a restaurant was suggested |
| restaurant | Last suggested restaurant |


## 3. Example

This example comes from a scripted session, so the times are not realistic. The full turn log is in `docs/example_log.csv`.

    turn,state,system_utterance,user_utterance,act,area,pricerange,restaurant
    1,WELCOME,"Hello, welcome to the restaurant system. How can I help you?",i want a cheap restaurant,INFORM,,,
    2,AREA_ASK,In which area would you like to eat?,hmm,NULL,,cheap,
    3,AREA_ASK,In which area would you like to eat?,in the north,INFORM,,cheap,
    4,SUGGEST_REST,I suggest da vinci pizzeria.,thank you goodbye,BYE,north,cheap,da vinci pizzeria

A row in `sessions.csv` looks like this:

    session_id,participant_id,condition,start,turns,duration_seconds,got_recommendation,restaurant
    20261010-170504,p01,baseline,2026-10-10T17:05:04,4,0.1,True,da vinci pizzeria

## 4. Reading the logs

No extra scripting is needed:

    import glob, pandas as pd
    turns = pd.concat(pd.read_csv(f) for f in glob.glob("logs/2*.csv"))
    sessions = pd.read_csv("logs/sessions.csv")

    sessions.groupby("condition")["duration_seconds"].mean()
    turns.groupby("session_id")["act"].value_counts()
