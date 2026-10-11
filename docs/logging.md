# Log format

Logging is switched on with a participant id:

    python main.py --participant p01 --condition baseline

Without `--participant` nothing is logged. `--condition` is a free text label for the version of the system, for example `baseline` or `tts`. 

Each run writes one csv file in the folder `logs/`

## Log file

File: `logs/<session_id>_<participant_id>.csv`. One row for every turn. A turn is one system prompt and the reply of the user.

| Column                            | Meaning                                                                                                       |
| ---                               | ---                                                                                                           |
| session_id                        | Start time of the run (`YYYYMMDD-HHMMSS`)                                                                     |
| participant_id                    | Value of `--participant`                                                                                      |
| condition                         | Value of `--condition`                                                                                        |
| turn                              | Turn number, starting at 1                                                                                    |
| state                             | Dialog state of the system prompt (name in the `State` enum)                                                  |
| system_utterance                  | What the system said                                                                                          |
| user_utterance                    | What the user typed, after cleaning                                                                           |
| act                               | Dialog act predicted by the classifier                                                                        |
| area, food, pricerange, additions | Preferences known when the prompt was given. This is before the reply of this turn. Empty means not known yet |
| restaurant                        | Recommended restaurant, empty until one is suggested                                                          |


See `docs/example_log.csv` for a full example log
