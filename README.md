# INFOMAIR Project part 1
Implementation of the INFOMAIR project by group F3.

Includes from part 1a: Dialog act classification for the restaurant recommendation dialog system. Utterances from the DSTC 2 dataset are classified into one of 15 dialog acts, using a rule-based baseline and machine learning classifiers such as logistic regression and MLP. Each classifier was trained and evaluated on a stratified split and a grouped split that prevents duplicate utterances from leaking between train and test.

Includes from part 2a: Full restaurant recommendation engine with state manager, slot extraction and an optional text-to-speech mode. Slot extraction is implemented using keyword matching, Levenshtein distance and cosine similarity for word embeddings. Adds options for text-to-speech, explanation of choices and interaction logging, as well as a (debug) verbose mode.


## Group F3
- Anouk van Ladesteijn ([a.r.vanladesteijn@uu.nl](a.r.vanladesteijn@uu.nl))
- Huub Statius Muller ([h.b.statiusmuller@students.uu.nl](h.b.statiusmuller@students.uu.nl))
- Yanpeng Wang ([y.wang36@students.uu.nl](y.wang36@students.uu.nl))
- Ruben Wijmenga ([ruben.wijmenga@students.uu.nl](mailto:r.wijmenga@students.uu.nl))


## Installation
Mamba:
```
mamba env create -f env.yml
mamba activate MAIR1a-F3
```
Conda:
```
conda env create -f env.yml
conda activate MAIR1a-F3
```
Vanilla:
```
python -m pip install -r requirements.txt
```


## Project structure
### Code
- `classifiers.py`: all classifier implementations and the shared `Classifier` base class (`RuleClassifier`, `LRClassifier`, `MLPClassifier`, `FrozenEmbeddingEncoder`, `EmbeddedLRClassifier`, `EmbeddedMLPClassifier`)
- `data.py`: for loading, cleaning and splitting the dataset (stratified split and grouped split)
- `evaluate.py`: evaluates all classifiers and stores the result to ./experiments/
- `main.py`: contains the command line interface for the entire system
- `manager.py`: contains the full restaurant reccomendation pipeline
- `responses.py`: formatted natural language prompts to send to the user (either by tts or in the console)
- `restaurant.py`: helper functions for loading and filtering the restaurants defined in the csv files
- `slot_extraction.py`: functions for extracting user preference from an utterance using keyword matching, Levenshtein distance and cosine similarity
- `tts.py`: wrapper for the sherpa_onnx text-to-speech engine

### Data
- `dialog_acts.dat`: the training/development dataset
- `dialog_acts_train.dat`: the generated (fixed seed) training dataset from regular stratified sampling
- `dialog_acts_test.dat`: the generated (fixed seed) testing dataset from regular stratified sampling
- `dialog_acts_train_grouped.dat`: the generated (fixed seed) training dataset from grouped stratified sampling
- `dialog_acts_test_grouped.dat`: the generated (fixed seed) testing dataset from grouped stratified sampling
- `restaurant_info.csv`: the restaurant info dataset without reasoning information
- `restaurant_info_extended.csv`: the restaurant info dataset with reasoning information

### Miscellaneous
- `env.yml` for installing packages with mamba or conda
- `requirements.txt` for installing packages on a regular python installation


## Running
```
python main.py [-h] [--model {rule,lr,clean_lr,mlp,clean_mlp,embedded_lr,clean_embedded_lr,embedded_mlp,clean_embedded_mlp}]
                    [--extractor {levenshtein,similarity}] [--classify] [--evaluate [PATH]] [--tts] [--verbose]

options:
  -h, --help            show this help message and exit
  --model {rule,lr,clean_lr,mlp,clean_mlp,embedded_lr,clean_embedded_lr,embedded_mlp,clean_embedded_mlp}
                        Which classifier to use (default: clean_mlp); can also be changed at runtime with '!model <name>'
  --extractor {levenshtein,similarity}
                        Which slot extraction method to use (levenshtein distance or DistilBERT cosine similarity)
  --classify            Only run the classifier, not the entire chatbot
  --evaluate [PATH]     Only run evaluation of all classifiers before exiting, optionally using a held-out dataset when [PATH] is given
  --tts                 Use text-to-speech instead of text output
  --explain             Explain why a restaurant is chosen
  --verbose             Show more detailed reasoning information
  --participant PARTICIPANT
                        Participant id; turns logging on
  --condition CONDITION
                        Name of the experimental condition
```


## Classification
`--classify` lets you enter sentences for the model to classify.

Use `!stop` to stop the program and `!model <modelname>` to choose a classifier, where `modelname` is one of `rule, lr, clean_lr, mlp, clean_mlp, embedded_lr, clean_embedded_lr, embedded_mlp, clean_embedded_mlp`


## Evaluation
`--evaluate` computes for each classifier/dataset combination the accuracy, balanced accuracy, macro F1,
weighted F1, per-class precision/recall/F1, a confusion matrix, and misclassified examples.

Everything gets saved under `experiments/`, one folder per classifier/dataset combination:
- `metrics.csv`: accuracy, balanced accuracy, macro F1, weighted F1
- `classification_report.csv`: precision, recall, F1 per class
- `predictions.csv`: every utterance with predicted and true label
- `errors.csv`: just the misclassified ones
- `confusion_matrix.png`: confusion matrix

`experiments/all_experiment_results.csv` has all classifiers together, and `experiments/held_out_results.csv` has the held-out results if you ran those.
