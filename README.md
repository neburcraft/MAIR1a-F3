# INFOMAIR Project part 1a
Dialog act classification for the restaurant recommendation dialog system. Utterances from the DSTC 2 dataset are classified into one of 15 dialog acts, using a rule-based baseline and machine learning classifiers such as logistic regression and MLP. Each classifier was trained and evaluated on a stratified split and a grouped split that prevents duplicate utterances from leaking between train and test.

## Group F3
- Anouk van Ladesteijn ([a.r.vanladesteijn@uu.nl](a.r.vanladesteijn@uu.nl))
- Huub Statius Muller ([h.b.statiusmuller@students.uu.nl](h.b.statiusmuller@students.uu.nl))
- Yanpeng Wang ([y.wang36@students.uu.nl](y.wang36@students.uu.nl))
- Ruben Wijmenga ([ruben.wijmenga@students.uu.nl](mailto:r.wijmenga@students.uu.nl))

## TODO
Each TODO should be implemented in its own branch. For testing your own additions, create a new file - do not edit main.py unless you need to. Merge whenever your part is finished and say in WhatsApp if you encounter any merge conflicts.

- [ ] [Ruben|04-10] Address feedback on 1a
- [ ] [Huub|04-10] Improve evaluation function
- [ ] [Anouk|03-10] Create state diagram
- [ ] [Anouk|06-10] Implement state diagram in python with state transition function
- [ ] [Albert|06-10] Implement slot extraction - keyword matching
- [ ] [Albert|06-10] Implement slot extraction - Lehvenstein edit distance (`python-Lehvenstein` library)
- [ ] [Ruben|06-10] Implement slot extraction - Semantic similarity via embeddings (with DistilBERT)
- [ ] [Ruben|08-10] Implement lookup function given act, filled slots and `restaurant_info_extended.csv`
- [ ] [Anouk|08-10] Implement response generation
- [ ] [Albert|08-10] Extend dialog manager with reasoning step (6 inference rules given in assignment)
- [ ] [Huub|09-10] Implement configurability and second comparison functionality
- [ ] [Huub|10-10] Update command-line interface
- [ ] [Anouk|10-10] Implement interaction logging
- [ ] [Ruben|11-10] Finalize code on cleanness, correctness, comments and README
- [ ] [Anouk|10-10] Rewrite report abstract
- [ ] [Anouk|09-10] Rewrite report introduction
- [ ] [Albert+Anouk|09-10] Rewrite report system architecture (classifier → dialog manager → reasoning; design decisions)
- [ ] [Huub|10-10] Rewrite report experiments and results
- [ ] [Ruben|10-10] Rewrite report discussion and conclusion
- [ ] [Albert+Huub|10-10] Update report appendices
- [ ] [Ruben|11-10] Finalize report and hand in all files

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
- `data.py`: loading, cleaning and splitting the dataset (stratified split and grouped split)
- `main.py`: trains all classifiers and starts the interactive prompt
- `evaluate.py`: trains all classifiers and evaluates them

### Data
- `dialog_acts.dat`: the training/development dataset
- `dialog_acts_train.dat`: the generated (fixed seed) training dataset from regular stratified sampling
- `dialog_acts_test.dat`: the generated (fixed seed) testing dataset from regular stratified sampling
- `dialog_acts_train_grouped.dat`: the generated (fixed seed) training dataset from grouped stratified sampling
- `dialog_acts_test_grouped.dat`: the generated (fixed seed) testing dataset from grouped stratified sampling

### Miscellaneous
- `env.yml` for installing packages with mamba or conda
- `requirements.txt` for installing packages on a regular python installation

## Running
```
python ./main.py [--help] [--verbose] [--model {model}]
```
Loads and splits the data, trains all classifiers, and starts a prompt where you can type an utterance to classify it. Type `stop` to quit, or `model <name>` to switch classifiers while it's running.

`--model` picks which classifier is used (default `mlp`). Run `python main.py --help` for the full list of model names. `--verbose` shows every classifier's prediction per utterance instead of just one and gives some extra information while fitting the models.


## Evaluation
```
python classifiers.py
```
Trains all classifiers and evaluates each one on the test split it belongs to (original or grouped) based on the regular accuracy metric.

```
python evaluate.py
```
Trains all classifiers and evaluates each one on the test split it belongs to (original or grouped). Pass a held-out `.dat` file as an argument to also evaluate on that file.

For each classifier/dataset combination this computes accuracy, balanced accuracy, macro F1, weighted F1, per-class precision/recall/F1, a confusion matrix, and the misclassified examples.

### Where the results end up
Everything gets saved under `experiments/`, one folder per classifier/dataset combination:
- `metrics.json`: accuracy, balanced accuracy, macro F1, weighted F1
- `classification_report.csv`: precision, recall, F1 per class
- `predictions.csv`: every utterance with predicted and true label
- `errors.csv`: just the misclassified ones
- `confusion_matrix.png`: confusion matrix

`experiments/all_experiment_results.csv` has all classifiers together, and `experiments/held_out_results.csv` has the held-out results if you ran those.
