# INFOMAIR Project part 1a
Dialog act classification for the restaurant recommendation dialog system. Utterances from the DSTC 2 dataset are classified into one of 15 dialog acts, using a rule-based baseline and machine learning classifiers such as logistic regression and MLR. Each classifier was trained and evaluated on a stratified split and a grouped split that prevents duplicate utterances from leaking between train and test.

## Group F3
- Anouk van Ladesteijn ([a.r.vanladesteijn@uu.nl](a.r.vanladesteijn@uu.nl))
- Huub Statius Muller ([h.b.statiusmuller@students.uu.nl](h.b.statiusmuller@students.uu.nl))
- Yanpeng Wang ([y.wang36@students.uu.nl](y.wang36@students.uu.nl))
- Ruben Wijmenga ([ruben.wijmenga@students.uu.nl](mailto:ruben.wijmenga@students.uu.nl))

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
- `classifiers.py`: all classifier implementations and the shared `Classifier` base class (`RuleClassifier`, `LRClassifier`, `MLPClassifier`, `FrozenEmbeddingEncoder`, `EmbeddedLRClassifier`, `EmbeddedMLPClassifier`)
- `data.py`: loading, cleaning and splitting the dataset (stratified split and grouped split)
- `main.py`: trains all classifiers and starts the interactive prompt
- `evaluate.py`: trains all classifiers and evaluates them
- `dialog_acts.dat`: the training/development dataset

## Running
```
python ./main.py [--verbose] 
```
Loads and splits the data, trains all classifiers, and starts a prompt where you can type an utterance to classify it. Type `stop` to quit, or `model <name>` to switch classifiers while it's running.

`--model` picks which classifier is used (default `mlp`). Run `python main.py --help` for the full list of model names. `--verbose` shows every classifier's prediction per utterance instead of just one.




## Evaluation
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
