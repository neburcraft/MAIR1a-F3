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

## Running
```
python ./main.py [--verbose] 
```
The implemented classifiers are in classifiers.py. All classifier implementations and the shared Classifier base class:
- RuleClassifier — keyword-based baseline
- LRClassifier / MLPClassifier — logistic regression and MLP on bag-of-words features
- FrozenEmbeddingEncoder, EmbeddedLRClassifier / EmbeddedMLPClassifier — logistic regression and MLP on frozen DistilBERT embeddings (batched, cached per unique utterance)


## Evaluation
```
python evaluate.py
```
The evaluation reports:

- Accuracy
- Balanced accuracy
- Macro F1-score
- Weighted F1-score
- Per-class precision, recall and F1-score
- Confusion matrices
- Misclassified examples
