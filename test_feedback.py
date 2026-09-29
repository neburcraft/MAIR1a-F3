from data import load_data, create_stratified_split
from classifiers import RuleClassifier


if __name__=="__main__":
    data = load_data()
    train_data, test_data = create_stratified_split(data)
    classifier = RuleClassifier()
    print(test_data)
    for i, y, x in train_data.itertuples():
        print(x, classifier.run(x))