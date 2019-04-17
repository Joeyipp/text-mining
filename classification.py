import sys

from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC


def main():
    training_data_file = sys.argv[1]
    feature_vectors, targets = load_svmlight_file(training_data_file)

    print(feature_vectors)
    print(targets)

main()