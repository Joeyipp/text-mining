# PART 2: CLASSIFICATION
# COMMANDLINE USAGE: python3 classification.py

import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_svmlight_file
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

def classification(classifier, feature_vectors, targets, k_fold = 5):
    # Multinomial Naive Bayes: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    if classifier == "Multinomial NB":
        algo = MultinomialNB()
    
    # Bernoulli Naive Bayes: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html
    elif classifier == "Bernoulli NB":
        algo = BernoulliNB()
    
    # k-Nearest Neighbor: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    elif classifier  == "kN-Neighbor":
        algo = KNeighborsClassifier()

    # Support Vector Machine: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    elif classifier == "SV-Machine":
        algo = SVC(gamma='auto')

    algo.fit(feature_vectors, targets)

    # Compute the performance metrics
    f1_macro = cross_val_score(algo, feature_vectors, targets, cv=k_fold, scoring='f1_macro')
    precision_macro = cross_val_score(algo, feature_vectors, targets, cv=k_fold, scoring='precision_macro')
    recall_macro = cross_val_score(algo, feature_vectors, targets, cv=k_fold, scoring='recall_macro')

    return f1_macro, precision_macro, recall_macro

def main():
    training_data_files = ["training_data_file.TF", "training_data_file.IDF", "training_data_file.TFIDF"]
    classifiers = ["Multinomial NB", "Bernoulli NB", "kN-Neighbor", "SV-Machine"]
    
    for training_data_file in training_data_files:
        feature_vectors, targets = load_svmlight_file(training_data_file)

        print("\n-> Classifiers Evaluation using {}\n".format(training_data_file))
        print("\t\t\tF1 Macro\t\tPrecision Macro\t\tRecall Macro")
        
        # Training classifiers with default parameters
        for classifier in classifiers:
            f1_macro, precision_macro, recall_macro = classification(classifier, feature_vectors, targets)
            print("{}\t\t{:.2f} (+/- {:.2f})\t\t{:.2f} (+/- {:.2f})\t\t{:.2f} (+/- {:.2f})".format(classifier, f1_macro.mean(), f1_macro.std() * 2, precision_macro.mean(), precision_macro.std() * 2, recall_macro.mean(), recall_macro.std() * 2))
            
    print("\n-> All done!")

if __name__ == '__main__':
    main()