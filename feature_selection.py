import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from classification import classification
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif

def plot_curve(classifier, X, y, top_k_features):
    top_k_f1_macro = []

    for top_k in top_k_features:
        X_new1 = SelectKBest(chi2, k=top_k).fit_transform(X, y)
        f1_macro, precision_macro, recall_macro = classification(classifier, X_new1, y)
        top_k_f1_macro.append(f1_macro.mean())
    
    plt.figure()
    plt.title("Testing Title")
    plt.xlabel("Top K Features")
    plt.ylabel("F1 Macro")
    plt.grid()
    plt.plot(top_k_features, top_k_f1_macro)
    plt.legend(loc="best")
    
def main():
    training_data_files = ["training_data_file.TF", "training_data_file.IDF", "training_data_file.TFIDF"]
    classifiers = ["Multinominal N. Bayes", "Bernoulli N. Bayes", "k-Nearest Neighbor", "Support Vector Machine"]

    # for training_data_file in training_data_files:
    #     feature_vectors, targets = load_svmlight_file(training_data_file)

    # X_new1 = SelectKBest(chi2, k=top_k).fit_transform(X, y)
    # X_new2 = SelectKBest(mutual_info_classif, k=top_k).fit_transform(X, y)

    top_k_features = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 32715]

    X, y = load_svmlight_file("training_data_file.TFIDF")

    for classifier in classifiers:
        plot_curve(classifier, X, y, top_k_features)


    plt.show()

if __name__ == '__main__':
    main()