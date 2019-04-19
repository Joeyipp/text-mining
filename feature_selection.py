# PART 3: FEATURE SELECTION

import os
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from classification import classification
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif

def plot_curve(selection_method, feature_vectors, targets):
    # List of classifiers
    classifiers = ["Multinomial NB", "Bernoulli NB", "kN-Neighbor", "SV-Machine"]
    best_k_no = []

    # List of top K features number
    top_k_features_no = []
    for top_k in range(100, 22000, 300):
        top_k_features_no.append(top_k)

    # Plot Configurations
    plt.figure()
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.title("K-Features Selection with {} Method".format(selection_method))
    plt.xlabel("Top K Features")
    plt.ylabel("F1 Macro (%)")
    plt.xlim = (0, 22500)
    plt.ylim = (0, 101)
    plt.grid()
    plt_color = {"Multinomial NB": "r",
                 "Bernoulli NB": "g",
                 "kN-Neighbor": "b",
                 "SV-Machine": "y"}
    
    # Plotting figures for each classifier with different selection methods
    for classifier in classifiers:
        print("---> Fitting data with {}".format(classifier))
        top_k_f1_macro = []
        top_k_f1_macro_pstd_dev = []
        top_k_f1_macro_nstd_dev = []

        for top_k in top_k_features_no:

            if selection_method == "chi2":
                # Select the top K features using Chi-Squared Method
                X_new = SelectKBest(chi2, k=top_k).fit_transform(feature_vectors, targets)

            elif selection_method == "mutual_info_classif":
                # Select the top K features using Mutual-Information Method
                X_new = SelectKBest(mutual_info_classif, k=top_k).fit_transform(feature_vectors, targets)

            # Train and fit the classifier with the new X vector with top K features
            f1_macro, precision_macro, recall_macro = classification(classifier, X_new, targets)

            top_k_f1_macro.append(f1_macro.mean() * 100)
            top_k_f1_macro_pstd_dev.append((f1_macro.mean() + f1_macro.std() * 2) * 100)
            top_k_f1_macro_nstd_dev.append((f1_macro.mean() - f1_macro.std() * 2) * 100)

        # Retrieve the best performing number of K
        argmax_f1 = np.argmax(top_k_f1_macro)
        best_k_no.append((classifier, top_k_features_no[argmax_f1], max(top_k_f1_macro)))

        # Plot figure
        plt.fill_between(top_k_features_no, top_k_f1_macro_nstd_dev, top_k_f1_macro_pstd_dev, alpha=0.1, color=plt_color[classifier])
        plt.plot(top_k_features_no, top_k_f1_macro, color=plt_color[classifier], label=classifier)
    
    return plt, best_k_no
    
def main():
    
    # Open and load the training_data_file
    training_data_file = "training_data_file.TFIDF"
    print("\n-> Loading data from {}".format(training_data_file))
    feature_vectors, targets = load_svmlight_file(training_data_file)
    print("Done!")
    
    # Evaluate classifiers with 2 feature selection methods
    selection_methods = ["chi2", "mutual_info_classif"]

    for selection_method in selection_methods:
        print("\n-> Plotting figure for {} selection method".format(selection_method))
        plt, best_k_no = plot_curve(selection_method, feature_vectors, targets)
        plt.legend(loc="best")
        plt.savefig("{}.png".format(selection_method))
        print("Figure saved to {} as {}.png".format(os.getcwd(), selection_method))

        # Show the best performing K number for each classifier
        print("\n-> The best performing K using {} method".format(selection_method))
        for k in best_k_no:
            print("---> {}: {} ({:.2f}%)".format(k[0], k[1], k[2]))
    
    print("\n-> All done!")
        
if __name__ == '__main__':
    main()