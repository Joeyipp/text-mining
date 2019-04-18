import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

def main():

    # Open and load the training_data_file
    training_data_file = "training_data_file.TFIDF"
    print("\n-> Loading data from {}".format(training_data_file))
    X_new, targets = load_svmlight_file(training_data_file)
    print("Done!")

    # Select the top K features from experiment 3
    #X_new = SelectKBest(chi2, k=8800).fit_transform(feature_vectors, targets)

    #kmeans_model = KMeans(n_clusters=20).fit(X_new)
    #clustering_labels = kmeans_model.labels_
    single_linkage_model = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X_new)
    clustering_labels = single_linkage_model.labels_
    print(metrics.silhouette_score(X_new, clustering_labels, metric='euclidean'))
    print(metrics.normalized_mutual_info_score(targets, clustering_labels))


main()