# PART 4: DOCUMENT CLUSTERING
# COMMANDLINE USAGE: python3 clustering.py

import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

def plot_curve(quality_measure_method, feature_vectors, targets):

    kMeans_metrics = []
    hier_clus_metrics = []
    i = []

    print("---> Clustering with kMeans and Single Linkage Models with [2, 25] clusters")
    with tqdm(total=25) as pbar:
        pbar.update(2)

        # Try a range of n_clusters (2-25)
        for n_clusters in range(2, 26):
            kMeans_model = KMeans(n_clusters=n_clusters).fit(feature_vectors)
            single_linkage_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(feature_vectors.toarray())

            kMeans_clustering_labels = kMeans_model.labels_
            single_linkage_clustering_labels = single_linkage_model.labels_

            # Measure the clustering quality using SC
            if quality_measure_method == "Sihouette Coefficient":
                kMeans_metrics.append(metrics.silhouette_score(feature_vectors, kMeans_clustering_labels, metric='euclidean'))
                hier_clus_metrics.append(metrics.silhouette_score(feature_vectors, single_linkage_clustering_labels, metric='euclidean'))
            
                # Measure the clustering quality using NMI
            elif quality_measure_method == "NMI":
                kMeans_metrics.append(metrics.normalized_mutual_info_score(targets, kMeans_clustering_labels))
                hier_clus_metrics.append(metrics.normalized_mutual_info_score(targets, single_linkage_clustering_labels))
            
            i.append(n_clusters)
            pbar.update(1)
    
    # Plot Configurations
    plt.figure()
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.title("Document Clustering Measures with {} Method".format(quality_measure_method))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Clustering Quality Measure")
    plt.grid()
    plt_color = {"kMeans": "r", "Hier. Clustering": "b"}
    
    plt.plot(i, kMeans_metrics, ".-", color=plt_color["kMeans"], label="kMeans")
    plt.plot(i, hier_clus_metrics, ".-", color=plt_color["Hier. Clustering"], label="Hier. Clustering")

    return plt

def main():

    # Open and load the training_data_file
    training_data_file = "training_data_file.TFIDF"
    print("\n-> Loading data from {}".format(training_data_file))
    feature_vectors, targets = load_svmlight_file(training_data_file)
    print("Done!")

    # Select the top K features from experiment 3
    X_new = SelectKBest(chi2, k=10000).fit_transform(feature_vectors, targets)

    # Evaluate the 2 clustering methods (kMeans and Hier. Clustering) with 2 quality measures (SC and NMI) methods
    quality_measure_methods = ["Sihouette Coefficient", "NMI"]

    for quality_measure_method in quality_measure_methods:
        print("\n-> Plotting figure for {} clustering quality measure".format(quality_measure_method))
        plt = plot_curve(quality_measure_method, X_new, targets)
        plt.legend(loc="best")
        plt.savefig("{}.png".format(quality_measure_method))
        print("Figure saved to {} as {}.png".format(os.getcwd(), quality_measure_method))

    print("\n-> All done!")

if __name__ == '__main__':
    main()