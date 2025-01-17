import numpy as np
import matplotlib.pyplot as plt


def plot_kmeans_clustering(X, labels, cluster_centers):
    plt.figure()

    for cluster_id in np.unique(labels):
        cluster_sample_indices = np.where(labels == cluster_id)[0]
        X_cluster = X[cluster_sample_indices]
        if X_cluster.shape[0] > 0:
            plt.scatter(X_cluster[:,0], X_cluster[:,1], marker='o', color='C{}'.format(cluster_id), s=150)

            for x in X_cluster:
                plt.plot([x[0],cluster_centers[cluster_id][0]], [x[1],cluster_centers[cluster_id][1]], '--', linewidth=0.5, color='k'.format(cluster_id))
            
        plt.scatter(cluster_centers[:,0], cluster_centers[:,1], marker='+', color='k', s=250, lw=5)

    plt.tight_layout()
    
    plt.show()