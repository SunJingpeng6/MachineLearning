import numpy as np
from sklearn import datasets

# from mlfromscratch.utils import Plot

class KMeans():
    """A simple clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to the center
    of the new formed clusters.


    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    n_iter: int
        The number of iterations the algorithm will run for if it does
        not converge before that.
    """
    def __init__(self, k=3, n_iter=200):
        self.k = k
        self.n_iter = n_iter

    def _init_random_centroids(self, X):
        """ Initialize the centroids as k random samples of X"""
        n_samples, n_features = X.shape
        idx = np.random.choice(range(n_samples), size=self.k, replace=False)
        centroids = X[idx]
        return centroids

    def _closest_centroid(self, sample, centroids):
        """ Return the index of the closest centroid to the sample """
        distances = np.array([self._euclidean_distance(sample, centroid) for centroid in centroids])
        # closest_dist = distances.min()
        closest_i = distances.argmin()
        return closest_i

    def _euclidean_distance(self, sample, centroid):
        """Calulate Euclidean distance"""
        distance = np.sqrt(np.inner(sample-centroid, sample-centroid))
        return distance

    def _create_clusters(self, centroids, X):
        """ Assign the samples to the closest centroids to create clusters """
        n_samples = X.shape[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def _calculate_centroids(self, clusters, X):
        """ Calculate new centroids as the means of the samples in each cluster  """
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(X[cluster], axis=0)
        return centroids

    def _get_cluster_labels(self, clusters, X):
        """Classify samples as the index of their clusters """
        y_pred = np.zeros(X.shape[0])
        for cluster_i, cluster in enumerate(clusters):
            y_pred[cluster] = cluster_i
        return y_pred

    def predict(self, X):
        """ Do K-Means clustering and return cluster indices """
        centroids = self._init_random_centroids(X)
        # Initialize centroids as k random samples from X
        for _ in range(self.n_iter):
            # Assign samples to closest centroids (create clusters)
            clusters = self._create_clusters(centroids, X)
            # Save current centroids for convergence check
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self._calculate_centroids(clusters, X)
            # If no centroids have changed => convergence
            diff = centroids - prev_centroids
            if not diff.any():
                break
        return self._get_cluster_labels(clusters, X)

if __name__ == '__main__':
    # Load the dataset
    X, y = datasets.make_blobs()
    # Cluster the data using K-Means
    clf = KMeans(k=3)
    y_pred = clf.predict(X).astype('int')
    print(y)
    print(y_pred)
    # Project the data onto the 2 primary principal components
    # p = Plot()
    # p.plot_in_2d(X, y_pred, title="K-Means Clustering")
    # p.plot_in_2d(X, y, title="Actual Clustering")
