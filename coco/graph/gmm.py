"""
Credits: source code borrowed & adapted from https://github.com/ahaque/cs231b
"""
from gaussian import Gaussian
from sklearn.cluster import KMeans
import numpy as np


class GMM:
    def __init__(self, K):
        self.K = K
        self.gaussians = [Gaussian() for _ in xrange(self.K)]
        self.weights = np.array([1.0 / K] * K)

    # X - Array of pixels, not necessarily an image
    def initialize_gmm(self, X):
        clusterer = KMeans(n_clusters=self.K, max_iter=10, random_state=None)
        clusters = clusterer.fit_predict(X)

        num_pixels = float(X.shape[0])

        for i, distribution in enumerate(self.gaussians):
            # TODO: check for empty cluster
            distribution.update_parameters(X[clusters == i])
            self.weights[i] = np.sum(clusters == i) / num_pixels

        return clusters

    def get_component(self, x):
        components = np.zeros((x.shape[0], len(self.gaussians)))

        for i, g in enumerate(self.gaussians):
            components[:, i] = self.weights[i] * g.compute_probability(x)

        return np.argmax(components, axis=1)

    # X -> -1, 1 .. K -> -1 = not current class
    def update_components(self, X, assignments):
        num_pixels = float(np.sum(assignments != -1))

        # print np.sum([np.sum(assignments==i) for i in xrange(len(self.gaussians))]), num_pixels
        # gaussians = []
        # weights = []
        for i, distribution in enumerate(self.gaussians):
            if X[assignments == i].shape[0] != 0:
                distribution.update_parameters(X[assignments == i])
                self.weights[i] = (np.sum(assignments == i) / num_pixels)
            else:
                # print 'Empty component',i
                distribution.mean = [-1e9, -1e9, -1e9]
                self.weights[i] = 0
                # self.gaussians = gaussians
                # self.weights = np.array(weights)
                # self.K = len(gaussians)

    def compute_probability(self, x):
        return np.dot(self.weights, [g.compute_probability(x) for g in self.gaussians])


def GMM_test():
    g = GMM(5)

    for i, distribution in enumerate(g.gaussians):
        distribution.mean = [i * 3, i * 3, i * 3]

    print g.get_component([1.5, 1.5, 1.5])

    X = np.zeros((0, 3))
    for i in range(5):
        X = np.concatenate((X, np.random.multivariate_normal([i * 3] * 3, np.eye(3), 10)), axis=0)

    g.initialize_gmm(X)

    print [g.get_component(x) for x in X]


def main():
    GMM_test()


if __name__ == '__main__':
    main()
