import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
# import helper functions
from mlfromscratch.supervised_learning import ClassificationTree
from mlfromscratch.utils import Plot

class RandomForest():
    """
    random forest classifier.
    use a collection of classification trees that trans on random subsets of data using a random subsets of the features

    Parameters:
    -------------
    n_estimators: int
        the number of classification trees that are used
    max_features:
        the maximum number of features that the classification trees are
        allowed to use
    min_samples_split:
        The minimum number of samples needed to make a split when building a tree.
    min_gain:
        the minimum impurity required to split the tree further
    max_depth:
        the maximum depth of a tree
    """
    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, min_gain=0, max_depth=float('inf')):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth

        # Initialize decision tree
        self.trees = []
        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_samples_split=self.min_samples_split, min_impurity=self.min_gain, max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X, y):
        n_features = np.shape(X)[1]
        # if max_features have not been defined
        # select it as sqrt(n_features)
        if not self.max_features:
            self.max_features = int(np.sqrt(n_features))

        for i in range(self.n_estimators):
            # choose one random subset of the data for each tree
            X_subset, y_subset = self._get_random_subsets(X, y)
            # feature bagging (select random subsets of the features)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=False)
            # save the indices of the features for prediction
            self.trees[i].feature_indices = idx
            # choose the features corresponding to the indices
            X_subset = X_subset[:,idx]
            # fit the tree to the data
            self.trees[i].fit(X_subset, y_subset)

    def _get_random_subsets(self, X, y, replacements=True):
        """return random subsets (with replacements) of the data."""
        n_samples = np.shape(X)[0]
        # uses 50% of training samples without
        subsample_size = int(n_samples / 2)
        if replacements:
            subsample_size = n_samples

        idx = np.random.choice(range(n_samples), size=subsample_size, replace=replacements)
        subset = (X[idx], y[idx])
        return subset

    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        # let each tree make a prediction on the data
        for i,tree in enumerate(self.trees):
            # indices of tree features that the tree has trained on
            idx = tree.feature_indices
            # make a prediction based on those features
            prediction = tree.predict(X[:, idx])
            y_preds[:,i] = prediction

        y_pred = []
        # for each sample
        for sample_predictions in y_preds:
            # select the most common class prediction
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred

if __name__ == "__main__":
    data = datasets.load_digits()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = RandomForest(n_estimators=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:%.2f" %accuracy)
