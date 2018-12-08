import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from mlfromscratch.supervised_learning.decision_tree import RegressionTree
from mlfromscratch.utils import Plot

class GradientBoosting():
    """
    Super class of GradientBoostingClassifier and GradientBoostinRegressor.
    uses a collection of regression trees that trains on predicting the gradient of the gradient of the loss function.

    Parameters:
    --------------
    n_estimators: int
        the number of classification trees that are used.
    learning_rate: float
        the step length that will be taken when following the negative
        gradient during training
    min_samples_split: int
        the minimum number of samples needed to make a split when
        building a tree
    min_impurity: float
        the minimum impurity required to split the tree further
    max_depth: int
        the maximum depth of a tree
    regression: boolean
        True of False depending on if we are doing regression or classification
    """
    def __init__(self, n_estimators, learning_rate, min_samples_split, min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        # square loss for regression
        # log loss for classification
        self.loss = SquareLoss()
        if not self.regression:
            self.loss = CrossEntropy()

        # Initialize regression trees
        self.trees = []
        for _ in range(self.n_estimators):
            tree = RegressionTree(min_samples_split=self.min_samples_split, min_impurity=self.min_impurity, max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X, y):
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in range(self.n_estimators):
            gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, gradient)
            update = np.array(self.trees[i].predict(X))
            # update y prediction
            y_pred += -self.learning_rate * update

    def predict(self, X):
        y_pred = np.array([])
        # make prediction
        for tree in self.trees:
            update = np.array(tree.predict(X))
            y_pred = -update if not y_pred.any() else y_pred - update

        if not self.regression:
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred

class SquareLoss():
    def __call__(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return - (y - y_pred)

class CrossEntropy():
    def __call__(self, y, p):
        p = np.clip(p, 1e-15, 1-1e-15)
        return -y * np.log(p) - (1-y) * np.log(1-p)

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1-1e-15)
        return -(y / p) + (1 - y) / (1 - p)

class GradientBoostinRegressor(GradientBoosting):
    def __init__(self, n_estimators=20, learning_rate=0.5, min_samples_split=2, max_depth=4, min_var_red=1e-7):
        super(GradientBoostinRegressor, self).__init__(n_estimators=n_estimators, learning_rate=learning_rate, min_samples_split=min_samples_split, min_impurity=min_var_red, max_depth=max_depth, regression=True)

class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=20, learning_rate=.5, min_samples_split=2, min_info_gain=1e-7, max_depth=2, debug=False):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_info_gain,
            max_depth=max_depth,
            regression=False)

    def _to_categorical(self, x, n_col=None):
        """one hot encoding of nominal values"""
        if not n_col:
            n_col = np.amax(x) + 1
        one_hot = np.zeros((x.shape[0], n_col))
        one_hot[np.arange(x.shape[0]), x] = 1
        return one_hot

    def fit(self, X, y):
        y = self._to_categorical(y)
        # print(y)
        super(GradientBoostingClassifier,self).fit(X, y)

def runGradientBoostinRegressor():
    def _mean_squared_error(y,y_pred):
        return (1/ len(y)) * np.inner(y-y_pred, y-y_pred)

    data = pd.read_csv('TempLinkoping2016.txt', sep="\t")
    time = np.atleast_2d(data['time'].values).T
    temp = np.atleast_2d(data['temp'].values).T
    X = time.reshape((-1,1))
    X = np.insert(X, 0, values=1, axis=1) # Insert bias term
    y = temp[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    model = GradientBoostinRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(y_test)
    print(y_pred)
    mse = _mean_squared_error(y_test, y_pred)
    print ("Mean Squared Error:", mse)

    y_pred_line = model.predict(X)
    # Color map
    cmap = plt.get_cmap('viridis')
    # Plot the results
    m1 = plt.scatter(366 * X_train[:, 1], y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test[:, 1], y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test[:, 1], y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()

def runGradientBoostingClassifier():
    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = np.mean(y_test == y_pred)
    print ("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred,
        title="Gradient Boosting",
        accuracy=accuracy,
        legend_labels=data.target_names)

if __name__ == "__main__" :

    # print ("-- Gradient Boosting Classification --")
    runGradientBoostingClassifier()

    # print ("-- Gradient Boosting Regression --")
    # runGradientBoostinRegressor()
