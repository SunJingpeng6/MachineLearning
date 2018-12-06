# 决策树 --分类树 --回归树
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

def divide_on_feature(X, feature_i, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])

def standardize(X):
    """ Standardize the dataset X """
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    # X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_std

class DecisionNode():
    """
    class that represents a decision node or leaf in the decision tree.

    Parameters:
    --------------
    feature_i, int
        feature index which we want to use as the threshold measure
    threshold, float
        the value that we will compare feature values at feature_i
        against to determine the prediction
    value, float
        the class prediction if classification tree, or float value if
        regression tree
    true_branch: DecisionNode
        next decision node for samples where features value met the threshold
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    ----------------
    """
    def __init__(self, feature_i=None, threshold=None, value=None, true_branch=None, false_branch=None):
        # index for the feature that is tested
        self.feature_i = feature_i
        # threshold value for feature
        self.threshold = threshold
        # value if the node is a leaf in the tree
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

# super class of regression tree and classification tree
class DecisionTree():
    """
    super class of regression tree and classification tree

    Parameters:
    -------------
    min_samples_split: int
        the minimum number of samples needed to make a split when build a tree
    min_impurity: float
        the minimum impurity required to split the tree further
    max_depth: int
        the maximum depth of a tree
    loss: function
        loss function that is used for gradient boosting models to
        calculate impurity
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float('inf'), loss=None):
        # root node in decision tree
        self.root = None
        # minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        #
        self.min_impurity = min_impurity
        # the max depth to grow the tree to
        self.max_depth = max_depth
        # function to calculate impurity
        # classif. -> info gain , regr -> variance reduction
        self._impurity_calculation = None
        # function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # if y is one hot encoded (multi-dim) or not (one_dim)
        self.one_dim = None
        # if gradient boost
        self.loss = loss

    def fit(self, X, y, loss=None):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""
        largest_impurity = 0
        # feature index and threshold
        best_criteria = None
        # Subsets of the data
        best_sets = None

        # check if expansion y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # add y as last column of X
        # Xy = np.hstack(X, y)
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and current_depth <=  self.max_depth:
            # calculate the impurity for each feature
            for feature_i in range(n_features):
                # all value of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # iterate through all unique values feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # divide x and y depending on if the feature value of X at index feature_i meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    if len(Xy1)>0 and len(Xy2)>0:
                        # Select the y value of two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        # if this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature index
                        if impurity > largest_impurity :
                            largest_impurity = impurity
                            best_criteria = {'feature_i': feature_i, 'threshold': threshold}
                            best_sets = {
                                'leftX': Xy1[:, :n_features],
                                'lefty': Xy1[:, n_features:],
                                'rightX': Xy2[:, :n_features],
                                'righty': Xy2[:, n_features:]
                                }

        if largest_impurity > self.min_impurity:
            # recursive bulid subtrees for right and left branches
            true_branch = self._build_tree(best_sets['leftX'], best_sets['lefty'], current_depth + 1)
            false_branch = self._build_tree(best_sets['rightX'], best_sets['righty'], current_depth + 1)
            return DecisionNode(feature_i=best_criteria['feature_i'], threshold=best_criteria['threshold'], true_branch=true_branch, false_branch=false_branch)

        # if at leaf -> determine value
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """
        do a recursive search down the tree and make a prediction
        of the data sample by the value of the leaf that we end up at
        """

        if tree is None:
            tree = self.root

        # if we have a value(i.e we're at a leaf) -> return value as the prediction
        # notice the difference
        # between --if tree.value  and --if tree.value is not none
        if tree.value is not None:
            return tree.value

        # choose the feature that we will test
        feature_value = x[tree.feature_i]

        # determine if we will follow left or right branch
        branch = tree.false_branch
        # print(tree.feature_i, tree.threshold, tree.value)
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        # test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """classify sampless one by one and return the sets of labels"""
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=' '):
        """recursively print the decision tree"""
        if not tree:
            tree = self.root

        # if we are at leaf -> print label
        if tree.value is not None:
            print(tree.value)
        # go deeper down the tree
        else:
            # print test
            print("%s:%s?" %(tree.feature_i, tree.threshold))
            # print the true scenario
            print('%sT->'%(indent), end='')
            self.print_tree(tree.true_branch, indent+indent)
            # print the false scenario
            print('%sF->'%(indent), end='')
            self.print_tree(tree.false_branch, indent+indent)


def calculate_variance(X):
    """return the variance of the features in dataset X"""
    mean = np.ones(X.shape) * X.mean(axis=0)
    n_samples = X.shape[0]
    variance = (1. / n_samples) * np.diag(np.dot((X-mean).T, X-mean))
    return variance

class RegressionTree(DecisionTree):
    def _calculate_variance(self, X):
        """return the variance of the features in dataset X"""
        mean = np.ones(X.shape) * X.mean(axis=0)
        n_samples = X.shape[0]
        variance = (1. / n_samples) * np.diag(np.dot((X-mean).T, X-mean))
        return variance

    def _calculate_variance_reduction(self, y, y1, y2):
        var_tot = self._calculate_variance(y)
        var_1 = self._calculate_variance(y1)
        var_2 = self._calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        # calculate the variance reduction
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return sum(variance_reduction)

    def _mean_of_y(self, y):
        value = np.mean(y, axis=0)
        return value if len(value)>1 else value[0]

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)

class ClassificationTree(DecisionTree):
    def _calculate_entropy(self, y):
        """calculate the entropy of label array y"""
        unique_labels = np.unique(y)
        entropy = 0
        for label in unique_labels:
            count = len(y[y == label])
            p = count / len(y)
            entropy += -p * np.log2(p)
        return entropy

    def _calculate_information_gain(self, y, y1, y2):
        # calculate information gain
        p = len(y1) / len(y)
        entropy = self._calculate_entropy(y)
        entropy1 = self._calculate_entropy(y1)
        entropy2 = self._calculate_entropy(y2)
        info_gain = entropy - (p * entropy1 + (1 - p) * entropy2)
        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # count number of occurences of samples with label
            count = len(y[y==label])
            if count > max_count:
                max_count = count
                most_common = label
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)

def decision_tree_classifier():
    data = datasets.load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = ClassificationTree()
    clf.fit(X_train, y_train)
    clf.print_tree()
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print('Accuracy: %.2f' %accuracy)

def decision_tree_regressor():
    data = pd.read_csv('TempLinkoping2016.txt', sep='\t')
    time = np.atleast_2d(data['time'].values).T
    temp = np.array(data['temp'].values)
    X = standardize(time)
    y = temp

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RegressionTree()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = (1. / len(y_pred)) * np.inner(y_test - y_pred, y_test - y_pred)
    print('mean squared error:', mse)
    model = RegressionTree()

    # Color map
    cmap = plt.get_cmap('viridis')
    # Plot the results
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test, y_pred, color='black', s=10)
    plt.suptitle("Regression Tree")
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3), ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()

if __name__ == '__main__':
    print ("-- Classification Tree --")
    decision_tree_classifier()
    print ("-- Regression Tree --")
    decision_tree_regressor()
