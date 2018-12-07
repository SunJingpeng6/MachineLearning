import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

# decision stump used as weak classifier in this impl. of adaboost
class DecisionStump():
    def __init__(self):
        # Determines if sample shall be classified as -1 or 1 given threshold
        self.polarity = 1
        # the index of the feature used to make classification
        self.feature_index = None
        # the threshold value that the feature should be measured against
        self.threshold = None
        # value indicative of the classifier's accuracy
        self.alpha = None

class Adaboost():
    """
    Boosting method that uses a number of weak classifiers in ensemble to
    make a strong classifier.
    This implementation uses decision stumps, which is a one level Decision Tree.

    Parameters:
    -----------
    n_clf: int
        the number of weak classifiers that will be used
    """
    def __init__(self, n_clf):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights to 1/N
        w = np.full(n_samples, (1/n_samples))

        # iterate through classifiers
        for _ in range(self.n_clf):
            clf = DecisionStump()
            # Minimum error given for using a certain feature value threshold
            # for predicting sample labels
            min_error = float('inf')
            # iterate through every unique feature value and see what
            # value makes the best threshold for predicting y
            for feature_i in range(n_features):
                feature_values = X[:, feature_i]
                unique_values = np.unique(feature_values)
                # try every unique feature value as threshold
                for threshold in unique_values:
                    p = 1
                    # set all prediction to '1' initially
                    prediction = np.ones(np.shape(y))
                    # label the samples whose values are below threshold as '-1'
                    prediction[X[:, feature_i] < threshold] = -1
                    error = sum(w[y != prediction])

                    # If the error is over 50% we filp the polarity
                    # so that samples that were classified as 0 are classified as 1,
                    # and vice versa eg. error = 0.8 -> (1-error) = 0.2
                    if error > 0.5:
                        error = 1 - error
                        p = -1

                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
            # calculate the alpha which is used to update the sample weights.
            # Alpha is also an approximation of classifer's proficiency
            clf.alpha = 0.5 * np.log((1. - min_error) / (min_error + 1e-10))
            # set all predictions to '1' initially
            predictions = np.ones(np.shape(y))
            # the index where the sample value are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            # label those as '-1'
            predictions[negative_idx] = -1
            # calculate new weights
            # Missclassified samples gets larger weights and correctly classifed samples smaller
            w *= np.exp(-clf.alpha * y * predictions)
            # normalize to one
            w /= np.sum(w)
            # save classifier
            self.clfs.append(clf)

    def predict(self, X):
        n_samples = np.shape(X)[0]
        y_pred = np.zeros(n_samples)
        # for each classifier -> label the sample
        for clf in self.clfs:
            # set all predictions to '1'
            predictions = np.ones(np.shape(y_pred))
            # the indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:,clf.feature_index] < clf.polarity * clf.threshold)
            # label those as '-1'
            predictions[negative_idx] = -1
            # add predictions weighted by the classifiers alpha
            y_pred += clf.alpha * predictions
        # return sign of prediction sum
        y_pred = np.sign(y_pred)
        return y_pred

if __name__ == "__main__":
    data = datasets.load_digits()
    X = data.data
    y = data.target

    digit1 = 1
    digit2 = 8
    idx = np.append(np.where(y == digit1)[0], np.where(y == digit2)[0])
    y = data.target[idx]
    # change labels to {-1, 1}
    y[y == digit1] = -1
    y[y == digit2] = 1
    X = data.data[idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # adaboost classification with 5 weak classifiers
    clf = Adaboost(n_clf=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print('accuracy: %.3f' %accuracy)
