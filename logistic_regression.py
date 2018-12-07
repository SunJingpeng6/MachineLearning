import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class Sigmoid():
    def __call__(self, x):
        return 1. / (1. + np.exp(-x))

class LogisticRegression():
    """
    Logistic Regression classifier.
    We use gradient descent method when training to minimize loss function
    Parameters:
    -----------
    learning_rate: float
        the step length that will be taken when following the negative gradient
        during training
    n_iter: int                                                                                                                   
        iteration times
    -----------
    """
    def __init__(self, learning_rate=0.1, n_iter=4000):
        self.params = None
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        #
        limit = 1 / np.sqrt(n_features)
        self.params = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        self._initialize_parameters(X)
        for i in range(self.n_iter):
            y_pred = self.sigmoid(np.dot(X, self.params))
            # move against the gradient of the loss function with
            # respect to the parameters to minimize the loss function
            self.params += self.learning_rate * np.dot((y - y_pred), X)

    def predict(self, X):
        y_pred = np.round(self.sigmoid(np.dot(X, self.params))).astype(int)
        return y_pred

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


if __name__ == '__main__':
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[[data.target != 0]]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    print ("Accuracy:", accuracy)
