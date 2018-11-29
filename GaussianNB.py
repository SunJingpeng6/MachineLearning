# 高斯朴素贝叶斯 Gaussian Naive Bayes
# 高斯朴素贝叶斯模型假设 在类别确定的条件下，数据每一维特征都独立地服从一维高斯分布
# 鸢尾花分类
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class GaussianNB():
    """The Gaussian Naive Bayes classifier. """
    def __init__(self, theta = 1):
        # 平滑因子
        self.theta = theta

    def fit(self, X, y):
        m, n = np.shape(X)
        self.X_train, self.y_train = X, y
        self.class_num = len(np.unique(y))
        self.y_labels= np.unique(y)

        self.prior = np.zeros(self.class_num)
        # 高斯分布参赛
        self.mu = np.zeros((self.class_num, n))
        self.var = np.zeros((self.class_num, n))
        # 计算先验概率
        self._calculate_prior()
        # 计算高斯分别参数
        self._calculate_gaussion_params()

    def _calculate_prior(self):
        """ Calculate the prior probability"""
        for i in range(len(self.y_train)):
            y_label = self.y_train[i]
            self.prior[y_label] += 1
        # 平滑
        self.prior = (self.prior + self.theta)/ (self.prior.sum() + self.theta * self.class_num)

    def _calculate_gaussion_params(self):
        """ Calculate gaussion parameters mu, var"""
        m, n = self.X_train.shape
        for y_label in self.y_labels:
            index = (self.y_train == y_label)
            X_where_lable = self.X_train[index]
            self.mu[y_label] = X_where_lable.mean(axis=0)
            self.var[y_label] = X_where_lable.var(axis=0)

    def _calculate_likelihood(self, mean, var, x):
        """ Gaussian likelihood of the data x given mean and var """
        eps = 1e-4
        coeff = 1. / np.sqrt(2.* np.pi * var + eps )
        exponent = np.exp(-(np.power(x - mean, 2) / (2. * var + eps)))
        return coeff * exponent

    def _pred_prob(self, sample, label):
        """
        Calculate the probability P(Y=label|X)
        P(Y=label|X) ～ P(X|Y=label)* P(Y=label)
        其中 P(X|Y=label) = P(x1|Y=label) * P(x2|Y=label) * ... * P(xn|Y=label)
        """
        posterior = self.prior[label]
        n = len(sample)
        for i in range(n):
            mean, var = self.mu[label, i], self.var[label, i]
            likelihood = self._calculate_likelihood(mean, var, sample[i])
            # 朴素贝叶斯
            posterior *= likelihood
        return posterior

    def _classify(self, sample):
        """
        Classification using Bayes Rule
        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        posteriors = [self._pred_prob(sample, label) for label in self.y_labels]
        return np.argmax(posteriors)

    def predict(self, X_test):
        """ Predict the class labels of the samples in X """
        y_pred = [self._classify(sample) for sample in X_test]
        return y_pred

if __name__ == '__main__':
    data = datasets.load_digits()
    X = data.data
    y = data.target

    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_test == y_pred)
    print('Accuracy: %.2f' %accuracy)
