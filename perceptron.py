import numpy as np
from sklearn.datasets import make_blobs
import random

class Perceptron():
    """
    Perceptron classifier
    感知机学习算法的原始形式
    具体推导过程见李航《统计学习方法》第2章感知机  2.3.1 感知机学习算法的原始形式
    Parameters：
    -----------------------
    alpha: float, learning_rate between 0.0 and 1.0, usually 0.01
    n_iter: int, the max iteration times
    ------------------------
    tips:
    1. 感知机(perceptron)是二类分类的线性分类模型,其输入为实例的特征向量,输出
    为实例的类别,取+1和–1二值。
    2. 当训练数据集是线性可分的，感知机算法的原始形式是收敛的。
    """

    def __init__(self, alpha=0.01, n_iter=100):
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X, y):
        # m 样本个数， n样本维度
        m, n = np.shape(X)
        # 初始化参数 w = 0, b = 0
        self.w = np.zeros(n)
        self.bais = 0
        # 随机梯度下降求解w，b
        for _ in range(self.n_iter):
            for xi, yi in zip(X, y):
                predict = np.inner(xi, self.w) + self.bais
                # 如果yi与predict不同号，则更新self.w self.bais
                if yi * predict <= 0:
                    # w <-- w + xi * yi * alpha,
                    # b <-- b + yi * alpha
                    self.w += yi * xi * self.alpha
                    self.bais += yi * self.alpha

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = np.sign(np.dot(X[i], self.w) + self.bais)
        return y_pred

if __name__ == '__main__':
    # 获取数据，划分训练集和测试集
    X, y = make_blobs(n_samples=500, centers=2, random_state=6)
    y[y==0] = 1
    X_train = X[:450]
    y_train = y[:450]
    X_test = X[450:]
    y_test = y[450:]
    # 训练和测试
    clf = Perceptron()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.sum(y_pred == y_test)/ len(y_test)
    print('accuracy:', accuracy)
