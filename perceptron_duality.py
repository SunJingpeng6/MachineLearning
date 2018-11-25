import numpy as np
from sklearn.datasets import make_blobs

class PerceptronDuality():
    """
    Perceptron classifier
    感知机学习算法的对偶形式
    具体推导过程见李航《统计学习方法》第2章感知机  2.3.3 感知机学习算法的对偶形式
    对偶形式中训练实例仅以内积的形式出现。为了方便,可以预先将训练集中实例间的
    内积计算出来并以矩阵的形式存储,这个矩阵就是Gram矩阵。
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

    def __init__(self, alpha=0.01, n_iter=10):
        self.alpha = alpha
        self.n_iter = n_iter

    def _gram(self, X):
        # 创建训练样本的Gram矩阵
        m, n = np.shape(X)
        self.gram_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                self.gram_matrix[i, j] = np.inner(X[i], X[j])

    def fit(self, X, y):
        self._gram(X)
        # m训练样本数， n样本维度
        m, n = np.shape(X)
        # self.a 每个训练实例的权重, m维向量
        self.a = np.zeros(m)
        self.w = np.zeros(n)
        self.bais = 0
        for _ in range(self.n_iter):
            for i in range(m):
                temp = self.a * y * self.gram_matrix[i]
                # predict 预测值
                predict = temp.sum() + self.bais
                # 如果y[i] 和 predict 异号，就改变i号样本的权重
                if y[i] * predict <= 0:
                    self.a[i] += self.alpha
                    self.bais += self.alpha * y[i]
        # 由self.a 求出 self.w
        for i in range(n):
            temp = self.a * y * X[:,i]
            self.w[i] = temp.sum()

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = np.sign(np.dot(X[i], self.w) + self.bais)
        return y_pred

if __name__ == '__main__':
    # 获取数据，划分训练集和测试集
    X, y = make_blobs(n_samples=500, centers=2, random_state=6)
    y[y==0] = -1
    X_train = X[0:450]
    y_train = y[0:450]
    X_test = X[450:500]
    y_test = y[450:500]
    # 训练和测试
    clf = PerceptronDuality()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.sum(y_pred == y_test)/ len(y_test)
    print('accuracy:', accuracy)
