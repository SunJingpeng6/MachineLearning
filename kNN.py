import numpy as np
from sklearn.datasets import make_blobs

class kNN():
    """
    K近邻算法 多分类
    分类时,对新的实例,根据其k个最近邻的训练实例的类别,
    通过多数表决等方式进行预测。
    具体推导过程见李航《统计学习方法》第3章 k近邻法 3.2 k近邻模型

    Parameters：
    -----------------------
    k: int, k个最近邻对测试数据进行预测，k一般取较小的值
    p: int, Lp距离参数p的值， p=1 曼哈顿距离， p=2 欧式距离
    ------------------------
    tips:
    k值的选择、距离度量及分类决策规则
    是k近邻法的三个基本要素。
    """
    def __init__(self, X_train, y_train, k=3, p=2):
        self.k = k
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    # 计算距离
    def _distance(self, xi):
        # 欧式距离
        def _euclidean_distance(xi):
            temp = np.square(self.X_train - xi)
            # distance 实际上是欧式距离的平方
            distance = temp.sum(axis=1)
            return distance
        # 曼哈顿距离
        def _manhattan_distance(xi):
            temp = np.abs(self.X_train - xi)
            distance = temp.sum(axis=1)
            return distance
        if self.p == 1:
            distance = _manhattan_distance(xi)
        elif self.p == 2:
            distance = _euclidean_distance(xi)
        return distance

    # 分类，多数表决规则 majority voting rule
    def _classify(self, distance):
        index = np.argsort(distance)[:self.k]
        predict = np.bincount(y_train[index]).argmax()
        return predict

    def predict(self, X):
        m, n = np.shape(X)
        y_pred = np.empty(m)
        for i in range(m):
            distance = self._distance(X[i])
            y_pred[i] = self._classify(distance)
        return y_pred

if __name__ == '__main__':
    # 获取数据，划分训练集和测试集
    X, y = make_blobs(n_samples=500, centers=2, random_state=6)
    X_train = X[0:450]
    y_train = y[0:450]
    X_test = X[450:500]
    y_test = y[450:500]
    # 训练和测试
    clf = kNN(X_train, y_train, k=1, p=2)
    y_pred = clf.predict(X_test)
    accuracy = np.sum(y_pred == y_test)/ len(y_test)
    print('accuracy:', accuracy)
