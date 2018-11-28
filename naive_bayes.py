# 采用朴素贝叶斯方法识别手写数字
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayes():
    """
    The Naive Bayes classifier.
    参见李航 统计学习方法 第4章 朴素贝叶斯法
    params：
    theta ： float， 平滑因子
    """
    def __init__(self, theta = 1):
        # theta 平滑因子 ， 取 1 时为 Laplace 平滑
        self.theta = theta

    # 将单通图片转化成二值图片
    def _img2binary(self, img):
        # img 为 784维的向量，每个元素取值 0 - 255
        cv_img = img.astype(np.uint8)
        # 图片像素值（ 0 - 255）大于 50 转化成 0， 小于 50 转化成 1
        ret, binary_img = cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV)
        return binary_img

    def fit(self, X, y):
        # m 训练样本数， n 每个样本维度
        m, n = np.shape(X)
        self.X_train = X
        self.y_train = y
        # class_num 要分成多少类
        self.class_num = len(np.unique(y))
        # self.prior P(Y) 先验概率
        self.prior = np.zeros(self.class_num)
        # self.likelihood P(X|Y) 似然概率
        self.likelihood = np.zeros((self.class_num, n, 2))

        self._calculate_prior()
        self._calculate_likelihood()
        self._laplace_smooth()
        print(self.prior)

    def _calculate_prior(self):
        """ Calculate the prior probability"""
        for i in range(len(self.y_train)):
            y_label = self.y_train[i]
            self.prior[y_label] += 1

    def _calculate_likelihood(self):
        """ Likelihood of the data X given label Y """
        m, n = self.X_train.shape
        for i in range(m):
            binary_img = self._img2binary(self.X_train[i])
            y_label = self.y_train[i]
            for j in range(n):
                self.likelihood[y_label, j, binary_img[j]] += 1

    def  _laplace_smooth(self):
        """
        Laplace 平滑，用极大似然估计可能会出现所要估计的概率值为0的情况。
        这会影响到后验概率的计算结果,使分类产生偏差。
        解决这一问题的方法是采用贝叶斯估计。
        具体地，在随机变量各个取值的频数上赋予一个正数 theta。
        当 theta=0 时就是极大似然估计。
        常取 theta=1,这时称为拉普拉斯平滑(Laplace smoothing)
        """
        self.prior = (self.prior + self.theta) / (self.prior.sum() + self.theta * self.class_num)
        # 取对数，避免乘法计算溢出
        self.prior = np.log(self.prior)

        for i in range(self.class_num):
            for j in range(self.X_train.shape[1]):
                temp = self.likelihood[i, j, :]
                self.likelihood[i,j, :] = (temp + self.theta)/ (temp.sum() + 2 * self.theta)
        # 取对数，避免乘法计算溢出
        self.likelihood = np.log(self.likelihood)

    def _calculate_probability(self, binary_img, label):
        """
        Calculate P(Y=label|X)
        log(P(Y|X)} = log(P(X|Y)) + logP(Y)
        """
        posterior = self.prior[label]
        for i in range(len(binary_img)):
            # 朴素贝叶斯 在已知类别的条件下，随机变量 X 各个维度的取值是独立的
            posterior += self.likelihood[label, i, binary_img[i]]
        return posterior

    def predict(self, X_test):
        """ Predict the class labels of the samples in X
        Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
            or Posterior = Likelihood * Prior / Scaling Factor

        P(Y|X) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by _calculate_likelihood)
        P(Y)   - Prior (given by _calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.

        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        """
        y_pred = np.zeros(len(X_test))
        for i, img in enumerate(X_test):
            img_data = self._img2binary(img)
            img_prob = np.zeros(self.class_num)
            for j in range(self.class_num):
                # 计算后验概率
                img_prob[j] = self._calculate_probability(img_data, j)
            # 分类，取概率最大的类
            y_pred[i] = np.argmax(img_prob)
        return y_pred

if __name__ == '__main__':
    # train.csv 下载自  https://www.kaggle.com/c/digit-recognizer/data
    data = pd.read_csv('train.csv')
    data = data.values
    imgs = data[:1000,1:]
    labels = data[:1000, 0]
    # 划分测试数据 训练数据
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.33)

    print(X_train.shape)
    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_pred, y_test)
    print ("The accruacy socre is ", score)
