import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# ensembling and stacking model
#
class Classifier():
    def __init__(self, clf, seed=0, params=None):
        # clf 是基本分类器， RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, SVC
        # params 是基本分类器的参数
        params['random_state'] = seed
        self.clf = clf(**params)

    # 训练
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    # 预测
    def predict(self, x):
        return self.clf.predict(x)

    def feature_importances(self, x, y):
        return self.clf.fit(x,y).feature_importances_

    # 得到基本分类器的预测结果，默认5折交叉验证
    def get_prediction(self, x_train, y_train, x_test, n_splits=5):
        kf = KFold(n_splits=n_splits, random_state=0)
        n_train = x_train.shape[0]
        n_test = x_test.shape[0]
        train_predict = np.zeros((n_train,))
        test_predict = np.zeros((n_test,))
        test_predict_skf = np.empty((n_splits, n_test))
        for i, (train_index, test_index) in enumerate(kf.split(x_train)):
            x_tr = x_train[train_index]
            y_tr = y_train[train_index]
            x_te = x_train[test_index]
            self.train(x_tr, y_tr)
            train_predict[test_index] = self.predict(x_te)
            test_predict_skf[i, :] = self.predict(x_test)
        test_predict[:] = test_predict_skf.mean(axis=0)
        return train_predict, test_predict

class BaseClassifiers():

    def __init__(self, seed=0):
        self.seed = seed

        # random forest parameters
        self.rf_params = {
            'n_jobs':-1,
            'n_estimators': 500,
            'warm_start':True,
            'max_depth':6,
            'min_samples_split':2,
            'verbose':0
        }

        # extra trees parameters
        self.et_params = {
            'n_jobs':-1,
            'n_estimators':500,
            'max_depth': 8,
            'min_samples_leaf': 2,
            'verbose': 0
        }

        # adaboost parameters
        self.ada_params = {
            'n_estimators': 500,
            'learning_rate': 0.75
        }

        # gradient boosting parameters
        self.gb_params = {
            'n_estimators': 500,
            'max_depth': 5,
            'min_samples_leaf': 2,
            'verbose': 0
        }

        # support vector classifier parameters
        self.svc_params = {
            'kernel': 'linear',
            'C': 0.025
        }

        self._creat_clfs()

    # 创建基本分类器
    def _creat_clfs(self):
        rf = Classifier(clf=RandomForestClassifier, params=self.rf_params)
        et = Classifier(clf=ExtraTreesClassifier, params=self.et_params)
        ada = Classifier(clf=AdaBoostClassifier, params=self.ada_params)
        gb = Classifier(clf=GradientBoostingClassifier, params=self.gb_params)
        svc = Classifier(clf=SVC, params=self.svc_params)
        self.clfs = {'rf':rf, 'et':et, 'ada':ada, 'gb':gb, 'svc':svc}

    # 利用基本分类器得到预测结果，并且组合在一起，待次级分类器使用
    def get_outputs(self, x_train, y_train, x_test):
        n_train = x_train.shape[0]
        n_test = x_test.shape[0]
        n_clfs = len(self.clfs)
        train = np.empty((n_train, n_clfs))
        test = np.empty((n_test, n_clfs))

        for i,clfname in enumerate(self.clfs):
            train[:,i], test[:,i] = self.clfs[clfname].get_prediction(x_train, y_train, x_test)
        base_predictions = pd.DataFrame(train, columns=['rf', 'et', 'ada', 'gb', 'svc'])
        # 不同基本分类器的预测结果之间的的协方差越小， stacking结果越好
        sns.heatmap(base_predictions.astype(float).corr(), fmt='.2f', annot=True)
        plt.show()
        print('base classifiers training complete!')
        return train, test

if __name__ == "__main__":
    # 已经清洗过的CSV数据
    train = pd.read_csv('train_cleaned.csv')
    test = pd.read_csv('test_cleaned.csv')

    # 得到numpy格式的训练集，测试集
    y_train = train.Survived.ravel()
    train = train.drop(['Survived'], axis=1)
    x_train = train.values
    x_test = test.values

    # 创建基本分类器， 并输出结果
    baseclf = BaseClassifiers()
    x_train, x_test = baseclf.get_outputs(x_train, y_train, x_test)

    # 创建次级分类器
    xgbm = xgb.XGBClassifier(
        n_estimators=2000,
        max_depth=4,
        min_child_weight=2,
        gamma=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        nthread=-1,
        scale_pos_weight=1
    )

    # 将base分类器输出作为次级分类器的输入，进行训练
    xgbm.fit(x_train, y_train)
    # 预测
    y_pred = xgbm.predict(x_test)
    PassengerId = np.arange(892, 1310)
    # 保存结果
    submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_pred})
    submission.to_csv('submisson7.csv', index=False)
