import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# stacking models

train = pd.read_csv('houseprice_train.csv')
y_train = train.SalePrice.values
train = train.drop('SalePrice', axis=1)
test = pd.read_csv('houseprice_test.csv')

# validate function
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring = 'neg_mean_squared_error', cv=kf))
    return (rmse)

# lasso regression
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.001, random_state=1))
# ElasticNet regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=0.9, random_state=3))
# kernel ridge regressoion
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# gradient boosting regression
GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)
# xgboost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.46, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=1000, reg_alpha=0.464, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7, nthread=-1)
# lightGBM
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720,  max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5, feature_fraction = 0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

# model stacking : adding a meta model
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.zeros((X.shape[0], len(self.base_models_)))
        for i, base_models in enumerate(self.base_models_):
            meta_features[:, i] = np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
        pred = self.meta_model_.predict(meta_features)
        return pred

# define a rmsle evaluation function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# stacked average models
stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print('stacked train', rmsle(y_train, stacked_train_pred))

# xgboost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print('xgb train', rmsle(y_train, xgb_train_pred))

#  lightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print('lgb train', rmsle(y_train, lgb_train_pred))

# final predict
ensemble = 0.7 * stacked_pred + 0.15 * xgb_pred + 0.15 * xgb_pred

# submission
sub = pd.DataFrame()
sub['Id'] = np.arange(1461, 2920)
print(len(np.arange(1461, 2920)))
sub['SalePrice'] = ensemble
sub.to_csv('submission3.csv', index=False)
