#%%
from cgi import test
from copyreg import pickle
from re import sub
from statistics import mean
from unicodedata import numeric
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from requests import head
import seaborn as sns
import pandas_profiling
from ipywidgets import HTML, Button, widgets
from pycaret.regression import *
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sqlalchemy import true

#pandasのカラムが100列まで見れるようにする
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)
pd.set_option('display.max_colwidth',None)
sns.set()


# %%
train = pd.read_csv('csv_data/train.csv')
test = pd.read_csv('csv_data/test.csv')
# %%
train.head()
# %%
train.shape
# %%
"""欠損値確認"""
print('train:')
print(train.dtypes)
#%%
print('Test:')
print(test.dtypes)
# %%
print('Train:')
print(train.isnull().sum())
# %%
print('Test:')
print(test.isnull().sum())
# %%
"""カラムの分布を確認"""
train.describe(include='all')
# %%
test.describe(include='all')
# %%
"""欠損値の穴埋め"""
#obj型のデータに対してNAで埋める
null_to_na_cols = ['Alley', 'MasVnrType','BsmtQual', 
                   'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                   'BsmtFinType2', 'FireplaceQu', 'GarageType',
                   'GarageFinish', 'GarageQual', 'GarageCond', 
                   'PoolQC', 'Fence', 'MiscFeature']
train[null_to_na_cols] = train[null_to_na_cols].fillna('NA')
test[null_to_na_cols] = test[null_to_na_cols].fillna('NA')
# %%
print('Train:')
print(train.isnull().sum())
# %%
print('Test:')
print(test.isnull().sum())
# %%
train.describe(include='all')
# %%
test.describe(include='all')
# %%
train[['YearBuilt', 'GarageYrBlt']].corr()

# %%
train.drop(columns=['GarageYrBlt', 'Utilities'], inplace=True)
test.drop(columns=['GarageYrBlt', 'Utilities'], inplace=True)
# %%
#数値型のデータは平均値と最頻値で穴埋め
ms_zones = train['MSZoning'].unique()
for zone in ms_zones:
  zone_mean_train = train[train['MSZoning'] == zone]['LotFrontage'].mean()
  zone_mean_test = train[train['MSZoning'] == zone]['LotFrontage'].mean()
  train[train['MSZoning'] == zone]['LotFrontage'].fillna(zone_mean_train, inplace=True)
  test[test['MSZoning'] == zone]['LotFrontage'].fillna(zone_mean_test, inplace=True)

train.fillna(train.mean(), inplace=True)
test.fillna(test.mean(), inplace=True)
train.fillna(train.mode().iloc[0], inplace=True)
test.fillna(test.mode().iloc[0], inplace=True)
# %%
print('Train:')
print(train.isnull().sum())
# %%
print('Test:')
print(test.isnull().sum())
# %%
"""
--振り返り用メモ--
ターゲットに対して有意な寄与を持つ列を調べる。
相関の欠点の1つは、線形関係のみを測定することである。
別の尺度は相互情報で、特定の変数の値に関する知識が別の変数の分散をどの程度減少させるかを測定する。
特徴工学コースから抜粋(https://www.kaggle.com/learn/feature-engineering)
"""
#%%
from sklearn.feature_selection import mutual_info_regression
cols = train.drop(columns=['Id', 'SalePrice']).columns

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    
mi_scores = make_mi_scores(train[cols], train['SalePrice'])

plt.subplots(figsize=(10,14))
plot_mi_scores(mi_scores.head(40))
# %%
plt.subplots(figsize=(10,14))
plot_mi_scores(mi_scores.tail(40))
# %%
"""TOP45のカラムに絞る"""
top_cols = mi_scores.head(45).index.values.tolist()
top_cols.append('SalePrice')
new_train = train[top_cols]
top_cols.remove('SalePrice')
new_test = test[top_cols]
# %%
new_train.head()
new_train.info()
# %%
"""カテゴリ変数を数値特徴に変換"""
ord_encoding_cols = ['ExterQual', 'HeatingQC', 'KitchenQual']
quality_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}

new_train[ord_encoding_cols] = new_train[ord_encoding_cols].replace(quality_map)
new_test[ord_encoding_cols] = new_test[ord_encoding_cols].replace(quality_map)

new_train.head()
# %%
ord_cols = ['OverallQual','OverallCond','ExterQual','HeatingQC','KitchenQual']
numeric_cols = list(new_train.select_dtypes(['int64','float64']).drop(columns=ord_cols).columns.values)
# %%
dummy_data_train = pd.get_dummies(new_train.select_dtypes(['object','category']))
new_new_train = pd.concat([dummy_data_train,new_train[ord_cols],new_train[numeric_cols]], axis=1)

numeric_cols.remove('SalePrice')
dummy_data_test = pd.get_dummies(new_test.select_dtypes(['object', 'category']))
new_new_test = pd.concat([dummy_data_test, new_test[ord_cols], new_test[numeric_cols]], axis=1)

new_new_train.head()
# %%
"""GarageFinish_NAとBsmtFinType1_NAはGarageType_NAとBsmtQual_NA同じなのでDrop"""
duplicate_cols = ['GarageFinish_NA', 'BsmtFinType1_NA']
new_new_train = new_new_train.drop(columns=duplicate_cols)
new_new_test = new_new_test.drop(columns=duplicate_cols)
# %%
new_new_train.shape
# %%
"""モデル作成と検証"""
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
# %%
cols = new_new_train.drop(columns=['SalePrice']).columns
# %%
X_train, X_val, y_train, y_val = train_test_split(new_new_train[cols],new_new_train['SalePrice'], test_size=15, random_state=1)
# %%
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.fit_transform(X_val[numeric_cols])
# %%
lr = LinearRegression()
lr.fit(X_train,y_train)
# %%
scores = cross_val_score(lr,X_train,y_train,scoring='neg_root_mean_squared_error', cv=10)
# %%
print(np.mean(scores))
# %%
scores = cross_val_score(lr,X_val,y_val,scoring='neg_root_mean_squared_error',cv=10)
# %%
print(np.mean(scores))
# %%
#r2スコア確認
r2_score(y_train, lr.predict(X_train))
#外れ値があると推測
# %%
# 分布確認
plt.subplots(figsize=(10,6))
sns.distplot(train['SalePrice'])
# %%
#対数変換して正規分布化
plt.subplots(figsize=(10,6))
sns.distplot(np.log(train['SalePrice']))
# %%
#再度モデルの実行
lr = LinearRegression()
lr.fit(X_train, np.log(y_train))

scores = cross_val_score(lr, X_train, np.log(y_train), scoring='neg_root_mean_squared_error', cv=10)
print('Train:')
print(np.mean(scores))

#%%
scores = cross_val_score(lr, X_val, np.log(y_val), scoring='neg_root_mean_squared_error', cv=10)
print('Validation:')
print(np.mean(scores))
# %%
r2_score(y_val, np.exp(lr.predict(X_val)))

#全開よりもスコアが上がったが、それでも大きい
# %%
#線形回帰モデル リッジ回帰とラッソ回帰を両方試してみる
from sklearn.linear_model import Ridge

#リッジ回帰
rdg = Ridge(random_state=1)
rdg.fit(X_train,np.log(y_train))
# %%
scores = cross_val_score(rdg, X_train, np.log(y_train), scoring='neg_root_mean_squared_error', cv=10)
print('Train:')
print(np.mean(scores))
# %%
scores = cross_val_score(rdg, X_val, np.log(y_val), scoring='neg_root_mean_squared_error', cv=10)
print('Validation:')
print(np.mean(scores))
# %%
r2_score(y_val, np.exp(rdg.predict(X_val)))
"""
Train:-0.1439610505841074
Validation:-0.1458451415937467
R2:0.8592927124071227
""" 
# %%
#ラッソ回帰
lasso = Lasso(random_state=1)
lasso.fit(X_train, np.log(y_train))
# %%
scores = cross_val_score(lasso, X_train, np.log(y_train), scoring='neg_root_mean_squared_error', cv=10)
print('Train:')
print(np.mean(scores))
# %%
scores = cross_val_score(lasso, X_val, np.log(y_val), scoring='neg_root_mean_squared_error', cv=10)
print('Validation:')
print(np.mean(scores))
# %%
r2_score(y_val, np.exp(lasso.predict(X_val)))

"""
Train:-0.39704137275311796
Validation:-0.4333672566186643
R2:-0.1608939409579635
""" 
# %%
#非線形モデル Random Forest, K-Nearest Neighbors, Support Vector Machine を試す
#RabomForest
from sklearn.ensemble import RandomForestRegressor
# %%
rf = RandomForestRegressor(random_state=1)
rf.fit(X_train, np.log(y_train))
# %%
scores = cross_val_score(rf, X_train, np.log(y_train), scoring='neg_root_mean_squared_error', cv=10)
print('Train:')
print(np.mean(scores))
# %%
scores = cross_val_score(rf, X_val, np.log(y_val), scoring='neg_root_mean_squared_error', cv=10)
print('Validation:')
print(np.mean(scores))
# %%
r2_score(y_val, np.exp(rf.predict(X_val)))
"""
Train:-0.1395849166293402
Validation:-0.27349247213070627
R2:0.9357661349499017
"""
# %%
#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
# %%
knn = KNeighborsRegressor()
knn.fit(X_train, np.log(y_train))
# %%
scores = cross_val_score(knn, X_train, np.log(y_train), scoring='neg_root_mean_squared_error', cv=10)
print('Train:')
print(np.mean(scores))
# %%
scores = cross_val_score(knn, X_val, np.log(y_val), scoring='neg_root_mean_squared_error', cv=10)
print('Validation:')
print(np.mean(scores))
# %%
r2_score(y_val, np.exp(knn.predict(X_val)))
"""
Train:-0.15659213317015908
Validation:-0.296969001868398
R2:0.8714773941116418
"""
# %%
#Support Vector Machine
from sklearn.svm import SVR
# %%
svm = SVR()
svm.fit(X_train, np.log(y_train))
# %%
scores = cross_val_score(svm, X_train, np.log(y_train), scoring='neg_root_mean_squared_error', cv=10)
print('Train:')
print(np.mean(scores))

# %%
scores = cross_val_score(svm, X_val, np.log(y_val), scoring='neg_root_mean_squared_error', cv=10)
print('Validation:')
print(np.mean(scores))
# %%
r2_score(y_val, np.exp(svm.predict(X_val)))

"""
Train:-0.12184840383734177
Validation:-0.2447649338526376
R2:0.9256370354553765
"""
# %%
#ハイパーパラメータの調整
from sklearn.model_selection import GridSearchCV
# %%
svm = SVR()
hyperparameters = {
  'kernel':['linear','poly','rbf'],
  'C':[2,1.5,1,.75],
  'gamma':[.0075, .005, .0025, .001]
}

gs = GridSearchCV(svm, param_grid=hyperparameters, cv=5, scoring='neg_root_mean_squared_error')
gs.fit(X_train, np.log(y_train))
# %%
best_params = gs.best_params_
best_score = gs.best_score_
# %%
print(best_params) 
print(best_score)
# %%
#ベストパラメータをセットして再度
svm = SVR(kernel='rbf', C=2, gamma=.0025)
svm.fit(X_train, np.log(y_train))
# %%
scores = cross_val_score(svm, X_train, np.log(y_train), scoring='neg_root_mean_squared_error', cv=10)
print('Train:')
print(np.mean(scores))
# %%
scores = cross_val_score(svm, X_val, np.log(y_val), scoring='neg_root_mean_squared_error', cv=10)
print('Validation:')
print(np.mean(scores))
# %%
r2_score(y_val, np.exp(svm.predict(X_val)))

"""
Train:-0.1217067652883432
Validation:-0.21632269156258582
R2:0.9315029529558645

"""
# %%
#ファイル番号を決める
import glob
import re
import os

subfile = len(glob.glob('csv_data/submission*'))
file_no = subfile +1

#%%
"""このモデルでsubmit"""
#submit用関数
def save_submission(model, cols, filename='csv_data/submission'+str(file_no)+'.csv'):
    test_data = new_new_test[cols]
    predictions = np.exp(model.predict(test_data))
    ids = test['Id']
    submission_df = {"Id": ids,
                    'SalePrice': predictions}
    submission = pd.DataFrame(submission_df)
    submission.to_csv(filename, index=False)
# %%
new_new_test[numeric_cols] = scaler.fit_transform(new_new_test[numeric_cols])
# %%
not_in_test = ['GarageQual_Ex', 'Exterior2nd_Other', 'HouseStyle_2.5Fin', 'Exterior1st_Stone', 'Exterior1st_ImStucc']
# %%
test_cols = list(cols.copy())
# %%
for i in not_in_test:
    test_cols.remove(i)
# %%
svm = SVR(kernel='rbf', C=2, gamma=.0025)
svm.fit(X_train[test_cols], np.log(y_train))
# %%
save_submission(svm, test_cols)
# %%
