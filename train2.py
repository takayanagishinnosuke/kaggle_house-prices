#%% 
from cgi import test
from copyreg import pickle
from re import sub
from statistics import mean
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

#%%
train_df = pd.read_csv('csv_data/train.csv')
test_df = pd.read_csv('csv_data/test.csv')
# %%
#サマリ化して全体を確認(htmlで保存。バグるので実行しない)
# pandas_profiling.ProfileReport(train_df)
# %%
# 学習データとテストデータのマージ
#trainとtestで分かるようフラグ
train_df['WhatlsData'] = 'Train' 
test_df['WhatlsData'] = 'Test'
test_df['SalePrice'] = 99999999999
# %%
# マージ
all_df = pd.concat([train_df,test_df],axis=0).reset_index(drop=True)
# %%
print('This size train is:'+str(train_df.shape))
print('This size test is:'+str(test_df.shape))
# %%
"""欠損値補完"""
#欠損値確認
train_df.isnull().sum()[train_df.isnull().sum()>0].sort_values(ascending=False)
test_df.isnull().sum()[test_df.isnull().sum()>0].sort_values(ascending=False)
# %%
#欠損を含むカラムをリスト化
na_col_list = all_df.isnull().sum()[all_df.isnull().sum()>0].index.tolist()
# %%
#データ型の確認
all_df[na_col_list].dtypes.sort_values()
# %%
#データ型に応じて欠損値を補完する
#floatの場合は0
#objectの場合は'NA'

#float64データ型のカラムをリスト化
na_float_cols = all_df[na_col_list].dtypes[all_df[na_col_list].dtypes=='float64'].index.tolist()
#objectデータ型のカラムをリスト化
na_obj_cols = all_df[na_col_list].dtypes[all_df[na_col_list].dtypes=='object'].index.tolist()

print(na_float_cols)
print(na_obj_cols)

#%%
#float64で欠損している場合は0を代入
for na_float_col in na_float_cols:
  all_df.loc[all_df[na_float_col].isnull(),na_float_col] = 0.0

# object型で欠損している場合はNAを代入
for na_obj_col in na_obj_cols:
  all_df.loc[all_df[na_obj_col].isnull(),na_obj_col] = 'NA'
# %%
# 確認
all_df.isnull().sum()[all_df.isnull().sum()>0].sort_values(ascending=False)
# %%
# カテゴリ変数の特徴量をリスト化
cat_cols = all_df.dtypes[all_df.dtypes=='object'].index.tolist()
print(cat_cols)
# %%
# 数値変数の特徴量をリスト化
num_cols = all_df.dtypes[all_df.dtypes!='object'].index.tolist()
print(num_cols)
# %%
#データ分割及び提出時に必要なカラムをリスト化
other_cols = ['Id','WhatlsData']
#余計な要素をリストから削除
cat_cols.remove('WhatlsData') #学習データ、テストデータ区別フラグ除去
num_cols.remove('Id') #id削除
# %%
#カテゴリカル変数をダミー化
all_df_cat = pd.get_dummies(all_df[cat_cols])
# %%
all_df_cat
# %%
#データ統合
all_df = pd.concat([all_df[other_cols],all_df[num_cols],all_df_cat],axis=1)
all_df
# all_df.to_csv('all_df.csv',index=False)
# %%
# 目的変数の分布変換
# 分布確認
sns.distplot(train_df['SalePrice'])
# %%
# 対数変換して正規分布化
sns.distplot(np.log(train_df['SalePrice']))

# %%
#マージデータを学習データとテストデータに分割
train_ = all_df[all_df['WhatlsData']=='Train'].drop(['WhatlsData','Id'],axis=1).reset_index(drop=True)
test_ = all_df[all_df['WhatlsData']=='Test'].drop(['WhatlsData','SalePrice'],axis=1).reset_index(drop=True)

# %%
#学習データ内の分割
train_x = train_.drop('SalePrice',axis=1)
train_y = np.log(train_['SalePrice'])
# %%
#テストデータ内の分割
test_id = test_['Id']
test_data = test_.drop('Id',axis=1)
# %%
"""予測モデルの構築"""

scaler = StandardScaler() #スケーリング
param_grid = [0.001,0.01,0.1,1.0,10.0,100.0,1000.0] #パラメータグリッド
cnt = 0
for alpha in param_grid:
  ls = Lasso(alpha=alpha) #Lasso回帰モデル
  pipeline = make_pipeline(scaler, ls) #パイプライン生成
  X_train,X_test,y_train,y_test = train_test_split(train_x,train_y,test_size=0.3,random_state=0)
  pipeline.fit(X_train,y_train)
  train_rmse = np.sqrt(mean_squared_error(y_train,pipeline.predict(X_train)))
  test_rmse = np.sqrt(mean_squared_error(y_test,pipeline.predict(X_test)))

  if cnt ==0:
    best_score = test_rmse
    best_estimator = pipeline
    best_param = alpha
  elif best_score > test_rmse:
    best_score = test_rmse
    best_estimator = pipeline
    best_param = alpha
  else:
    pass
  cnt = cnt + 1

print('alpha:' +str(best_param))
print('test score is:' +str(best_score))
# %%
"""検証"""
plt.subplots_adjust(wspace=0.4)
plt.subplot(121)
plt.scatter(np.exp(y_train),np.exp(best_estimator.predict(X_train))) #学習データ
plt.subplot(122)
plt.scatter(np.exp(y_test),np.exp(best_estimator.predict(X_test))) #テストデータ

# %%
"""Submit"""
ls = Lasso(alpha=0.01)
pipeline = make_pipeline(scaler,ls)
pipeline.fit(train_x,train_y)
test_SalePrice = pd.DataFrame(np.exp(pipeline.predict(test_data)),columns=['SalePrice'])
test_SalePrice
# %%
test_id = pd.DataFrame(test_id,columns=['Id'])
pd.concat([test_id,test_SalePrice],axis=1).to_csv('csv_data/submission2.csv', index=False)
# %%
