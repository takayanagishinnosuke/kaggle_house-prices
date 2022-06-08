#%%
from cgi import test
from copyreg import pickle
from re import sub
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from requests import head
import seaborn as sns
import pandas_profiling
from ipywidgets import HTML, Button, widgets
from pycaret.regression import *
from sqlalchemy import true

#pandasのカラムが100列まで見れるようにする
pd.set_option('display.max_columns',100)

# %%
# データ読み込み
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# %%
# 確認
train_data
#%%
test_data
# %%
#全体俯瞰
train_data.info()
# %%
train_data.isnull().sum()
# %%
#EDA
train_data.describe()

# %%
#データレポート化
train_data.profile_report()

#%%
#train_dataのSalePriceを対数に変換する
# train_data['SalePrice'] = np.log(train_data['SalePrice'])

#%%
#train_dataとtest_dataをマージ
data_all = pd.concat([train_data,test_data], sort=True)
data_all

# %%
"""pycaretに突っ込んでみる"""
clf1 = setup(data_all,
            target='SalePrice',
            numeric_imputation='median',
            categorical_imputation='mode',
            )
# %%
# 前処理された全ての説明変数
X = get_config('X')
X
# %%
# 全ての目的変数
y = get_config('y')
y
# %%
#分割された学習データ
X_train = get_config('X_train')
X_train.head()
# %%
#分割された学習用目的変数
y_train = get_config('y_train')
y_train.head()
# %%
#分割されたテストデータ(説明変数)
X_test = get_config('X_test')
X_test.head()
# %%
#分割されたテストデータ(目的変数)
y_test = get_config('y_test')
y_test.head()
# %%
#前処理した学習データとテストデータを結合して前処理後のデータセットを作成
df = pd.merge(X,y,left_index=True,right_index=True)
df
# %%
#前処理したデータをcsv保存
df.to_csv('train_test.csv')
# %%
#モデル作成と比較
top_model = compare_models(sort='RMSE',n_select=5)
# %%
# モデル精度の確認
print([predict_model(model) for model in top_model])

#%%
#モデル選別
model = create_model('omp')
print(model)
# %%
#アンサンブル学習をさせてみる(バギング)
bagged_model = ensemble_model(model, method='Bagging', optimize='RMSE')
# %%
#アンサンブル学習をさせてみる(ブースティング)
boosted_model = ensemble_model(model, method='Boosting', optimize='RMSE')
# %%
#アンサンブル学習をさせてみる(ブレンディング)
blend_model = blend_models(estimator_list = boosted_model, optimize='RMSE')
# %%
#アンサンブル学習をさせてみる(スタッキング)
stack_model = stack_models(estimator_list= boosted_model, optimize='RMSE')
# %%
#モデル確定
pre_model = finalize_model(model)
print(pre_model)
# %%
#推論の実行
predict_ans = predict_model(pre_model,data=test_data)
predict_ans
# %%
"""提出用ファイルの作成"""
submission_data = pd.read_csv('sample_submission.csv')
submission_data
#%%
submission_data = submission_data.drop('SalePrice', axis=1)
submission_data = submission_data.join(predict_ans['Label'])
submission_data
#%%
submission_data = submission_data.rename(columns={'Label':'SalePrice'})
submission_data

#%%
submission_data.to_csv('submission1.csv',index=False)

#%%
"""EDAパート2"""
#まずは読み込み
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.head()

#%%
#目的変数についても見ておく
sns.distplot(train_df['SalePrice'])

#売却価格の概要をみてみる
print(train_df['SalePrice'].describe())
print(f"歪度: {round(train_df['SalePrice'].skew(),4)}" )
print(f"尖度: {round(train_df['SalePrice'].kurt(),4)}" )

#%%
# データを結合
all_df = pd.concat([train_df,test_df])
all_df.head()
# all_df.isnull().sum()

#%%
clf2 = setup(all_df,
            target='SalePrice',
            )
# %%
# 前処理された全ての説明変数
X = get_config('X')
X
# %%
# 全ての目的変数
y = get_config('y')
y
# %%
#分割された学習データ
X_train = get_config('X_train')
X_train.head()
# %%
#分割された学習用目的変数
y_train = get_config('y_train')
y_train.head()
# %%
#分割されたテストデータ(説明変数)
X_test = get_config('X_test')
X_test.head()
# %%
#分割されたテストデータ(目的変数)
y_test = get_config('y_test')
y_test.head()
# %%
#前処理した学習データとテストデータを結合して前処理後のデータセットを作成
df2 = pd.merge(X,y,left_index=True,right_index=True)
df2
# %%
df2.to_csv('train_test.csv')
# %%
top3 = compare_models(sort='RMSE', n_select=3, fold=3)
# %%
gbr_model = create_model('gbr')

#%%
tuns_model = tune_model(gbr_model)

#%%
#性能を分析
evaluate_model(tuns_model)

"""こんがらがってきたので別ファイルでEDAからやり直す。train2.pyへ"""
