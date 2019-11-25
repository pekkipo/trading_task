# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 02:14:52 2019

@author: aleks
"""


from joblib import dump, load
import pandas as pd
import params
from sklearn.model_selection import (train_test_split, GridSearchCV)
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

numbers = load("data/numbers_df.joblib")
frequencies = load("data/frequency_df.joblib")
events_stats = load("data/events_stats_df.joblib")

deposits = pd.read_csv(params.deposits, delimiter=params.delimiter)
positions = pd.read_csv(params.positions, delimiter=params.delimiter)
appsflyer = pd.read_csv(params.appsflyer, delimiter=params.delimiter)
events = pd.read_csv(params.events, delimiter=params.delimiter)

dep_ids = list(deposits.user_id.unique()) # 212
pos_ids = list(positions.user_id.unique()) # 2556
ev_ids = list(events.user_id.unique()) # 299 I ll have to go with just true or false event feature

# the order was correct, so two dataset can just be concatenated
positions_stats = pd.concat([numbers, frequencies], axis=1)
positions_stats.reset_index(inplace=True)
positions_stats.rename(columns={"index": "user_id"}, inplace=True)


events_stats.reset_index(inplace=True)
events_stats.rename(columns={"index": "user_id"}, inplace=True)

#test1 = positions_stats.merge(events_stats, left_on='user_id', right_on='rkey', how='outer')


merged_dataset = pd.merge(events_stats, positions_stats, on='user_id', how='right')
merged_dataset.fillna(0, inplace=True)


print("Done")

# let's add target var here to see how it relates to potential categories
merged_dataset['target'] = 0
merged_dataset.loc[merged_dataset['user_id'].isin(dep_ids),'target'] = 1


X = np.array(merged_dataset.drop(['target'], axis=1))
y = merged_dataset['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 12)



params = {
    'application': 'binary',
    'boosting': 'gbdt', # traditional gradient boosting decision tree
    'num_iterations': 100, 
    'learning_rate': 0.05,
    'num_leaves': 62,
    'max_depth': -1, # <0 means no limit
    'max_bin': 510, # Small number of bins may reduce training accuracy but can deal with over-fitting
    'lambda_l1': 5, # L1 regularization
    'lambda_l2': 10, # L2 regularization
    'metric' : 'auc', #'binary_error',
    'subsample_for_bin': 200, # number of samples for constructing bins
    'subsample': 1, # subsample ratio of the training instance
    'colsample_bytree': 0.8, # subsample ratio of columns when constructing the tree
    'min_split_gain': 0.5, # minimum loss reduction required to make further partition on a leaf node of the tree
    'min_child_weight': 1, # minimum sum of instance weight (hessian) needed in a leaf
    'min_child_samples': 5# minimum number of data needed in a leaf
}

# Initiate classifier to use
model = lgb.LGBMClassifier(boosting_type= 'gbdt', 
          objective = 'binary', 
          n_jobs = 5, 
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'], 
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'], 
          min_split_gain = params['min_split_gain'], 
          min_child_weight = params['min_child_weight'], 
          min_child_samples = params['min_child_samples'])



grid_params = {
    'learning_rate': [0.005, 0.01],
    'n_estimators': [8,16,24],
    'num_leaves': [6,8,12,16], # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
    'objective' : ['binary'],
    'max_bin':[255, 510], # large max_bin helps improve accuracy but might slow down training progress
    'random_state' : [500],
    'colsample_bytree' : [0.64, 0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }

grid = GridSearchCV(model, grid_params, verbose=1, cv=4, n_jobs=-1)
# Run the grid
grid.fit(X, y)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)


params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate'] 
params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
params['reg_alpha'] = grid.best_params_['reg_alpha']
params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state = 12)
    

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid) 

watchlist = [d_train, d_valid]


model = lgb.train(params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=4)

predictions = model.predict(X_test)
auc_lgb  = round(roc_auc_score(y_test, predictions), 4)
print('\nLightGBM VAL AUC: {}'.format(auc_lgb))

dump(model, 'lgbm_model.joblib')

feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')