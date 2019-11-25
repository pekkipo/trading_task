# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:25:47 2019

@author: aleks
"""


from joblib import dump, load
import pandas as pd
import params
from sklearn.model_selection import (train_test_split, GridSearchCV, RandomizedSearchCV)
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from process_appsflyer import process_appsflyer_dataset
from process_positions import process_positions_dataset
from process_events import process_events_dataset
from utils import calculate_intersection_and_differences
from sklearn.metrics import roc_auc_score


# Read original datasets
deposits = pd.read_csv(params.deposits, delimiter=params.delimiter)
positions = pd.read_csv(params.positions, delimiter=params.delimiter)
appsflyer = pd.read_csv(params.appsflyer, delimiter=params.delimiter)
events = pd.read_csv(params.events, delimiter=params.delimiter)

# Get unique user_id values
dep_ids = list(deposits.user_id.unique()) # 212
pos_ids = list(positions.user_id.unique()) # 2556
ev_ids = list(events.user_id.unique()) # 299 
apf_ids = list(appsflyer.user_id.unique()) # 4991

# don't need it but it prints some info so I ll keep it here
calculate_intersection_and_differences(pos_ids, dep_ids, ev_ids, apf_ids)


# Create datasets that will be used for training (datasets are mostly based on statistics)
if not params.read_prepared_data:
    #positions_stats = process_positions_dataset(positions)
    #events_stats = process_events_dataset(events)
    appsflyer_stats = process_appsflyer_dataset(appsflyer)
else: # if read parameter in params is set to True then just read the data
    positions_stats_numbers = load('data/positions_numbers_df.joblib')
    positions_stats_freq = load('data/positions_frequency_df.joblib')
    # Concatinate two datasets
    positions_stats = pd.concat([positions_stats_numbers, positions_stats_freq], axis=1)
    positions_stats.reset_index(inplace=True)
    positions_stats.rename(columns={"index": "user_id"}, inplace=True)
    
    events_stats_numbers = load('data/events_numbers_df.joblib')
    events_stats_info = load('data/events_stats_df.joblib')
    # Concatinate two datasets
    events_stats = pd.concat([events_stats_numbers, events_stats_info], axis=1)
    events_stats.reset_index(inplace=True)
    events_stats.rename(columns={"index": "user_id"}, inplace=True)
    
    appsflyer_stats = load('data/appsflyer_stats_df.joblib')


# Merged dataset must contain 2556 rows
# 202 rows would contain target var 1, 2354 target var is 0
# Merge datasets
merged_dataset = pd.merge(events_stats, positions_stats, on='user_id', how='right')
merged_dataset.fillna(0, inplace=True)
# Merge with appsflyer
merged_dataset = pd.merge(merged_dataset, appsflyer_stats, on='user_id', how='left')
merged_dataset.fillna(0, inplace=True)

# Add target var
merged_dataset['target'] = 0
merged_dataset.loc[merged_dataset['user_id'].isin(dep_ids),'target'] = 1

dump(merged_dataset, 'data/merged_dataset.joblib')

# Check if the numbers add up (They do)
check_1 = (merged_dataset.target.values == 1).sum()
check_0 = (merged_dataset.target.values == 0).sum()


X = np.array(merged_dataset.drop(['target'], axis=1))
y = merged_dataset['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# I use ROC AUC curve metrics as the dataset is imbalanced, 202 - 1 against 2354 - 0
def evaluate(model, model_name, X_text, y_test):
    predictions = model.predict(X_test)
    model_auc = round(roc_auc_score(y_test, predictions), 4)
    print('\n{} VAL AUC: {}'.format(model_name, model_auc))  
    return model_auc

# %% Random forest with scikit random search

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search training
# Search for best hyperparameters

rf = RandomForestRegressor() # base model to be tuned
# Random search of parameters, using 5 fold cross validation, 
# Search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)

best_random_rf = rf_random.best_estimator_
random_auc = evaluate(best_random_rf, 'Random Forest', X_test, y_test)

dump(best_random_rf, 'rf_model.joblib')
    
# %% LGBM with GridSearch. Will give the feature importance graph
params = {
    'application': 'binary',
    'boosting': 'gbdt',
    'num_iterations': 100, 
    'learning_rate': 0.05,
    'num_leaves': 62,
    'max_depth': -1, 
    'max_bin': 510, -
    'lambda_l1': 5, 
    'lambda_l2': 10,
    'metric' : 'auc', 
    'subsample_for_bin': 200, 
    'subsample': 1, 
    'colsample_bytree': 0.8, 
    'min_split_gain': 0.5, 
    'min_child_weight': 1,
    'min_child_samples': 5
}

# Initiate classifier to use
lgb_model = lgb.LGBMClassifier(boosting_type= 'gbdt', 
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
    'num_leaves': [6,8,12,16], # large num_leaves might lead to over-fitting
    'boosting_type' : ['gbdt', 'dart'], # dart can give better accuracy
    'objective' : ['binary'],
    'max_bin':[255, 510], # large max_bin can slow down training progress
    'random_state' : [500],
    'colsample_bytree' : [0.64, 0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }

grid = GridSearchCV(lgb_model, grid_params, verbose=1, cv=4, n_jobs=-1)
# Run the grid
grid.fit(X_train, y_train)

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



X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.10, random_state = 42)
    

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid) 

watchlist = [d_train, d_valid]


lgb_model = lgb.train(params, train_set=d_train, num_boost_round=1000, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=4)

lbgm_auc = evaluate(lgb_model, 'LGBM', X_test, y_test)

dump(lgb_model, 'lgbm_model.joblib')

feature_imp = pd.DataFrame(sorted(zip(lgb_model.feature_importances_,X.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances.png')

print("Done")




