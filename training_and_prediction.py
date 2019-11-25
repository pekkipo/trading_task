# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:36:32 2019

@author: aleks
"""

from data_utils import merge_created_datasets, prepare_dataset_for_training, trim_dataset_with_relevant_features
from feature_selection import rfe_selection, pierson_correlation, load_optimal_features
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import utils
from sklearn.tree import DecisionTreeClassifier    
import lightgbm as lgb

# %% Load data

dataset = merge_created_datasets()

# load selected features
# Choose one of those
features_active = load_optimal_features('rfe')
#features_active = load_optimal_features('piers')

# Add some features from my previous observations
features_active.extend(['deals_per_balance_type_1_count', 'instrument_strike_sum_balance_type_1',
                    'ids_events_count', 'instrument_strike_q75', 'deals_per_position_type_long_count'])

dataset = trim_dataset_with_relevant_features(dataset, features_active)
X, y = prepare_dataset_for_training(dataset)

# %% Feature selection
# This is needed if no features were created previously and saved in feats folder
#feats_rfe = rfe_selection(dataset, X, y)
# Check correlation among features already chosen with rfe 
#feats_piers = pierson_correlation(dataset)

# Var needed just for model and plots names
feats_choice = 'rfe'

# %% Training and prediction Decision Trees

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# baseline model
# Train
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dump(dt, 'models/dt_model_{}.joblib'.format(feats_choice))

# Predict and evaluate
dt_auc = utils.evaluate(dt, 'Decision Trees' + feats_choice, X_test, y_test)



# %% Training and prediction LGBM
# LGBM with GridSearch
# Training auc 0.97, test auc 0.94
params = {
    'application': 'binary',
    'boosting': 'gbdt',
    'num_iterations': 100, 
    'learning_rate': 0.05,
    'num_leaves': 62,
    'max_depth': -1, 
    'max_bin': 510,
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
lbgm_auc = utils.evaluate(lgb_model, 'LGBM'+ feats_choice, X_test, y_test)
dump(lgb_model, 'models/lgbm_model_{}.joblib'.format(feats_choice))


# %% Feature importance
# %% Load mode and show feature importance
ds = dataset.drop(['target', 'user_id'], axis=1)
lgbm_importance = lgb_model.feature_importance()
utils.plotImp(lgbm_importance, ds, 20, "lgbm_importance_{}".format(feats_choice))

dt_importance = dt.feature_importances_
utils.plotImp(dt_importance, ds, 20, "dt_importance_{}".format(feats_choice))
