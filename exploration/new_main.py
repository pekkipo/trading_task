# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:28:29 2019

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


## Some ids have signifficantly more records in positions than others
## and can be considered outliers. 26 of these were found in one of the exploration file and
## saved into a list. 
## Removing them actually doesn't influence the result that much so I will keep these records
#outliers_ids = load('outliers_ids.joblib')
#merged_dataset = merged_dataset[~merged_dataset.user_id.isin(outliers_ids)]
#print(merged_dataset.shape)


# Try removing top feats 
"""
freats_to_remove = ['open_underlying_price_sum_balance_type_1', 
                    'duration_mean',
                    'close_underlying_price_sum_balance_type_1',
                    'sell_amount_mean',
                    'name_get-first-candles_count']
"""
""" Drop some features
feats_to_keep = load("top_N_feats_dt.joblib")
feats_to_keep = feats_to_keep[:40]
feats_to_keep.extend(["target", "user_id"])

#merged_dataset.drop(columns=freats_to_remove, inplace = True)
merged_dataset = merged_dataset[feats_to_keep]
"""


## Create sampled ds
"""
feats_to_keep = load("top_N_feats_dt.joblib")
feats_to_keep = feats_to_keep[:50]
feats_to_keep.extend(["target", "user_id"])
merged_dataset = merged_dataset[feats_to_keep]
"""

X = np.array(merged_dataset.drop(['target', 'user_id'], axis=1))
y = merged_dataset['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

"""
Some convenience funcs
"""
# I use ROC AUC curve metrics as the dataset is imbalanced, 202 - 1 against 2354 - 0
def evaluate(model, model_name, X_test, y_test):
    """
    Shows AUC score
    
    """
    predictions = model.predict(X_test)
    model_auc = round(roc_auc_score(y_test, predictions), 4)
    print('\n{} VAL AUC: {}'.format(model_name, model_auc))  
    return model_auc

def plotImp(impts, X, num = 20, name="feats_importance"):
    """
    Plots importance of features
    
    parameters:
        num: number of features to display
        impts: list of feature importance values
        name: name for the plot
    """
    feature_imp = pd.DataFrame({'Value':impts,'Feature': X.columns})
    plt.figure(figsize=(40, 20))
    sns.set(font_scale = 3)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:num])
    
    
    plt.title('Features importances')
    plt.tight_layout()
    plt.savefig('{}.png'.format(name))
    plt.show()
    


# %% Do some EDA
"""
# Feature selection 1 - using Pierson Correlation
targets = merged_dataset['target']
df = merged_dataset.drop(['user_id'], axis=1)

#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor["target"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
print(relevant_features)

dump(relevant_features, "feats/top_feats_pierson_cor.joblib")
"""

# %% Feature selection - 2 RFE
from sklearn.tree import DecisionTreeClassifier  
from sklearn.feature_selection import RFE
"""
dt = DecisionTreeClassifier()
#Initializing RFE model
rfe = RFE(estimator=dt, n_features_to_select=50, step=1)

#Transforming data using RFE
X_rfe = rfe.fit_transform(X, y)  
#Fitting the data to model
dt.fit(X_rfe,y)

sups = rfe.support_
ranks = rfe.ranking_
print(sups)
print(ranks)
sups = list(sups)


from itertools import compress
list_a = list(df.columns) 
features_to_keep = list(compress(list_a, sups))
dump(features_to_keep, "feats/top_feats_dt_rfe.joblib")
"""
df = merged_dataset.drop(['target', 'user_id'], axis=1)

## Determine number of features
def find_optimal_feat_count(X, y, start = 10, upto=60):
    #no of features
    nof_list=np.arange(start, upto)            
    high_score=0
    #Variable to store the optimum features
    nof=0           
    score_list =[]
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
        model = DecisionTreeClassifier()
        rfe = RFE(model,nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            nof = nof_list[n]
    print("Optimum number of features: {}".format(nof))
    print("Score with {} features: {}".format(nof, high_score))
    return nof

cols = list(df.columns)
#cols = cols[:-1]
dt = DecisionTreeClassifier()

nof = find_optimal_feat_count(X, y) 
print('Optimal number of features: {}'.format(nof)) # 30
rfe = RFE(dt, nof)             
# Transform data using RFE
X_rfe = rfe.fit_transform(X, y) 
 
dt.fit(X_rfe, y)              
temp = pd.Series(rfe.support_, index = cols)
selected_features_list = list(temp[temp==True].index)
print(selected_features_list)
dump(selected_features_list, "feats/top_feats_dt_rfe.joblib")


# %% Simple decision tree. AUC = 0.8544
# Use it for features importance graph
from sklearn.tree import DecisionTreeClassifier    
"""
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dump(dt, 'dt_model_50.joblib')

imps_dt = dt.feature_importances_
print(sorted(imps_dt)) # plot them later
dt_auc = evaluate(dt, 'Decision Tree', X_test, y_test)

targets = merged_dataset['target']
ds = merged_dataset.drop(['target', 'user_id'], axis=1)
plotImp(imps_dt, ds, 50, "dt_importance_50")
"""



