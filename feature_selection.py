# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:58:27 2019

@author: aleks
"""
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load, dump
from sklearn.tree import DecisionTreeClassifier  
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def pierson_correlation(df, save=True, plot=True, valid_percentage=0.2):
    
    df = df.drop(['user_id'], axis=1)
    #Using Pearson Correlation
    cor = df.corr()
    if plot:
        plt.figure(figsize=(12,10))
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()
    #Correlation with output variable
    cor_target = abs(cor["target"])
    #Selecting highly correlated features
    relevant_features = list(cor_target[cor_target>valid_percentage].index)
    relevant_features.remove("target")
    print(relevant_features)
    if save:
        dump(relevant_features, "feats/top_feats_pierson_cor.joblib")
    
    return relevant_features


def rfe_selection(df, X, y, max_feats=60, save=True, drop_cols=['target', 'user_id'], find_optimal=False, optimal=30):
    df = df.drop(drop_cols, axis=1)
    
    ## Determine number of features
    def find_optimal_feat_count(X, y, start = 10, upto=max_feats):
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
    dt = DecisionTreeClassifier()
    if find_optimal:
        nof = find_optimal_feat_count(X, y) 
    else:
        nof = optimal
    print('Optimal number of features: {}'.format(nof)) # 30
    rfe = RFE(dt, nof)             
    # Transform data using RFE
    X_rfe = rfe.fit_transform(X, y) 
     
    dt.fit(X_rfe, y)              
    temp = pd.Series(rfe.support_, index = cols)
    selected_features_list = list(temp[temp==True].index)
    print(selected_features_list)
    if save:
        dump(selected_features_list, "feats/top_feats_dt_rfe.joblib")
        
def load_optimal_features(method='rfe'):
    
    if method=='rfe':
        return load('feats/top_feats_dt_rfe.joblib')
    else:
        return load('feats/top_feats_pierson_cor.joblib')
