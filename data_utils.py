# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 20:41:23 2019

@author: aleks
"""

import params
import pandas as pd
from joblib import dump, load
from process_appsflyer import process_appsflyer_dataset
from process_positions import process_positions_dataset
from process_events import process_events_dataset
from utils import calculate_intersection_and_differences
import numpy as np


def read_original_files():
    
    deposits = pd.read_csv(params.deposits, delimiter=params.delimiter)
    positions = pd.read_csv(params.positions, delimiter=params.delimiter)
    appsflyer = pd.read_csv(params.appsflyer, delimiter=params.delimiter)
    events = pd.read_csv(params.events, delimiter=params.delimiter)
    
    return deposits, positions, appsflyer, events

def get_datasets(create=False):
    
    
    if create:
        
        deposits, positions, appsflyer, events = read_original_files()
        
        positions_stats = process_positions_dataset(positions)
        events_stats = process_events_dataset(events)
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
        
        return positions_stats, events_stats, appsflyer_stats
    

def get_user_ids(show_info=False):
    
    deposits, positions, appsflyer, events = read_original_files()
    # Get unique user_id values
    dep_ids = list(deposits.user_id.unique()) # 212
    pos_ids = list(positions.user_id.unique()) # 2556
    ev_ids = list(events.user_id.unique()) # 299 
    apf_ids = list(appsflyer.user_id.unique()) # 4991
        
    if show_info:
        calculate_intersection_and_differences(pos_ids, dep_ids, ev_ids, apf_ids)
        
    return dep_ids, pos_ids, ev_ids, apf_ids
        
        
def merge_created_datasets(create=False, add_target=True, dump_df=False):
    
    positions_stats, events_stats, appsflyer_stats = get_datasets(create=False)
    
    merged_dataset = pd.merge(events_stats, positions_stats, on='user_id', how='right')
    merged_dataset.fillna(0, inplace=True)
    # Merge with appsflyer
    merged_dataset = pd.merge(merged_dataset, appsflyer_stats, on='user_id', how='left')
    merged_dataset.fillna(0, inplace=True)
    
    if add_target:
        merged_dataset = add_target_var(merged_dataset) 
        
    if dump_df:
        dump(merged_dataset, 'data/merged_dataset.joblib')
    
    return merged_dataset

def add_target_var(df):
    
    dep_ids, pos_ids, ev_ids, apf_ids = get_user_ids()
    
    # Add target var
    df['target'] = 0
    df.loc[df['user_id'].isin(dep_ids),'target'] = 1
    
    return df

def trim_dataset_with_relevant_features(df, relevant_feats):
    #relevant_feats = relevant_feats[:50]
    relevant_feats.extend(["target", "user_id"])
    df = df[relevant_feats]
    return df
    
def prepare_dataset_for_training(df):
    X = np.array(df.drop(['target', 'user_id'], axis=1))
    y = df['target'].values
    return X, y

