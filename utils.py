# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:55:44 2019

@author: aleks
"""

import pandas as pd
import numpy as np
import json

def fill_instr_types_columns(vals, col_name): 
    return 1 if col_name in vals else 0

def handle_categorical_types(df, column_names, col_name_suffix=None):
    resulting_df = pd.DataFrame() 
    for col in column_names:
        resulting_df[col] = df # now I have the same list in every column     
        # I will turn those lists into single True of False values for this particular column
        resulting_df[col] = resulting_df[col].apply(fill_instr_types_columns, args=([col]))
        # passing a list as a workaround otherwise apply sees each character as a separate argument
        if col_name_suffix:
            new_col_name = col_name_suffix + '{}'.format(col)
            resulting_df.rename(columns={col: new_col_name}, inplace=True)
    return resulting_df
"""
def calculate_amount_of_deals(categories)
for cat in categories: 
    frequency['deals_per_{}_instr_count'.format(instr)] = positions.where(positions.instrument_type == instr).groupby('user_id').size()
    #(positions.groupby('user_id')['instrument_type']. == instr).sum()
    # if the user haven't used certain instrument type the value would be NaN, therefore we replace it with 0
    frequency['deals_per_{}_instr_count'.format(instr)].fillna(0, inplace=True)
"""

"""
Create features for each feature based on its occurence rate
"""
def create_features_based_on_categories(source_df, df, feature, list_of_uniques, suffix="_"):
    """
    Creates features for each feature based on its occurence rate
    
    Parameters: 
    source_df: Original dataset, in this case positions
    df: Target dataset
    feature: column name that is being processed
    list_of_uniques: list of all the values that occure in this column
    suffix: string suffix that will be a part of the new feature names
  
    Returns: 
    df: target dataset with added features
    """
    """ Removed this part as this when I do the counting of values further in this function for each type - the info about using this
    type is already encoded. If count is zero - obviously it is not used
    list_of_types_per_user = source_df.groupby('user_id')[feature].unique()
    df_with_types = handle_categorical_types(list_of_types_per_user, list_of_uniques, suffix)
    df = pd.concat([df, df_with_types], axis=1)
    """
    
    # Calculate amount of deals involving each instrument type
    for one_type in list_of_uniques: 
        df['{}_{}_count'.format(feature, one_type)] = source_df.where(source_df[feature] == one_type).groupby('user_id').size()
        #(positions.groupby('user_id')['instrument_type']. == instr).sum()
        # if the user haven't used certain instrument type the value would be NaN, therefore we replace it with 0
        df['{}_{}_count'.format(feature, one_type)].fillna(0, inplace=True)
       
    return df

"""
Creates features like "uses_platform_x" True of False
"""
def create_features_for_appflyer(source_df, df, feature, list_of_uniques, suffix="_"):
    """
    Creates features for each feature based on its occurence rate
    
    Parameters: 
    source_df: Original dataset, in this case positions
    df: Target dataset
    feature: column name that is being processed
    list_of_uniques: list of all the values that occure in this column
    suffix: string suffix that will be a part of the new feature names
  
    Returns: 
    df: target dataset with added features
    """
    
    list_of_types_per_user = source_df.groupby('user_id')[feature].unique()
    df_with_types = handle_categorical_types(list_of_types_per_user, list_of_uniques, suffix)
    df = pd.concat([df, df_with_types], axis=1)
   
    return df


def handle_json_dict(data):
    d = json.loads(data)
    return d
    

def check_if_nested(parameters_dict):
    return any(isinstance(i, dict) for i in parameters_dict.values())

# These funcs are used cauz I need to pass % parameter to agg quantile func
def q10(x):
    return x.quantile(0.1)
# 90th Percentile
def q90(x):
    return x.quantile(0.9)
        
def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)
        
    
def check_currency_conversions(df, features):
    
    for feature in features:
        # Column that is a result of subtraction a "enrolled" version from the feature
        df['difference'] = df[feature] - df[feature + '_enrolled']
        
        # If the difference is not zero then - conversion happened (1) otherwise not (0)
        df['{}_conversion_happened'.format(feature)] = np.where(df['difference'] == 0, 0, 1)
        
        df.drop(columns=['difference'], inplace=True)
    
    return df

# Intersections
def list_intersection(a, b):
    return list(set(a).intersection(b))

def list_difference(a,b):
    return list(set(a).difference(b))

def calculate_intersection_and_differences(pos_ids, dep_ids, ev_ids, apf_ids):
    # ids who made the deposit
    inter_pos_dep = list_intersection(pos_ids, dep_ids) # 202
    # 202 ids made a deposit
    
    # ids who didn't make a deposit
    dif_pos_dep = list_difference(pos_ids, dep_ids) # 2354
    # 2354 ids didn'make a deposit
    
    # just checking if it adds up and it does
    dif_dep_pos = list_difference(dep_ids, pos_ids) # 10
    
    # deposits ids that are also present in events
    inter_dep_ev = list_intersection(dep_ids, ev_ids) # 132
    # So for 132 ids who made a deposit I have corrspdoning info in events dataset
    
    
    # ids from positions that are also present in events
    inter_pos_events = list_intersection(pos_ids, ev_ids) # 266
    # So for 266 ids who sealed any deals at all I have corresponding info in events table
    
    # let's count for how many ids who DIDN't make a deal I have corresponding events info
    inter_pos_no_dep_events = list_intersection(dif_pos_dep, ev_ids) # 141
    # So for 141 ids who didn't make a deposit I have corresponding info in events table
    
    # ids from positions that are also present in appsflyer
    inter_pos_af = list_intersection(pos_ids, apf_ids) # 2556
    # So for all ids in deals I have corresponding info in appsflyer dataset!
    
    log  = (
            f"Amount of ids that MADE a deposit: {len(inter_pos_dep)} \n"
            f"Amount of ids that DIDN'T make a deposit: {len(dif_pos_dep)} \n"
            f"Amount of ids that MADE a deposit and have corresponding info in the Events dataset: {len(inter_dep_ev)} \n"
            f"Amount of ids that DIDN'T a deposit and have corresponding info in the Events dataset: {len(inter_pos_no_dep_events)} \n"
            f"Amount of ids that are present in Positions dataset and have corresponding info in the Events dataset: {len(inter_pos_events)} \n"
            f"Amount of ids that are present in Positions dataset and have corresponding info in the Appsflyer dataset: {len(inter_pos_af)} \n"
            )
    print(log)
    
    return {
            'inter_pos_dep':inter_pos_dep,
            'dif_pos_dep':dif_pos_dep,
            'inter_dep_ev':inter_dep_ev,
            'inter_pos_no_dep_events':inter_pos_no_dep_events,
            'inter_pos_events':inter_pos_events,
            'inter_pos_af':inter_pos_af,        
            }
        