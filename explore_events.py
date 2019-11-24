# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:15:21 2019

@author: aleks
"""

import params
import pandas as pd
from utils import list_intersection, list_difference, calculate_intersection_and_differences, fill_instr_types_columns, handle_categorical_types, create_features_based_on_categories, q10, q90, check_currency_conversions
from joblib import dump
import ast
import json

"""
Read datasets
"""
deposits = pd.read_csv(params.deposits, delimiter=params.delimiter)
positions = pd.read_csv(params.positions, delimiter=params.delimiter)
appsflyer = pd.read_csv(params.appsflyer, delimiter=params.delimiter)
events = pd.read_csv(params.events, delimiter=params.delimiter)

# Replace NaNs in duration with 0s
print(events.isna().any())
events['duration'].fillna(0, inplace=True)
events.fillna({'parameters':'{}'}, inplace=True)


"""
dep_ids = list(deposits.user_id.unique()) # 212
pos_ids = list(positions.user_id.unique()) # 2556
ev_ids = list(events.user_id.unique()) # 299 I ll have to go with just true or false event feature
apf_ids = list(appsflyer.user_id.unique()) # 4991 какого хрена так много
"""
events_names = list(events.name.unique()) 
events_platform_ids = list(events.platform_id.unique()) # just 5
events_device_ids = list(events.device_id.unique()) # 178 
events_category = list(events.category.unique()) # 178

# handle date
def handle_dates(df):
    df['time'] = pd.to_datetime(df['time'])
    df['Month'] = df['time'].dt.month
    df['Year'] = df['time'].dt.year
    df['Day'] = df['time'].dt.day
    
    # give total seconds for easier comparison
    df['timestamp_sec'] = df['time'].astype('int64')//1e9
    return df

events = handle_dates(events)
    


def handle_json_dict(data):
    d = json.loads(data)
    return d
    

# unpack parameters
def unpack_parameters_old(df):
    for index, row in df.iterrows():
        str_dict = df.at[index, 'parameters']
        try:
                        
            parameters = ast.literal_eval(str_dict)
            for item in parameters.items():
                df.at[index, item[0]] = item[1]
        except:
            print('ERROR')
            print('{}'.format(index))
            print(str_dict)
            print(" ")     
            
            pars = handle_json_dict(str_dict)
            new_df = pd.io.json.json_normalize(pars, sep='_')
            
            print(new_df)
            
        #for item in parameters.items():
         #       df.at[index, item[0]] = item[1]
    return df


def check_if_nested(parameters_dict):
    return any(isinstance(i,dict) for i in parameters_dict.values())

def unpack_parameters2(df):
    for index, row in df.iterrows():
        str_dict = df.at[index, 'parameters']
        try:
            parameters = ast.literal_eval(str_dict)
        except:
            # get json parameters
            pars = handle_json_dict(str_dict)
            # flatten (needed in case of nested dict)
            parameters = pd.io.json.json_normalize(pars, sep='_')   
            parameters = parameters.to_dict()
            for item in parameters.items():
                df.at[index, item[0]] = item[1]
        
    return df

def unpack_parameters(df):
    df = df.head(500)
    for index, row in df.iterrows():
        str_dict = df.at[index, 'parameters']
        try:
            # get json parameters
            parameters = handle_json_dict(str_dict)
            if check_if_nested(parameters):
                # flatten (needed in case of nested dict)
                parameters = pd.io.json.json_normalize(parameters, sep='_')
                columns = list(parameters)
                for column in columns:
                    value = parameters.at[0, column]
                    df.at[index, column] = value
            else:
                for item in parameters.items():
                    if isinstance(item[1], list):
                        df.at[index, item[0]] = item[1][0]
                    else:
                        df.at[index, item[0]] = item[1]
        except:
            print('ERROR')
        
    return df

events = unpack_parameters(events)
    
    
print("Done")

# duration handle like numeric

# Total time for events

# events per month

# events per day

# Check correlations of events and positions based on time
# like amount of events on each platform

# maybe merge events and positions based on user_id and position_id or smth

