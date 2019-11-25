# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:15:21 2019

@author: aleks
"""

import params
import pandas as pd
from utils import check_if_nested, handle_json_dict, list_intersection, list_difference, calculate_intersection_and_differences, fill_instr_types_columns, handle_categorical_types, create_features_based_on_categories, q10,q25, q75, q90, check_currency_conversions
from joblib import dump
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

"""
Read datasets
"""
deposits = pd.read_csv(params.deposits, delimiter=params.delimiter)
positions = pd.read_csv(params.positions, delimiter=params.delimiter)
appsflyer = pd.read_csv(params.appsflyer, delimiter=params.delimiter)
events = pd.read_csv(params.events, delimiter=params.delimiter)

dep_ids = list(deposits.user_id.unique()) # 212
pos_ids = list(positions.user_id.unique()) # 2556
ev_ids = list(events.user_id.unique()) # 299 I ll have to go with just true or false event feature
apf_ids = list(appsflyer.user_id.unique()) # 4991 

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
events_names = list(events.name.unique()) # 12
events_platform_ids = list(events.platform_id.unique()) # 10
events_device_ids = list(events.device_id.unique()) # 606
events_category = list(events.category.unique()) # 2

# Handle dates
def handle_dates(df):
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['year'] = df['time'].dt.year
    df['day'] = df['time'].dt.day
    
    # give total seconds for easier comparison
    df['timestamp_sec'] = df['time'].astype('int64')//1e9
    return df

events = handle_dates(events)
    
"""
Я помучался, но анпакнул большинство параметров, но периодически все равно вылезают ошибки, с которыми мне уже надоело
бороться. Так или иначе все равно получилась сильно разреженная матрица,
я посмотрел какие там параметры и решил просто убрать все эти фичи. Но код оставляю
"""
def unpack_parameters(df):
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

"""
Я решил кое-что все-таки анпакнуть, а именно три типа фич, которые могут пригодиться
'balance_type', 'position_id', 'instrument_type'
"""
def unpack_parameters_simple(df):
    parameters_of_interest = ['balance_type', 'position_id', 'instrument_type']
    for index, row in df.iterrows():
        str_dict = df.at[index, 'parameters']
        # get json parameters
        parameters = handle_json_dict(str_dict)
        if not check_if_nested(parameters):
            for item in parameters.items():
                if item[0] in parameters_of_interest:
                    df.at[index, item[0]] = item[1]
                else:
                    for par in parameters_of_interest:
                        df.at[index, par] = "NaN"
        else:
            for par in parameters_of_interest:
                df.at[index, par] = "NaN" 
        """
        try:
            # get json parameters
            parameters = handle_json_dict(str_dict)
            if not check_if_nested(parameters):
                for item in parameters.items():
                    if item[0] in parameters_of_interest:
                        df.at[index, item[0]] = item[1]
                    else:
                        for par in parameters_of_interest:
                            df.at[index, par] = 0
            else:
                for par in parameters_of_interest:
                    df.at[index, par] = 0 # just zeroes then
        except:
            print('ERROR')
        """
    return df

#events = unpack_parameters(events)
#events = unpack_parameters_simple(events)
    

# let's add target var here to see how it relates to potential categories
events['made_depo'] = 0
events.loc[events['user_id'].isin(dep_ids),'made_depo'] = 1
    
print("Done")


# duration handle like numeric

# Total time for events

# events per month

events_stats = pd.DataFrame()
events_stats['ids_events_count'] = events.groupby('user_id').size()


# events per platform id for users
for platform in events_platform_ids: 
    events_stats['events_per_platform_{}'.format(platform)] = events.where(events.platform_id == platform).groupby('user_id').size()
    events_stats['events_per_platform_{}'.format(platform)].fillna(0, inplace=True)
    
"""
Create features for category
"""
events_stats = create_features_based_on_categories(events, events_stats, 'category', events_category)


"""
HANDLE NUMERICAL FEATURES
"""
"""
Get statistical info per user id
This aggregated df will be called numbers

"""
# For these fe
feats_agg = ['user_id', 'duration'] # features which gonna be used to gather stats

# Some features contain almost completely zeroes with rare exceptions. I will exclude them:
# swap, margin_call, custodial, commission, charge from the lists above and further below

operations = ['mean', 'median', 'max', 'min', q10, q90] # operations to be used
# not gonna use SUM as I will sum based on balance types later
numbers = events[feats_agg].groupby(['user_id']).agg(operations)
numbers.columns = ['_'.join(str(i) for i in col) for col in numbers.columns]

"""
Calculate duration per platform
"""
for platform in events_platform_ids: 
    events_stats['duration_per_platform_{}'.format(platform)] = events.where(events.platform_id == platform).groupby('user_id')['duration'].sum()
    events_stats['duration_per_platform_{}'.format(platform)].fillna(0, inplace=True)
    

un_months = list(events.month.unique())
um_days = list(events.day.unique())

check_month_9 = (events.month.values == 9).sum() # 132895
check_month_10 = (events.month.values == 10).sum() # 244079


"""
How many events per day per platform. Do the same for positions
"""
for platform in events_platform_ids: 
    events_stats['duration_per_platform_{}'.format(platform)] = events.where(events.platform_id == platform).groupby('user_id')['duration'].sum()
    events_stats['duration_per_platform_{}'.format(platform)].fillna(0, inplace=True)
    

dump(events_stats, 'data/events_stats_df.joblib') 
    
# Get two datasets
# One for 9th Month - train
# One for 10th Month - test
# Instead of random train test split

"""
EXPLORE
"""


def handle_user_balance_type(val):
    if val == 1:
        return 'Real_money'
    else:
        return 'Whatever_else'
    
def handle_numeric_category(val):
    return 'cat_id_{}'.format(val)


def handle_column(df,col, func):
    return df[col].apply(func)




events.platform_id = handle_column(events, 'platform_id', handle_numeric_category)


list_of_cat_vars = ['platform_id', 
                    'category', 
                    'device_id'
                    ]

"""
Normalized distribution of each class per feature and plotted the 
difference between positive and negative frequencies. 
Positive values imply this category favors users that will 
make a depo and negative values implies the opposite
"""
def show_distributions(df, categorical_vars):
    for col in categorical_vars:
        plt.figure(figsize=(20,10))
        # Returns counts of unique values for each outcome for each feature
        pos_counts = df.loc[df.made_depo.values == 1, col].value_counts() 
        neg_counts = df.loc[df.made_depo.values == 0, col].value_counts()
        
        all_counts = list(set(list(pos_counts.index) + list(neg_counts.index)))
        
        # Counts of how often each outcome was recorded.
        freq_pos = (df.made_depo.values == 1).sum()
        freq_neg = (df.made_depo.values == 0).sum()
        
        pos_counts = pos_counts.to_dict()
        neg_counts = neg_counts.to_dict()
        
        all_index = list(all_counts)
        all_counts = [pos_counts.get(k, 0) / freq_pos - neg_counts.get(k, 0) / freq_neg for k in all_counts]
    
        sns.barplot(all_counts, all_index)
        plt.title(col)
        plt.tight_layout()
        plt.show()
        
#show_distributions(events, list_of_cat_vars)
# events per day

# Check correlations of events and positions based on time
# like amount of events on each platform

# maybe merge events and positions based on user_id and position_id or smth

