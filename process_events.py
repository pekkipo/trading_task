# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:50:34 2019

@author: aleks
"""

import pandas as pd
from utils import check_if_nested, handle_json_dict, create_features_based_on_categories, q10, q90
from joblib import dump


def process_events_dataset(events):

    # Replace NaNs in duration with 0s
    print(events.isna().any())
    events['duration'].fillna(0, inplace=True)
    events.fillna({'parameters':'{}'}, inplace=True)
    
    events_names = list(events.name.unique()) # 12
    events_platform_ids = list(events.platform_id.unique()) # 10
    # events_device_ids = list(events.device_id.unique()) # 606 nit gonna use it
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
    
    events = handle_dates(events) # I won't use it either, need more time
        
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
    
    #events = unpack_parameters(events)
        
    ### HANDLE CATEGORICAL FEATURES
    ## Get statistical info per user id
    ## This aggregated df will be called events_stats
    
    ## Events per user id
    events_stats = pd.DataFrame()
    events_stats['ids_events_count'] = events.groupby('user_id').size()
    
    
    ## Events per platform id for users
    for platform in events_platform_ids: 
        events_stats['events_per_platform_{}'.format(platform)] = events.where(events.platform_id == platform).groupby('user_id').size()
        events_stats['events_per_platform_{}'.format(platform)].fillna(0, inplace=True)
        

    ## Create features for category
    events_stats = create_features_based_on_categories(events, events_stats, 'category', events_category)
    
    
    ## Create features for names
    events_stats = create_features_based_on_categories(events, events_stats, 'name', events_names)
    
    dump(events_stats, 'data/events_stats_df.joblib')
    
    ### HANDLE NUMERICAL FEATURES
    ## Get statistical info per user id
    ## This aggregated df will be called numbers
    
    feats_agg = ['user_id', 'duration'] # features which gonna be used to gather stats
    
    # Some features contain almost completely zeroes with rare exceptions. I will exclude them:
    # swap, margin_call, custodial, commission, charge from the lists above and further below
    
    operations = ['mean', 'median', 'max', 'min', q10, q90] # operations to be used
    # not gonna use SUM as I will sum based on balance types later
    numbers = events[feats_agg].groupby(['user_id']).agg(operations)
    numbers.columns = ['_'.join(str(i) for i in col) for col in numbers.columns]
    

    ## Calculate duration per platform
    for platform in events_platform_ids: 
        numbers['duration_per_platform_{}'.format(platform)] = events.where(events.platform_id == platform).groupby('user_id')['duration'].sum()
        numbers['duration_per_platform_{}'.format(platform)].fillna(0, inplace=True)
        
    """
    un_months = list(events.month.unique())
    um_days = list(events.day.unique())
    
    check_month_9 = (events.month.values == 9).sum() # 132895
    check_month_10 = (events.month.values == 10).sum() # 244079
    """
             
    dump(numbers, 'data/events_numbers_df.joblib') 
    
    events_stats = pd.concat([numbers, events_stats], axis=1)
    events_stats.reset_index(inplace=True)
    events_stats.rename(columns={"index": "user_id"}, inplace=True)
    
    return events_stats
        