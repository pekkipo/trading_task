# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:43:11 2019

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

# leave only ids that are present in pos_ids

# Replace NaNs in duration with 0s
print(appsflyer.isna().any())
appsflyer.fillna({'country_id':-1, 'media_source':'Unknown'}, inplace=True)



af_platforms = list(appsflyer.client_platform_id.unique()) # 5
af_countries = list(appsflyer.country_id.unique()) # 172
af_devices = list(appsflyer.device.unique()) # 2982
af_os_version = list(appsflyer.os_version.unique()) # nah 97
af_af_devices = list(appsflyer.apps_flyer_device_id.unique()) # 29726 too much
af_aff_ids = list(appsflyer.aff_id.unique()) # 412
af_msources = list(appsflyer.media_source.unique()) # 313 # will remove this one as it contains almost the same info as aff_ids


most_common_affs = list(appsflyer.aff_id.value_counts().reset_index(name="count").query("count > 100")["index"])
# 99999 means basically 'OTHER' category
replacement = lambda aff_id: aff_id if aff_id in most_common_affs else 99999
appsflyer['aff_id'] = appsflyer['aff_id'].map(replacement)
af_aff_ids = list(appsflyer.aff_id.unique()) # new list of unique values

most_common_devices = list(appsflyer.device.value_counts().reset_index(name="count").query("count > 100")["index"])
replacement = lambda device: device if device in most_common_devices else 'Other'
appsflyer['device'] = appsflyer['device'].map(replacement)
af_devices = list(appsflyer.device.unique())

contr = appsflyer.country_id.value_counts()

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


# let's add target var here to see how it relates to potential categories
# Which is not exactly correct idea but for the sake of simplicty I ll leave it be
appsflyer['made_depo'] = 0
appsflyer.loc[appsflyer['user_id'].isin(dep_ids),'made_depo'] = 1



#appsflyer.platform_id = handle_column(appsflyer, 'client_platform_id', handle_numeric_category)
appsflyer.country_id = handle_column(appsflyer, 'country_id', handle_numeric_category)
appsflyer.aff_id = handle_column(appsflyer, 'aff_id', handle_numeric_category)


list_of_cat_vars = ['client_platform_id', 
                    'country_id', 
                    'device', 
                     'aff_id',
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
        
        
show_distributions(appsflyer, list_of_cat_vars)

print("Done")

