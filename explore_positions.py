# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 12:20:50 2019

@author: aleks
"""

import params
import pandas as pd
from utils import list_intersection, list_difference, calculate_intersection_and_differences, fill_instr_types_columns, handle_categorical_types, create_features_based_on_categories, q10, q90, check_currency_conversions
import numpy as np


deposits = pd.read_csv(params.deposits, delimiter=params.delimiter)
positions = pd.read_csv(params.positions, delimiter=params.delimiter)
dep_ids = list(deposits.user_id.unique()) # 212
pos_ids = list(positions.user_id.unique()) # 2556

pos_pl_ids = list(positions.client_platform_id.unique()) 
pos_instr_types = list(positions.instrument_type.unique()) # just 5
pos_instr_under = list(positions.instrument_underlying.unique()) # 178 
pos_instr_aid = list(positions.instrument_active_id.unique()) # 178
pos_cl_pl = list(positions.client_platform_id.unique()) #12
pos_balance_types = list(positions.user_balance_type.unique()) #2
pos_position_types = list(positions.position_type.unique()) #2
pos_instrument_dir = list(positions.instrument_dir.unique()) #2
pos_close_reason = list(positions.close_reason.unique()) #5

#  Drop the duplicates from the dataset.
print(positions.shape)
positions = positions.drop_duplicates(keep='first')
print(positions.shape)


# See which columns have NaNs. Will go the easy way and just replace them with zeroes for this task
print(positions.isna().any())

# Remove columns filled with only NaNs
positions.drop(['take_profit_at', 'stop_lose_at', 'take_profit_date', 'stop_lose_date', 'spread_total_enrolled', 'instrument_expiration'], inplace=True)

# Instrument strike - replace with 0s. Because
positions['instrument_strike'].fillna(0, inplace=True)
# Volatility - replace with mean
positions['volatility'].fillna(positions['volatility'].mean(), inplace=True)




# let's add target var here to see how it relates to potential categories
positions['made_depo'] = 0
positions.loc[positions['user_id'].isin(dep_ids),'made_depo'] = 1

made_depos = positions.made_depo.value_counts()
#positions.made_depo.value_counts().plot(kind='bar')

# Check that the numbers add up, they do btw
check1=positions.where(positions.made_depo == 0).groupby('user_id').size() # 2354 unique ids that didn't make a depo
check2=positions.where(positions.made_depo == 1).groupby('user_id').size() # 202 that did

# cound how many deals sealed for every user id
frequency = pd.DataFrame()
frequency['ids_deals_count'] = positions.groupby('user_id').size()
"""
THIS COLUMN WILL HAVE TO BE REMOVED! USED ONLY FOR CHECKING THE NUMBERS
"""

f_mean=frequency['ids_deals_count'].mean() # 55
f_med=frequency['ids_deals_count'].median() # 8
 # 10th percentile
f_quan10=frequency['ids_deals_count'].quantile(0.1) # 1.0
# 90th percentile
f_quan90=frequency['ids_deals_count'].quantile(0.9) # 76.5
# 99th percentile
f_quan99=frequency['ids_deals_count'].quantile(0.99) # 991.7
# 1th percentile - obviously 1.0
# Min - also 1
# Max and min
f_max=frequency['ids_deals_count'].max() # 6185

# I might want to exclude this ID from my dataset cauz it's a bit too extreme
# and everything that is larger than 99 percentile value
# though I can study them separately
# so 26 ids will be excluded
frequency_without_outliers = frequency.loc[frequency['ids_deals_count'] <= f_quan99]

"""
Handle features that involve potentially useful numbers, get statistical info per user
"""
feats_agg = ['user_id', 'count', 'buy_amount',
             'sell_amount', 'pnl_realized', 'close_effect_amount',
             'pnl_total', 'close_underlying_price', 'swap', 'open_underlying_price', 
             'margin_call', 'custodial'] # features which gonna be used to gather stats

operations = ['mean', 'median', 'max', 'min', q10, q90] # operations to use
# not gonna use SUM as I will sum based on balance types later
money = positions[feats_agg].groupby(['user_id']).agg(operations)
money.columns = ['_'.join(str(i) for i in col) for col in money.columns]

"""
Count conversions that happend for features that involve "enrolled"
"""
feats_with_conversion = ['buy_amount', 'sell_amount', 
                         'pnl_realized', 'pnl_total', 'swap', 'margin_call', 'custodial']
positions = check_currency_conversions(positions, feats_with_conversion)
# then calculate the amount of conversion
for feature in feats_with_conversion:
    money['amount_of_conversions_for_feature_{}'.format(feature)] = positions.where(positions['{}_conversion_happened'.format(feature)] == 1).groupby('user_id').size()
    money['amount_of_conversions_for_feature_{}'.format(feature)].fillna(0, inplace=True)

"""
Now sum of money involved in positions based on different balance type
"""
feats_sum = ['user_id', 'user_balance_type']
feats_money = ['buy_amount',
             'sell_amount', 'pnl_realized', 'close_effect_amount',
             'pnl_total', 'close_underlying_price', 'swap', 'open_underlying_price', 
             'margin_call', 'custodial'] 
for feat in feats_money:
    for balance_type in pos_balance_types: 
        #money['deals_per_balance_type_{}_count'.format(balance_type)] = positions[feats_sum].where(positions[feats_sum].user_balance_type == balance_type).groupby('user_id').sum() #.agg(['sum'])
        #money['test'] = positions[feats_sum].groupby('user_id')['buy_amount'].sum() 
        money['{}_sum_balance_type_{}'.format(feat, balance_type)] = positions.where(positions[feats_sum].user_balance_type == balance_type).groupby('user_id')[feat].sum() 
        money['{}_sum_balance_type_{}'.format(feat, balance_type)].fillna(0, inplace=True)
    


print("Done")

