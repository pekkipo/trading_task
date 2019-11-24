# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 12:20:50 2019

@author: aleks
"""

import params
import pandas as pd
from utils import list_intersection, list_difference, calculate_intersection_and_differences, fill_instr_types_columns, handle_categorical_types, create_features_based_on_categories, q10, q25, q75, q90, check_currency_conversions
from joblib import dump

"""
Read datasets
"""
deposits = pd.read_csv(params.deposits, delimiter=params.delimiter)
positions = pd.read_csv(params.positions, delimiter=params.delimiter)
dep_ids = list(deposits.user_id.unique()) # 212
pos_ids = list(positions.user_id.unique()) # 2556

"""
Get lists of unique values in the columns Im interested in
"""
pos_pl_ids = list(positions.client_platform_id.unique()) 
pos_instr_types = list(positions.instrument_type.unique()) # just 5
pos_instr_under = list(positions.instrument_underlying.unique()) # 178 
pos_instr_aid = list(positions.instrument_active_id.unique()) # 178
pos_cl_pl = list(positions.client_platform_id.unique()) #12
pos_balance_types = list(positions.user_balance_type.unique()) #2
pos_position_types = list(positions.position_type.unique()) #2
pos_instrument_dir = list(positions.instrument_dir.unique()) #2
pos_close_reason = list(positions.close_reason.unique()) #5
pos_leverage = list(positions.leverage.unique()) #16


"""
Process positions dataset a bit
"""
#  Drop the duplicates from the dataset.
print(positions.shape)
positions = positions.drop_duplicates(keep='first')
print(positions.shape)

# See which columns have NaNs
print(positions.isna().any())

# Remove columns filled with only NaNs
positions.drop(['take_profit_at', 'stop_lose_at', 'take_profit_date', 'stop_lose_date', 'spread_total_enrolled', 'instrument_expiration'], axis=1, inplace=True)
# Instrument strike - replace with 0s. Because
positions['instrument_strike'].fillna(0, inplace=True)
# Volatility - replace with mean
positions['volatility'].fillna(positions['volatility'].mean(), inplace=True)
# Instr dir NaNs - there are a few, I handle them later. Kind of as a separate type

# let's add target var here to see how it relates to potential categories
# Which is not exactly correct idea but for the sake of simplicty I ll leave it be
positions['made_depo'] = 0
positions.loc[positions['user_id'].isin(dep_ids),'made_depo'] = 1

made_depos = positions.made_depo.value_counts()
positions.made_depo.value_counts().plot(kind='bar') # plot distribution

# Check that the numbers add up (They do btw)
check1=positions.where(positions.made_depo == 0).groupby('user_id').size() # 2354 unique ids that didn't make a depo
check2=positions.where(positions.made_depo == 1).groupby('user_id').size() # 202 that did

# Count how many deals sealed for every user id
frequency = pd.DataFrame()
frequency['ids_deals_count'] = positions.groupby('user_id').size()
# this column will be REMOVED later, utilized to check if the numbers add up

# See what's whith records number per users
f_mean=frequency['ids_deals_count'].mean() # 55
f_med=frequency['ids_deals_count'].median() # 8
 # 10th percentile
f_quan10=frequency['ids_deals_count'].quantile(0.1) # 1..well ofc
# 90th percentile
f_quan90=frequency['ids_deals_count'].quantile(0.9) # 76
# 99th percentile
f_quan99=frequency['ids_deals_count'].quantile(0.99) # 991
# Min - 1
# Max 
f_max=frequency['ids_deals_count'].max() # 6185

# I might want to exclude this ID from my dataset cauz it's a bit too extreme
# and everything that is larger than 99 percentile value
# though I can study them separately
# so 26 ids will be excluded
frequency_without_outliers = frequency.loc[frequency['ids_deals_count'] <= f_quan99]

"""
HANDLE NUMERICAL FEATURES
"""
"""
Get statistical info per user id
This aggregated df will be called numbers

"""
# For these fe
feats_agg = ['user_id', 'count', 'buy_amount',
             'sell_amount', 'pnl_realized', 'close_effect_amount',
             'pnl_total', 'close_underlying_price', 'open_underlying_price', 'instrument_strike'] # features which gonna be used to gather stats

# Some features contain almost completely zeroes with rare exceptions. I will exclude them:
# swap, margin_call, custodial, commission, charge from the lists above and further below

operations = ['mean', 'median', 'max', 'min', q10, q25, q75, q90] # operations to be used
# not gonna use SUM as I will sum based on balance types later
numbers = positions[feats_agg].groupby(['user_id']).agg(operations)
numbers.columns = ['_'.join(str(i) for i in col) for col in numbers.columns]

"""
Count conversions that happend for features that involve "enrolled"
I think kinda of redundant but why not
"""
feats_with_conversion = ['buy_amount', 'sell_amount', 
                         'pnl_realized', 'pnl_total']
positions = check_currency_conversions(positions, feats_with_conversion)
# then calculate the amount of conversion
for feature in feats_with_conversion:
    numbers['amount_of_conversions_for_feature_{}'.format(feature)] = positions.where(positions['{}_conversion_happened'.format(feature)] == 1).groupby('user_id').size()
    numbers['amount_of_conversions_for_feature_{}'.format(feature)].fillna(0, inplace=True)

"""
Now sum of money involved in positions based on different balance type
"""
feats_sum = ['user_id', 'user_balance_type']
feats_money = ['buy_amount',
             'sell_amount', 'pnl_realized', 'close_effect_amount',
             'pnl_total', 'close_underlying_price', 'open_underlying_price', 'instrument_strike'] 
for feat in feats_money:
    for balance_type in pos_balance_types: 
        numbers['{}_sum_balance_type_{}'.format(feat, balance_type)] = positions.where(positions[feats_sum].user_balance_type == balance_type).groupby('user_id')[feat].sum() 
        numbers['{}_sum_balance_type_{}'.format(feat, balance_type)].fillna(0, inplace=True)
    
    
# At this point I have numbers df which is aggregate information on numerical features grouped by user_id
dump(numbers, 'data/numbers_df.joblib') 

"""
HANDLE CATEGORICAL FEATURES
"""
"""
Get statistical info for categorical features per user id
This aggregated df was already defined above and will be called "frequency"
"""

"""
Calculate amount of deals with different client platforms per user
"""
frequency['ids_client_platform_count'] = positions.groupby('user_id')['client_platform_id'].nunique()

"""
Create features for each instrument type True/False
"""
# this _uses_ feature is also redundant. Or actually not, I ll keep it
list_of_instr_types_per_user = positions.groupby('user_id').instrument_type.unique()
df_with_instr_types = handle_categorical_types(list_of_instr_types_per_user, pos_instr_types, 'uses_')
frequency = pd.concat([frequency, df_with_instr_types], axis=1)

# Calculate amount of deals involving each instrument type
for instr in pos_instr_types: 
    frequency['deals_per_{}_instr_count'.format(instr)] = positions.where(positions.instrument_type == instr).groupby('user_id').size()
    #(positions.groupby('user_id')['instrument_type']. == instr).sum()
    # if the user haven't used certain instrument type the value would be NaN, therefore we replace it with 0
    frequency['deals_per_{}_instr_count'.format(instr)].fillna(0, inplace=True)

"""
Create features for balance_type type
"""
#frequency['ids_balance_types_count'] = positions.groupby('user_id')['user_balance_type'].nunique() REDUNDANT
# No need to do this cauz amount of balances would be the same as each balance apparently belongs to a different type
# instead - make true false vars has_deals_type1 and has_deals_type2
#list_of_balance_types_per_user = positions.groupby('user_id').user_balance_type.unique()
#df_with_balance_types = handle_categorical_types(list_of_balance_types_per_user, pos_balance_types, 'has_balance_type_')
#frequency = pd.concat([frequency, df_with_balance_types], axis=1)

# Calculate amount of deals involving each balance type
for balance_type in pos_balance_types: 
    frequency['deals_per_balance_type_{}_count'.format(balance_type)] = positions.where(positions.user_balance_type == balance_type).groupby('user_id').size()
    # if the user haven't used certain instrument type the value would be NaN, therefore we replace it with 0
    frequency['deals_per_balance_type_{}_count'.format(balance_type)].fillna(0, inplace=True)


"""
Calculate amount of deals of different position types
"""
#frequency['ids_position_types_count'] = positions.groupby('user_id')['position_type'].nunique()  REDUNDANT
#list_of_position_types_per_user = positions.groupby('user_id').position_type.unique()
#df_with_position_types = handle_categorical_types(list_of_position_types_per_user, pos_position_types, 'has_position_type_')
#frequency = pd.concat([frequency, df_with_position_types], axis=1)

# Calculate amount of deals involving each balance type
for position_type in pos_position_types: 
    frequency['deals_per_position_type_{}_count'.format(position_type)] = positions.where(positions.position_type == position_type).groupby('user_id').size()
    frequency['deals_per_position_type_{}_count'.format(position_type)].fillna(0, inplace=True)


### "HAS SMTH" FEATURE TYPE IS REDUNDANT AS IT IS ALREADY CONTAINED IN LATER COUNTS
## I used them to check if the numbers add up

"""
Calculate amount of deals of different instrument dir types
"""
#frequency['ids_different_types_count'] = positions.groupby('user_id')['position_type'].nunique() REDUNDANT
#list_of_instrument_dir_types_per_user = positions.groupby('user_id').instrument_dir.unique()
#df_with_instrument_dir_types = handle_categorical_types(list_of_instrument_dir_types_per_user, pos_instrument_dir, 'has_instr_dir_type_')
#frequency = pd.concat([frequency, df_with_instrument_dir_types], axis=1)

# Calculate amount of deals involving each balance type
for instr_dir_type in pos_instrument_dir: 
    frequency['deals_per_instr_dir_type_{}_count'.format(instr_dir_type)] = positions.where(positions.instrument_dir == instr_dir_type).groupby('user_id').size()
    frequency['deals_per_instr_dir_type_{}_count'.format(instr_dir_type)].fillna(0, inplace=True)


"""
Create features for each client platform
"""
# redundant
"""
list_of_client_platforms_per_user = positions.groupby('user_id').client_platform_id.unique()
df_with_client_platform_types = handle_categorical_types(list_of_client_platforms_per_user, pos_cl_pl, 'uses_')
frequency = pd.concat([frequency, df_with_client_platform_types], axis=1)
"""
# Calculate amount of deals involving each client platform
for cl_pl_id in pos_cl_pl: 
    frequency['deals_per_client_platform_{}_count'.format(cl_pl_id)] = positions.where(positions.client_platform_id == cl_pl_id).groupby('user_id').size()
    #(positions.groupby('user_id')['instrument_type']. == instr).sum()
    # if the user haven't used certain instrument type the value would be NaN, therefore we replace it with 0
    frequency['deals_per_client_platform_{}_count'.format(cl_pl_id)].fillna(0, inplace=True)

"""
Create features for close_reason
"""
frequency = create_features_based_on_categories(positions, frequency, 'close_reason', pos_close_reason, 'is_')

"""
Create features for instrument_underlying
"""
# There are too many underlying instruments involved and using them all makes not much sense
# I ll keep a few main ones and turn all others into just 'Other' category
# The main ones I ll determine based on the distribution graph made in the beginning
underlying_types = ['EURUSD', 'GBPJPY', 'AUDCAD-OTC', 'EURJPY', 'EURGBP-OTC', 'AUDCAD', 'USDJPY']
replacement = lambda und_type: und_type if und_type in underlying_types else 'OTHER'
positions['instrument_underlying']= positions['instrument_underlying'].map(replacement)
pos_instr_under = list(positions.instrument_underlying.unique())
# Transform the column


frequency = create_features_based_on_categories(positions, frequency, 'instrument_underlying', pos_instr_under, 'uses_')

# instrument active id column shows exactly the same as instrument_underlying so I won't use it

"""
Create features for leverage (It seems to be categories, so I ll treat that feature accordingly)
"""
frequency = create_features_based_on_categories(positions, frequency, 'leverage', pos_leverage, 'uses_')

# At this point I have frequency df which is aggregate information on categorical features grouped by user_id
dump(frequency, 'data/frequency_df.joblib') 


## DO SMTH WITH DATES!


print("Done")

