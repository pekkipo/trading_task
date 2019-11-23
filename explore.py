# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 21:02:37 2019

@author: aleks
"""


# %% Load data


import params
import pandas as pd
from utils import list_intersection, list_difference, calculate_intersection_and_differences, fill_instr_types_columns, handle_categorical_types
import seaborn as sns
import matplotlib.pyplot as plt



deposits = pd.read_csv(params.deposits, delimiter=params.delimiter)
positions = pd.read_csv(params.positions, delimiter=params.delimiter)
appsflyer = pd.read_csv(params.appsflyer, delimiter=params.delimiter)

events = pd.read_csv(params.events, delimiter=params.delimiter)


# %% Do smth with the data

# Получим все айди из каждого датасета
dep_ids = list(deposits.user_id.unique()) # 212
pos_ids = list(positions.user_id.unique()) # 2556
ev_ids = list(events.user_id.unique()) # 299 I ll have to go with just true or false event feature
apf_ids = list(appsflyer.user_id.unique()) # 4991 какого хрена так много

#inter_dict = calculate_intersection_and_differences(pos_ids, dep_ids, ev_ids, apf_ids)

######    

# ids who made the deposit
inter_pos_dep = list_intersection(pos_ids, dep_ids) # 202
# 202 ids made a deposit

# ids who didn't make a deposit
dif_pos_dep = list_difference(pos_ids, dep_ids) # 2354
# 2354 ids didn'make a deposit

# just checking if it adds up and it does
dif_dep_pos = list_difference(dep_ids, pos_ids) # 10
# these ids are present in deposits but are not present in positions

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


print("Done")


# So from now on Im working only with IDS that are present in positions dataset, i.e. 2556 ids


# Получим список айди которые делали депозит и которые нет


# %% 

# %% Explore positions

# position
"""
print('DESCRIBE POSTIONS DS')
pos_desc=positions.describe()
print(pos_desc)

print('DESCRIBE PLATFORM ID POSTIONS DS')
pos_desc_clpl=positions.client_platform_id.describe()
print(pos_desc_clpl)
"""

pos_pl_ids = list(positions.client_platform_id.unique()) 
#positions.client_platform_id.value_counts().plot(kind='bar')

pos_instr_types = list(positions.instrument_type.unique()) # just 5
pos_instr_under = list(positions.instrument_underlying.unique()) # 178 
pos_instr_aid = list(positions.instrument_active_id.unique()) # 178
pos_cl_pl = list(positions.client_platform_id.unique()) #12
pos_balance_types = list(positions.user_balance_type.unique()) #2
pos_position_types = list(positions.position_type.unique()) #2

#  Drop the duplicates from the dataset.
print(positions.shape)
positions = positions.drop_duplicates(keep='first')
print(positions.shape)


# let's add target var here to see how it relates to potential categories
positions['made_depo'] = 0
positions.loc[positions['user_id'].isin(dep_ids),'made_depo'] = 1

made_depos = positions.made_depo.value_counts()
#positions.made_depo.value_counts().plot(kind='bar')

# Check that the numbers add up, they do btw
check1=positions.where(positions.made_depo == 0).groupby('user_id').size() # 2354 unique ids that didn't make a depo
check2=positions.where(positions.made_depo == 1).groupby('user_id').size() # 202 that did

check3 = (positions.made_depo.values == 1).sum()
check4 = (positions.made_depo.values == 0).sum()

# cound how many deals sealed for every user id
frequency = pd.DataFrame()
frequency['ids_deals_count'] = positions.groupby('user_id').size()

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
# so ids to exclude are: 
ids_to_exclude = list(frequency['ids_deals_count'].where(frequency['ids_deals_count'] > f_quan99))

"""
Check how many balances each user id has
"""
frequency['ids_balances_count'] = positions.groupby('user_id')['user_balance_id'].nunique()
# found out that users have only 1 or 2 balances
# 90% of them have 1 balance
frequency.ids_balances_count.value_counts().plot(kind='barh')

"""
Calculate amount of deals with different client platforms per user
"""
frequency['ids_client_platform_count'] = positions.groupby('user_id')['client_platform_id'].nunique()

"""
Create features for each instrument type True/False
"""
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
Create features for balance_type type True/False
"""
frequency['ids_balance_types_count'] = positions.groupby('user_id')['user_balance_type'].nunique()
# No need to do this cauz amount of balances would be the same as each balance apparently belongs to a different type
# instead - make true false vars has_deals_type1 and has_deals_type2
list_of_balance_types_per_user = positions.groupby('user_id').user_balance_type.unique()
df_with_balance_types = handle_categorical_types(list_of_balance_types_per_user, pos_balance_types, 'has_balance_type_')
frequency = pd.concat([frequency, df_with_balance_types], axis=1)

# Calculate amount of deals involving each balance type
frequency['deals_per_balance_type_1_count'] = (positions.user_balance_type.values == 1).sum()
frequency['deals_per_balance_type_4_count'] = (positions.user_balance_type.values == 4).sum()


# Maybe should also count how many deals were made with each platform per user?


"""
Calculate amount of deals of different position types
"""


# reset index at the end! and add target column 
frequency = frequency.reset_index() # this causes problems now
# now add target var column and study how it relates to the other columns
frequency['made_depo'] = 0
frequency.loc[frequency['user_id'].isin(dep_ids),'made_depo'] = 1

# can create a feature which a ratio of deals with real money and not real money



## Change client_platform_id, user_balance_type, instrument_active_id, instrument underlying
# to categorical (only for better visualisation for now)

def handle_user_balance_type(val):
    if val == 1:
        return 'Real_money'
    else:
        return 'Whatever_else'
    
def handle_numeric_category(val):
    return 'cat_id_{}'.format(val)


def handle_column(df,col, func):
    return df[col].apply(func)



positions.user_balance_type = handle_column(positions, 'user_balance_type', handle_user_balance_type)
positions.client_platform_id = handle_column(positions, 'client_platform_id', handle_numeric_category)
positions.instrument_active_id = handle_column(positions, 'instrument_active_id', handle_numeric_category)    


list_of_cat_vars = ['instrument_type', 
                    'instrument_underlying', 
                    'instrument_active_id',
                    'instrument_dir',
                    'position_type',
                    'close_reason',
                    'user_balance_type',
                    'client_platform_id'
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
        #Returns counts of unique values for each outcome for each feature
        pos_counts = df.loc[df.made_depo.values == 1, col].value_counts() 
        neg_counts = df.loc[df.made_depo.values == 0, col].value_counts()
        
        all_counts = list(set(list(pos_counts.index) + list(neg_counts.index)))
        
        #Counts of how often each outcome was recorded.
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
        

#show_distributions(positions, list_of_cat_vars)
        
"""
from that i can infer that (positions dataset):
1) Position_type should be used as feature 
2) same goes for instrument dir
3) also user_balance_type
4) also position type
5) also close reason
    
    
"""

print("Done")

# remove columns with nans


# %% Explore deposits
"""
print('DESCRIBE DEPOSTIS DS')
dep_desc = deposits.describe()

print('DESCRIBE PLATFORM ID DEPOSITS DS')
deposits.client_platform_id.describe()
"""


dep_pl_ids = list(deposits.client_platform_id.unique()) 
deposits.client_platform_id.value_counts().plot(kind='bar')

# Turns out that the most deposits were made using cliend platform 15 while this platform is not even present
# in positions dataset

# %% Explore events
ev_pl_ids = list(events.platform_id.unique()) 
events.platform_id.value_counts().plot(kind='bar')


ev_cats = list(events.category.unique()) 
events.category.value_counts().plot(kind='bar')

ev_names = list(events.name.unique()) 
events.name.value_counts().plot(kind='bar')


# %% Explore Appsflyer

addsflyer = list(positions.client_platform_id.unique()) 
#addsflyer.client_platform_id.value_counts().plot(kind='bar')

"""
Посмотреть какие юзеры делали депозит и какие нет - check

Посчитать количество сделок на каждого юзера

Посмотреть какие типы ивентов есть, посчитать их на каждого юзера.
Либо глянуть есть ли ивент на юзера в принципе, 
либо может сделать фичи типа какие ивенты по типам, если ивент есть.
Разбить фичи по ивентам по платформам.
Посчитать сколько есть имен ивентов, посмотреть что по параметрам.


Фича - совершал сделку на демо счете, совершал на реальном счете






"""