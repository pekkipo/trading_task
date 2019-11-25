# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:05:25 2019

@author: aleks
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:50:34 2019

@author: aleks
"""

import pandas as pd
from utils import create_features_based_on_categories, create_features_for_appflyer
from joblib import dump


def process_appsflyer_dataset(appsflyer):

    
    # Replace NaNs in duration with 0s
    print(appsflyer.isna().any())
    appsflyer.fillna({'country_id':-1, 'media_source':'Unknown'}, inplace=True)
    
      
    af_platforms = list(appsflyer.client_platform_id.unique()) # 5
    af_countries = list(appsflyer.country_id.unique()) # 172
    af_devices = list(appsflyer.device.unique()) # 2982
    #af_os_version = list(appsflyer.os_version.unique()) # 97 exclude
    #af_af_devices = list(appsflyer.apps_flyer_device_id.unique()) # 29726 exclude
    af_aff_ids = list(appsflyer.aff_id.unique()) # 412
    #af_msources = list(appsflyer.media_source.unique()) # 313 # will remove this one as it contains almost the same info as aff_ids
    
    
    def replace_values_with_common(df, feature, repl_value):
        """
        Replaces rare value with common type like "Other" or similar
        In order to reduce the number of meaningful features
        """
        most_common_types = list(appsflyer[feature].value_counts().reset_index(name="count").query("count > 100")["index"])
        replacement = lambda aff_id: aff_id if aff_id in most_common_types else repl_value
        df[feature] = df[feature].map(replacement)
        return df
    
    appsflyer = replace_values_with_common(appsflyer, 'aff_id', 99999)
    af_aff_ids = list(appsflyer.aff_id.unique())
    appsflyer = replace_values_with_common(appsflyer, 'device', 'Other')
    af_devices = list(appsflyer.device.unique())
    appsflyer = replace_values_with_common(appsflyer, 'country_id', 999)
    af_countries = list(appsflyer.country_id.unique())
    
    """
    most_common_affs = list(appsflyer.aff_id.value_counts().reset_index(name="count").query("count > 100")["index"])
    # 99999 means basically 'OTHER' category
    replacement = lambda aff_id: aff_id if aff_id in most_common_affs else 99999
    appsflyer['aff_id'] = appsflyer['aff_id'].map(replacement)
    af_aff_ids = list(appsflyer.aff_id.unique()) # new list of unique values
    
    most_common_devices = list(appsflyer.device.value_counts().reset_index(name="count").query("count > 100")["index"])
    replacement = lambda device: device if device in most_common_devices else 'Other'
    appsflyer['device'] = appsflyer['device'].map(replacement)
    af_devices = list(appsflyer.device.unique())
    """
    
    ## Overall records per user, i.e. clicks I assume
    appsflyer_stats = pd.DataFrame()
    appsflyer_stats['ids_records_count'] = appsflyer.groupby('user_id').size()
    
    ## Count clicks per platform types
    for platform in af_platforms:
        appsflyer_stats['amount_of_clicks_per_platform_{}'.format(platform)] = appsflyer.where(appsflyer.client_platform_id == platform).groupby('user_id').size()
        appsflyer_stats['amount_of_clicks_per_platform_{}'.format(platform)].fillna(0, inplace=True)
        
    ## Count clicks per aff_id types
    for aff in af_aff_ids:
        appsflyer_stats['amount_of_clicks_per_aff_{}'.format(aff)] = appsflyer.where(appsflyer.aff_id == aff).groupby('user_id').size()
        appsflyer_stats['amount_of_clicks_per_aff_{}'.format(aff)].fillna(0, inplace=True)

    
    ## Create features for platforms
    appsflyer_stats = create_features_for_appflyer(appsflyer, appsflyer_stats, 'client_platform_id', af_platforms, 'uses_')
    
    ## Create features for aff_ids
    appsflyer_stats = create_features_for_appflyer(appsflyer, appsflyer_stats, 'aff_id', af_aff_ids, 'uses_')
    
    ## Create features for countries
    appsflyer_stats = create_features_for_appflyer(appsflyer, appsflyer_stats, 'country_id', af_countries, 'is_')
    
    ## Create features for devices
    appsflyer_stats = create_features_for_appflyer(appsflyer, appsflyer_stats, 'device', af_devices, 'uses_')
    
    
    
    dump(appsflyer_stats, 'data/appsflyer_stats_df.joblib') 

    return appsflyer_stats
        