# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:55:44 2019

@author: aleks
"""

import pandas as pd

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
            resulting_df.rename(columns={col: col_name_suffix + col}, inplace=True)
    return resulting_df

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
        