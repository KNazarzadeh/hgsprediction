#!/home/knazarzadeh/miniconda3/envs/disorderspredwp3/bin/python3

"""
Checks the availability of Handgrip Strength (HGS) data
on a session level.
"""

# Authors: # Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

import pandas as pd
import numpy as np
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
def check_hgs_availability(df):
    """ 
    Checks the availability of handgrip strength data on session level
    in a given DataFrame. 
    It then creates different DataFrames for:
    1. All session with HGS availability
    2. Each session with HGS availability

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of data specified.

    Returns
    --------
    df_hgs, df_session0, df_session1, df_session2, df_session3: pandas.DataFrame
        Dataframes will contain all subjects that
        had non-zero and non-NaN handgrip strength data
        for both left and right hands in:
            1.  df_hgs: All session with HGS availability
            2.  df_session0,
                df_session1,
                df_session2,
                df_session3: Each session with HGS availability

    """
    # Handgrip strength info
    # for Left and Right Hands
    hgs_left = "46"  # Handgrip_strength_(left)
    hgs_right = "47"  # Handgrip_strength_(right)
    # Sessions/ Instances info
    sessions = 4

    # Create an empty dataframe for output
    df_hgs = pd.DataFrame()

    # Check non-Zero and non_NaN Handgrip strength
    # for Left and Right Hands
    for ses in range(0, sessions):
        df_tmp = df[
        ((~df[f'{hgs_left}-{ses}.0'].isna()) &
            (df[f'{hgs_left}-{ses}.0'] !=  0))
        & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
            (df[f'{hgs_right}-{ses}.0'] !=  0))
        ]
        # The resulting DataFrame, will contain all subjects that
        # had non-zero and non-NaN handgrip strength data
        # for both left and right hands in any of the sessions.
        df_hgs = pd.concat([df_hgs, df_tmp])

    # Drop the duplicated subjects
    # based on 'eid' column (subject ID)
    df_hgs = df_hgs.drop_duplicates(subset=['eid'])
    
    # Keep only data/subjects for session 0
    # Session/Instance 0: Initial assessment visit (2006-2010)
    # at which participants were recruited and consent given 
    df_session0 = df_hgs[
    df_hgs.columns[
        ~df_hgs.columns.str.contains("-1.*|-2.*|-3.*")]]

    # Keep only data/subjects for session 1
    # Session/Instance 1: First repeat assessment visit (2012-13) 
    df_session1 = df_hgs[
    df_hgs.columns[
        ~df_hgs.columns.str.contains("-0.*|-2.*|-3.*")]]
    
    # Keep only data/subjects for session 2
    # Session/Instance 2: Imaging visit (2014+) 
    df_session2 = df_hgs[
    df_hgs.columns[
        ~df_hgs.columns.str.contains("-0.*|-1.*|-3.*")]]
    
    # Keep only data/subjects for session 3
    # Session/Instance 3: First repeat imaging visit (2019+)
    df_session3 = df_hgs[
    df_hgs.columns[
        ~df_hgs.columns.str.contains("-0.*|-1.*|-2.*")]]          
        
    return df_hgs, df_session0, df_session1, df_session2, df_session3

