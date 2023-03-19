#!/usr/bin/env Disorderspredwp3
"""Perform different preprocessing on populations."""

# Authors: # Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

import pandas as pd
import numpy as np
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Original Data per sessionon the population
def originaldata_per_session(dataframe, ses):
        
    if ses == 0:
            df_tmp = dataframe[
            dataframe.columns[
                ~dataframe.columns.str.contains("-1.*|-2.*|-3.*")]]
    elif ses == 1:
        df_tmp = dataframe[
        dataframe.columns[
            ~dataframe.columns.str.contains("-0.*|-2.*|-3.*")]]
    elif ses == 2:
        df_tmp = dataframe[
        dataframe.columns[
            ~dataframe.columns.str.contains("-0.*|-1.*|-3.*")]]
    elif ses == 3:
        df_tmp = dataframe[
        dataframe.columns[
            ~dataframe.columns.str.contains("-0.*|-1.*|-2.*")]]          

    
    return df_tmp

###############################################################################
def check_hgs_availability(dataframe):
    """ Check availability of Handgrip_strength on sessions level.
        Create a list of different sessions dataframes for HGS.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame of data specified.

    Returns
    --------
    pandas.DataFrame
        DataFrame of data specified.

    dfs_List : List of dataframes
        List of dataframes based on sessions of HGS.

    """
    # Handgrip strength info
    # for Left and Right Hands
    hgs_left = "46"  # Handgrip_strength_(left)
    hgs_right = "47"  # Handgrip_strength_(right)
    # Sessions/ Instances info
    sessions = 4

    # Create an empty dataframe for output
    dataframe_output = pd.DataFrame()
    # Check non-Zero and non_NaN Handgrip strength
    # for Left and Right Hands
    for ses in range(0, sessions):
        dataframe_tmp = dataframe[
            ((~dataframe[f'{hgs_left}-{ses}.0'].isna()) &
             (dataframe[f'{hgs_left}-{ses}.0'] !=  0))
            & ((~dataframe[f'{hgs_right}-{ses}.0'].isna()) &
               (dataframe[f'{hgs_right}-{ses}.0'] !=  0))
        ]
        dataframe_output = pd.concat([dataframe_output, dataframe_tmp])

    # Drop the duplicated subjects
    # based on 'eid' column (subject ID)
    dataframe_output = dataframe_output.drop_duplicates(subset=['eid'])

    return dataframe_output


###############################################################################
def check_hgs_availability_per_session(dataframe):
    """ Check availability of Handgrip_strength on sessions level.
        Create a list of different sessions dataframes for HGS.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame of data specified.

    Returns
    --------
    pandas.DataFrame
        DataFrame of data specified.

    dfs_List : List of dataframes
        List of dataframes based on sessions of HGS.

    """
    # Handgrip strength info
    # for Left and Right Hands
    hgs_left = "46"  # Handgrip_strength_(left)
    hgs_right = "47"  # Handgrip_strength_(right)
    # Sessions/ Instances info
    sessions = 4
    # Create an empty list of dataframes for output
    # It will contain 4 dataframe for 4 sessions
    dfs_list = []

    # Check non-Zero and non_NaN Handgrip strength
    # for Left and Right Hands
    for ses in range(0, sessions):
        dataframe_tmp = dataframe[
            ((~dataframe[f'{hgs_left}-{ses}.0'].isna()) &
             (dataframe[f'{hgs_left}-{ses}.0'] !=  0))
            & ((~dataframe[f'{hgs_right}-{ses}.0'].isna()) &
               (dataframe[f'{hgs_right}-{ses}.0'] !=  0))
        ]
        if ses == 0:
            df_tmp = dataframe_tmp[
            dataframe_tmp.columns[
                ~dataframe_tmp.columns.str.contains("-1.*|-2.*|-3.*")]]
        elif ses == 1:
            df_tmp = dataframe_tmp[
            dataframe_tmp.columns[
                ~dataframe_tmp.columns.str.contains("-0.*|-2.*|-3.*")]]
        elif ses == 2:
            df_tmp = dataframe_tmp[
            dataframe_tmp.columns[
                ~dataframe_tmp.columns.str.contains("-0.*|-1.*|-3.*")]]
        elif ses == 3:
            df_tmp = dataframe_tmp[
            dataframe_tmp.columns[
                ~dataframe_tmp.columns.str.contains("-0.*|-1.*|-2.*")]]          
        
        dfs_list.append(df_tmp)

    return dfs_list

###############################################################################
def check_hgs_availability_per_session(dataframe, session):
    """ Check availability of Handgrip_strength on sessions level.
        Create a list of different sessions dataframes for HGS.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame of data specified.

    Returns
    --------
    pandas.DataFrame
        DataFrame of data specified.

    dfs_List : List of dataframes
        List of dataframes based on sessions of HGS.

    """
    # Handgrip strength info
    # for Left and Right Hands
    hgs_left = "46"  # Handgrip_strength_(left)
    hgs_right = "47"  # Handgrip_strength_(right)
    # Create an empty list of dataframes for output
    # It will contain 4 dataframe for 4 sessions

    # Check non-Zero and non_NaN Handgrip strength
    # for Left and Right Hands
    dataframe_tmp = dataframe[
        ((~dataframe[f'{hgs_left}-{session}.0'].isna()) &
            (dataframe[f'{hgs_left}-{session}.0'] !=  0)) 
        & ((~dataframe[f'{hgs_right}-{session}.0'].isna()) &
            (dataframe[f'{hgs_right}-{session}.0'] !=  0))].reset_index()

    return dataframe_tmp
