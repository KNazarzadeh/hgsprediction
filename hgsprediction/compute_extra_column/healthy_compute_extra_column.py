#!/usr/bin/env Disorderspredwp3

"""
Compute Target, Calculate and Add new columns based on corresponding Field-IDs and conditions

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import numpy as np
import pandas as pd

from ptpython.repl import embed

###############################################################################
###############################################################################
# This class extract all required features from data:
def compute_extra_column(df, mri_status, extra_column):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(mri_status, str ), "mri_status must be a string!"    
    assert isinstance(extra_column, str ), "target must be a string!"
    if mri_status == "nonmri":
        session = "0"
    elif mri_status == "mri":
        session = "2"

    if extra_column == "handedness":
        df = calculate_handedness(df, session)
            
            
    return df
###############################################################################
###############################################################################
def calculate_handedness(df, session):
    """Calculate sum of Handgrips
    and add "hgs(L+R)" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with calculating extra column for: (HGS Left + HGS Right)
    """
    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    # Add a new column 'new_column'
    handedness = f"handedness-{session}.0"
    
    # hgs_left field-ID: 46
    # hgs_right field-ID: 47
    # ------------------------------------
    # ------- Handedness Field-ID: 1707
    # Data-Coding: 100430
    #           1	Right-handed
    #           2	Left-handed
    #           3	Use both right and left hands equally
    #           -3	Prefer not to answer
    # ------------------------------------
    # If handedness is equal to 1
    # Right hand is Dominant
    # Find handedness equal to 1:        
    if session == "0":
        # Add and new column "hgs_dominant"
        # And assign Right hand HGS value
        df.loc[df["1707-0.0"] == 1.0, handedness] = 1.0
        # If handedness is equal to 2
        # Right hand is Non-Dominant
        # Find handedness equal to 2:
        # Add and new column "hgs_dominant"
        # And assign Left hand HGS value:  
        df.loc[df["1707-0.0"] == 2.0, handedness] = 2.0
        # ------------------------------------
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer) OR
        # NaN value
        # Dominant will be the Highest Handgrip score from both hands.
        # Find handedness equal to 3, -3 or NaN:
        # Add and new column "hgs_dominant"
        # And assign Highest HGS value among Right and Left HGS:
        # Add and new column "hgs_dominant"
        # And assign lowest HGS value among Right and Left HGS:
        df.loc[df["1707-0.0"].isin([3.0, -3.0, np.nan]), handedness] = 3.0
        
    elif session == "2":
        index = df[df.loc[:, "1707-2.0"] == 1.0].index
        df.loc[index, handedness] = 1.0
        index = df[df.loc[:, "1707-2.0"] == 2.0].index
        df.loc[index, handedness] = 2.0
            
        index = df[df.loc[:, "1707-2.0"].isin([3.0, -3.0, np.NaN])].index
        filtered_df = df.loc[index, :]
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"] == 1.0].index
        df.loc[inx, handedness] = 1.0
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"] == 2.0].index
        df.loc[inx, handedness] = 2.0
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"].isin([3.0, -3.0, np.NaN])].index
        df.loc[inx, handedness] = 3.0

    return df

###############################################################################
