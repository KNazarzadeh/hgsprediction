
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
def compute_extra_column(df, session_column, extra_column):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str ), "session_column must be a string!"    
    assert isinstance(extra_column, str ), "target must be a string!"

    if extra_column == "handedness":
        df = calculate_handedness(df, session_column)
        
    elif extra_column == "years":
        df = calculate_years(df, session_column)
            
    return df

###############################################################################
def calculate_years(df, session_column):
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
    assert isinstance(session_column, str), "session_column must be a string!"
    # -----------------------------------------------------------
    substring_to_remove = "session"
    # Add a new column 'new_column'
    days = session_column.replace(substring_to_remove, "days")
    years = session_column.replace(substring_to_remove, "years")
    
    df.loc[:, years] = df.loc[:, days]/365

    return df

###############################################################################
def calculate_handedness(df, session_column):
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
    assert isinstance(session_column, str), "session_column must be a string!"
    # -----------------------------------------------------------
    substring_to_remove = "session"
    # Add a new column 'new_column'
    handedness = session_column.replace(substring_to_remove, "handedness")
    # -----------------------------------------------------------
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
    if df[session_column].isin([0.0, 1.0, 3.0]).any():
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
        
    elif df[session_column].isin([2.0]).any():
        
        index = df[df.loc[:, "1707-2.0"] == 1.0].index
        df.loc[index, hgs_dominant] = df.loc[index, "47-2.0"]
        df.loc[index, hgs_nondominant] = df.loc[index, "46-2.0"]
        index = df[df.loc[:, "1707-2.0"] == 2.0].index
        df.loc[index, hgs_dominant] = df.loc[index, "46-2.0"]
        df.loc[index, hgs_nondominant] = df.loc[index, "47-2.0"]
            
        index = df[df.loc[:, "1707-2.0"].isin([3.0, -3.0, np.NaN])].index
        filtered_df = df.loc[index, :]
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"] == 1.0].index
        df.loc[inx, hgs_dominant] = df.loc[inx, "47-2.0"]
        df.loc[inx, hgs_nondominant] = df.loc[inx, "46-2.0"]
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"] == 2.0].index
        df.loc[inx, hgs_dominant] = df.loc[inx, "46-2.0"]
        df.loc[inx, hgs_nondominant] = df.loc[inx, "47-2.0"]
        inx = filtered_df[filtered_df.loc[:, "1707-0.0"].isin([3.0, -3.0, np.NaN])].index
        df.loc[inx, hgs_dominant] = df.loc[inx, ["46-2.0", "47-2.0"]].max(axis=1)
        df.loc[inx, hgs_nondominant] = df.loc[inx, ["46-2.0", "47-2.0"]].min(axis=1)
        
        
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 1.0, handedness] = 1.0
        
            
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 2.0, handedness] = 2.0
        
            
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"].isin([3.0, -3.0, np.NaN]), handedness] = 3.0
       

        df.loc[df["1707-2.0"] == 1.0, handedness] = 1.0
        df.loc[df["1707-2.0"] == 2.0, handedness] = 2.0

    return df

###############################################################################
