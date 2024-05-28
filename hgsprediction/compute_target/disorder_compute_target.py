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
def compute_target(df, session_column, target):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str ), "session_column must be a string!"    
    assert isinstance(target, str ), "target must be a string!"

    if target == "hgs_L+R":
        df = calculate_sum_hgs(df, session_column)
            
    elif target == "hgs_left":
        df = calculate_left_hgs(df, session_column)

    elif target == "hgs_right":
        df = calculate_right_hgs(df, session_column)

    elif target in ["hgs_dominant", "hgs_nondominant"]:
        df = df.copy()
    
    elif target == "hgs_L-R":
        df = calculate_sub_hgs(df, session_column)
        
    elif target == "hgs_LI":
        df = calculate_laterality_index_hgs(df, session_column)
            
    return df
###############################################################################
###############################################################################
def calculate_sum_hgs(df, session_column):
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
    hgs_sum = session_column.replace(substring_to_remove, "hgs_L+R")
    
    # Add new column "hgs_L+R" by the following process: 
    # hgs_left field-ID: 46
    # hgs_right field-ID: 47
    # Addition of Handgrips (Left + Right)
    df[hgs_sum] = df.apply(lambda row: row[f"46-{row[session_column]}"]+row[f"47-{row[session_column]}"], axis=1)

    return df

###############################################################################
def calculate_left_hgs(df, session_column):
    """Calculate right and add "hgs(left)" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with calculating extra column for:
        HGS(Left)
    """

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    # -----------------------------------------------------------
    substring_to_remove = "session"
    # Add a new column 'new_column'
    hgs_left = session_column.replace(substring_to_remove, "hgs_left")
        
    df[hgs_left] = df.apply(lambda row: row[f"46-{row[session_column]}"], axis=1)

    return df

###############################################################################
def calculate_right_hgs(df, session_column):
    """Calculate right and add "hgs(right)" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with calculating extra column for:
        HGS(Right)
    """

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    # -----------------------------------------------------------
    substring_to_remove = "session"
    # Add a new column 'new_column'
    hgs_right = session_column.replace(substring_to_remove, "hgs_right")
    df[hgs_right] = df.apply(lambda row: row[f"47-{row[session_column]}"], axis=1)

    return df

###############################################################################
def calculate_sub_hgs(df, session_column):
    """Calculate subtraction of Handgrips
    and add "hgs(L-R)" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with calculating extra column for: (HGS Left - HGS Right)
    """

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    # -----------------------------------------------------------
    substring_to_remove = "session"
    # Add a new column 'new_column'
    hgs_sub = session_column.replace(substring_to_remove, "hgs_L-R")
    
    # Add new column "hgs_L-R" by the following process: 
    # hgs_left field-ID: 46
    # hgs_right field-ID: 47
    # Subtraction of Handgrips (Left - Right)
    df[hgs_sub] = df.apply(lambda row: row[f"46-{row[session_column]}"]-row[f"47-{row[session_column]}"], axis=1)

    return df

###############################################################################
def calculate_laterality_index_hgs(df, session_column):
    """Calculate Laterality Index and add "hgs(LI)" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with calculating extra column for:
        HGS(Left - Right)/HGS(Left + Right)
    """

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str), "session_column must be a string!"
    # -----------------------------------------------------------
    substring_to_remove = "session"
    # Add a new column 'new_column'
    hgs_sub = session_column.replace(substring_to_remove, "hgs_L-R")
    hgs_sum = session_column.replace(substring_to_remove, "hgs_L+R")
    hgs_LI = session_column.replace(substring_to_remove, "hgs_LI")
    
    df = calculate_sub_hgs(df, session_column)
    df = calculate_sum_hgs(df, session_column)
    
    # df.loc[:, hgs_LI] = df_sub.loc[:, hgs_sub] / df_sum.loc[:, hgs_sum]
    df[hgs_LI] = df.apply(lambda row: row[hgs_sub]/row[hgs_sum] if row[hgs_sum] != 0 else np.nan, axis=1)

    return df
###############################################################################
