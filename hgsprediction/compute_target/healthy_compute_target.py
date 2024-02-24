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
def compute_target(df, mri_status, session, target):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(mri_status, str ), "mri_status must be a string!"    
    assert isinstance(target, str ), "target must be a string!"
    # if mri_status == "nonmri":
    #     session = "0"
    # elif mri_status == "mri":
    #     session = "2"

    if target == "hgs_L+R":
        df = calculate_sum_hgs(df, session)
            
    elif target == "hgs_left":
        df = calculate_left_hgs(df, session)

    elif target == "hgs_right":
        df = calculate_right_hgs(df, session)

    elif target in ["hgs_dominant", "hgs_nondominant"]:
        df = calculate_dominant_nondominant_hgs(df, session)
    
    elif target == "hgs_L-R":
        df = calculate_sub_hgs(df, session)
        
    elif target == "hgs_LI":
        df = calculate_laterality_index_hgs(df, session)
            
    return df
###############################################################################
###############################################################################
def calculate_sum_hgs(df, session):
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
    hgs_sum = f"hgs_L+R-{session}.0"
    
    # Add new column "hgs_L+R" by the following process: 
    # hgs_left field-ID: 46
    # hgs_right field-ID: 47
    # Addition of Handgrips (Left + Right)
    df.loc[:, hgs_sum] = df.loc[:, f"46-{session}.0"]+df.loc[:, f"47-{session}.0"]

    return df

###############################################################################
def calculate_left_hgs(df, session):
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
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    # Add a new column 'new_column'
    hgs_left = f"hgs_left-{session}.0"

    df.loc[:, hgs_left] = df.loc[:, f"46-{session}.0"]

    return df

###############################################################################
def calculate_right_hgs(df, session):
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
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    
    # Add a new column 'new_column'
    hgs_right = f"hgs_right-{session}.0"
    df.loc[:, hgs_right] = df.loc[:, f"47-{session}.0"]

    return df

###############################################################################
def calculate_sub_hgs(df, session):
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
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    # Add a new column 'new_column'
    hgs_sub = f"hgs_L-R-{session}.0"
    
    # Add new column "hgs_L-R" by the following process: 
    # hgs_left field-ID: 46
    # hgs_right field-ID: 47
    # Subtraction of Handgrips (Left - Right)
    df.loc[:, hgs_sub] = df.loc[:, f"46-{session}.0"] - df.loc[:, f"47-{session}.0"]

    return df

###############################################################################
def calculate_laterality_index_hgs(df, session):
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
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    
    # Add a new column 'new_column'
    hgs_sub = f"hgs_L-R-{session}.0"
    hgs_sum = f"hgs_L+R-{session}.0"
    hgs_LI = f"hgs_LI-{session}.0"
    
    df = calculate_sub_hgs(df, session)
    df = calculate_sum_hgs(df, session)
    
    df.loc[:, hgs_LI] = df.loc[:, hgs_sub] / df.loc[:, hgs_sum]

    return df

###############################################################################
