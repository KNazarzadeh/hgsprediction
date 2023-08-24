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
def compute_target(df, mri_status, target):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(mri_status, str ), "mri_status must be a string!"    
    assert isinstance(target, str ), "target must be a string!"
    if mri_status == "nonmri":
        session = "0"
    elif mri_status == "mri":
        session = "2"

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
    df[hgs_sum] = df.apply(lambda row: row[f"46-{row[session]}"]+row[f"47-{row[session]}"], axis=1)

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
        
    df[hgs_left] = df.apply(lambda row: row[f"46-{row[session]}"], axis=1)

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
    df[hgs_right] = df.apply(lambda row: row[f"47-{row[session]}"], axis=1)

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
    df[hgs_sub] = df.apply(lambda row: row[f"46-{row[session]}"]-row[f"47-{row[session]}"], axis=1)

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
    
    df = calculate_sub_hgs(df)
    df = calculate_sum_hgs(df)
    
    # df.loc[:, hgs_LI] = df_sub.loc[:, hgs_sub] / df_sum.loc[:, hgs_sum]
    df[hgs_LI] = df.apply(lambda row: row[hgs_sub]/row[hgs_sum], axis=1)

    return df

###############################################################################
def calculate_dominant_nondominant_hgs(df, session):
    """Calculate dominant handgrip
    and add "hgs_dominant" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with extra column for: Dominant hand Handgrip strength
    """

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session, str), "session must be a string!"
    # -----------------------------------------------------------
    
    # Add a new column 'new_column'
    hgs_dominant = f"hgs_dominant-{session}.0"
    hgs_nondominant = f"hgs_nondominant-{session}.0"
    
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
    if session == "0.0":
        # Add and new column "hgs_dominant"
        # And assign Right hand HGS value
        df.loc[df["1707-0.0"] == 1.0, hgs_dominant] = df.loc[df["1707-0.0"] == 1.0, "47-0.0"]
        df.loc[df["1707-0.0"] == 1.0, hgs_nondominant] = df.loc[df["1707-0.0"] == 1.0, "46-0.0"]
        # If handedness is equal to 2
        # Right hand is Non-Dominant
        # Find handedness equal to 2:
        # Add and new column "hgs_dominant"
        # And assign Left hand HGS value:  
        df.loc[df["1707-0.0"] == 2.0, hgs_dominant] = df.loc[df["1707-0.0"] == 2.0, "46-0.0"]
        df.loc[df["1707-0.0"] == 2.0, hgs_nondominant] = df.loc[df["1707-0.0"] == 2.0, "47-0.0"]
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
        df.loc[df["1707-0.0"].isin([3.0, -3.0, np.nan]), hgs_dominant] = df[["46-0.0", "47-0.0"]].max(axis=1)
        df.loc[df["1707-0.0"].isin([3.0, -3.0, np.nan]), hgs_nondominant] = df[["46-0.0", "47-0.0"]].min(axis=1)
    elif session == "2.0":
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 1.0, hgs_dominant] = \
            df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 1.0, "47-0.0"]
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 1.0, hgs_nondominant] = \
            df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 1.0, "46-0.0"]
            
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 2.0, hgs_dominant] = \
            df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 2.0, "46-0.0"]
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 2.0, hgs_nondominant] = \
            df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"] == 2.0, "47-0.0"]
            
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"].isin([3.0, -3.0, np.NaN]), hgs_dominant] = \
            df[["46-0.0", "47-0.0"]].max(axis=1)
        df.loc[df["1707-2.0"].isin([3.0, -3.0, np.NaN]) & df["1707-0.0"].isin([3.0, -3.0, np.NaN]), hgs_nondominant] = \
            df[["46-0.0", "47-0.0"]].min(axis=1)

        df.loc[df["1707-2.0"] == 1.0, hgs_dominant] = df.loc[df["1707-2.0"] == 1.0, "47-2.0"]
        df.loc[df["1707-2.0"] == 1.0, hgs_nondominant] = df.loc[df["1707-2.0"] == 1.0, "46-2.0"]
        df.loc[df["1707-2.0"] == 2.0, hgs_dominant] = df.loc[df["1707-2.0"] == 2.0, "46-2.0"]
        df.loc[df["1707-2.0"] == 2.0, hgs_nondominant] = df.loc[df["1707-2.0"] == 2.0, "47-2.0"]
        
        
    return df

###############################################################################