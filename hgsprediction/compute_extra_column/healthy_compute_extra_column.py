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
def compute_extra_column(df, mri_status, extra_column, session):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(mri_status, str ), "mri_status must be a string!"    
    assert isinstance(extra_column, str ), "target must be a string!"
    # if mri_status == "nonmri":
    #     session = "0"
    # elif mri_status == "mri":
    #     session = "2"

    if extra_column == "handedness":
        df = calculate_handedness(df, session)
    
    # elif extra_column == "age_range":
    #     df = calculate_age_range(df, session)     

    elif extra_column == "hgs_cutoff":
        df = calculate_cutoff_value_hgs(df, session)     
            
    return df

###############################################################################
def calculate_handedness(df, session):
    """Calculate dominant handgrip
    and add "handedness" column to dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe that desired to analysis

    Return
    ----------
    df : dataframe
        with extra column for: Dominant hand Handgrip strength
    """
    # session = self.session
    
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
    if session in ["0", "3"]:
        # Add and new column "handedness"
        # And assign Right hand HGS value
        df.loc[df["1707-0.0"] == 1.0, handedness] = 1.0
        # If handedness is equal to 2
        # Right hand is Non-Dominant
        # Find handedness equal to 2:
        # Add and new column "handedness"
        # And assign Left hand HGS value:  
        df.loc[df["1707-0.0"] == 2.0, handedness] = 2.0
        # ------------------------------------
        # If handedness is equal to:
        # 3 (Use both right and left hands equally) OR
        # -3 (handiness is not available/Prefer not to answer) OR
        # NaN value
        # Dominant will be the Highest Handgrip score from both hands.
        # Find handedness equal to 3, -3 or NaN:
        # Add and new column "handedness"
        # And assign Highest HGS value among Right and Left HGS:
        # Add and new column "handedness"
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
# def calculate_age_range(df, session):
#     # Define your age bins/ranges
#     bins = np.arange(35, 80, 5)
#     labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]  # Adjusted labels

#     # Use pd.cut() to create the age range column
#     df["age_range"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)  # Adjusted labels and right parameter
    # retun df
###############################################################################
def calculate_cutoff_value_hgs(df, session):

    # Define your age bins/ranges
    bins = np.arange(35, 80, 5)
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]  # Adjusted labels

    # Use pd.cut() to create the age range column
    df["age_range"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)  # Adjusted labels and right parameter

    # Define age-specific percentiles and cutoff values
    percentiles = [10, 20, 25, 30, 40, 50, 75, 90, 95]

    # Initialize a dictionary to store cutoff values for each age range and each specified percentile
    cutoff_values = {"age_range": labels}
    for p in percentiles:
        cutoff_values[f"{p}th Percentile"] = []

    # Calculate cutoff values for each age range and each specified percentile
    for age_range in labels:
        for p in percentiles:
            age_group_data = df[(df["age_range"] == age_range)]["hgs_dominant-0.0"]
            if not age_group_data.empty:
                cutoff_value = np.percentile(age_group_data, p)
                cutoff_values[f"{p}th Percentile"].append(cutoff_value)
            else:
                cutoff_values[f"{p}th Percentile"].append(None)

    # Create a DataFrame from the cutoff values dictionary
    cutoff_df = pd.DataFrame(cutoff_values)

    # Merge the cutoff values back into the original DataFrame
    df = df.merge(cutoff_df, on="age_range", how="left")

    # Now, df contains the original data with age ranges and cutoff percentiles

    return df