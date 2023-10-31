
#!/usr/bin/env Disorderspredwp3

"""
Compute Target, Calculate and Add new columns based on corresponding Field-IDs and conditions

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

"""

import numpy as np
import pandas as pd

from ptpython.repl import embed

###############################################################################
# This class extract all required features from data:
def compute_extra_column(df, session_column, extra_column):

    assert isinstance(df, pd.DataFrame), "df must be a dataframe!"
    assert isinstance(session_column, str ), "session_column must be a string!"    
    assert isinstance(extra_column, str ), "target must be a string!"

    if extra_column == "hgs_cutoff":
        df = calculate_cutoff_value_hgs(df, session_column)     
            
###############################################################################
def calculate_age_range(df, session):
    # Define your age bins/ranges
    bins = np.arange(35, 80, 5)
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]  # Adjusted labels

    # Use pd.cut() to create the age range column
    df["age_range"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)  # Adjusted labels and right parameter
    
    return df
###############################################################################
def calculate_cutoff_value_hgs(df, session):
    # Define age bins and labels
    bins = np.arange(35, 80, 5)
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]

    df["age_range"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)  # Adjusted labels and right parameter

    # Create a dictionary to store cutoff DataFrames for each gender
    cutoff_dfs = {}

    # Define age-specific percentiles
    percentiles = [10, 20, 25, 30, 40, 50, 75, 90, 95]

    # Calculate cutoff values for each gender
    for gender in df["gender"].unique():
        # Initialize a dictionary to store cutoff values for this gender and age range
        cutoff_values = {"age_range": labels}
        for p in percentiles:
            cutoff_values[f"{p}th_percentile"] = []

        # Filter the DataFrame by gender
        df_gender = df[df["gender"] == gender]

        # Calculate cutoff values for each age range and each specified percentile for this gender
        for age_range in labels:
            for p in percentiles:
                age_group_data = df_gender[df_gender["age_range"] == age_range][f"hgs_dominant-{session}.0"]
                if not age_group_data.empty:
                    cutoff_value = np.percentile(age_group_data, p)
                    cutoff_values[f"{p}th_percentile"].append(cutoff_value)
                else:
                    cutoff_values[f"{p}th_percentile"].append(None)

        # Create a DataFrame for this gender and store it in the dictionary
        cutoff_df = pd.DataFrame(cutoff_values)
        cutoff_dfs[gender] = cutoff_df
    # Merge and concatenate DataFrames by gender while preserving the original index
    df_merged_female = df[df["gender"] == 0].merge(cutoff_dfs[0], on="age_range", how="left")
    df_merged_male = df[df["gender"] == 1].merge(cutoff_dfs[1], on="age_range", how="left")

    # Concatenate the two DataFrames
    result_df = pd.concat([df_merged_female, df_merged_male])

    # Reset the index to match the original df index
    result_df.set_index(df.index, inplace=True)

    return result_df
###############################################################################
