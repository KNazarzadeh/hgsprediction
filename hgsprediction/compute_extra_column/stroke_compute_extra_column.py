
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
def calculate_cutoff_value_hgs(df, session_column):

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

