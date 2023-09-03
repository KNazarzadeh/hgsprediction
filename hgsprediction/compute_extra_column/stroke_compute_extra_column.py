
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
        
    if extra_column == "years":
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