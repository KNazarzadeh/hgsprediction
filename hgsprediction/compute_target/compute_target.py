#!/usr/bin/env Disorderspredwp3
"""Define the Features for extracting from populations data."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>
# License: AGPL

import pandas as pd

###############################################################################
def compute_target(df, mri_status, target):
    if mri_status == "nonmri":
        session = 0
    elif mri_status == "mri":
        session = 2

    if target == "left_hgs":
        df.loc[:, f"left_hgs-{session}.0"] = df.loc[:, f"46-{session}.0"]
    elif target == "right_hgs":
        df.loc[:, f"right_hgs-{session}.0"] = df.loc[:, f"47-{session}.0"]        
        
    return df

###############################################################################
# Define the target which should be predict.
def extract_target(
    df,
    target,
):
    """
    Define target.

    Parameters
    ----------
    population: str
        Name of the population which to  to be analyse.

    Returns
    --------
    target : str
        List of different list of features.

    """
    filter_col = [col for col in df if col.startswith(f"{target}_hgs")]
    y = filter_col[0]

    return y
