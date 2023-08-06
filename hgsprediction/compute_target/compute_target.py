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

    if target == "hgs_left":
        df.loc[:, f"hgs_left-{session}.0"] = df.loc[:, f"46-{session}.0"]
    elif target == "hgs_right":
        df.loc[:, f"hgs_right-{session}.0"] = df.loc[:, f"47-{session}.0"]        
        
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
