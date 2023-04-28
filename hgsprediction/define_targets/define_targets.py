#!/usr/bin/env Disorderspredwp3
"""Define the Features for extracting from populations data."""

# Authors: Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>
# License: AGPL

import pandas as pd


###############################################################################
# Define the target which should be predict.
def define_targets(
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
