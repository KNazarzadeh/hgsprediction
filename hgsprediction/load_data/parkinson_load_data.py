
#!/usr/bin/env python3
"""Load data from different locations.
    Based on different purposes for loading in different files.
"""

import os
import pandas as pd

###############################################################################
# Load original data from the the original folder
# which fetched from UK Biobank data
def load_original_data(population, mri_status):
    """Get data from the original csv file.

    Parameters
    ----------
    motor : str
     Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame of data specified.
    """
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "original_data",
        f"{mri_status}_{population}"
    )

    file_path = os.path.join(
        folder_path,
        f"{mri_status}_{population}.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
  
    return df
###############################################################################
def load_validate_hgs_data(
    population,
    mri_status,
    parkinson_type,
    gender,
):
    """
    Load train set after binned process.
    90 percent train from the binned data.
    Parameters
    ----------
    population: str
        Specifies the population.
    mri_status: str
        Specifies the MRI status.
    Return
    ----------    
    df : pandas.DataFrame
        DataFrame that should be save in specific folder.
    """
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        f"{mri_status}_{population}",
        "validated_hgs_data",
        f"{parkinson_type}",
    )

    file_path = os.path.join(
        folder_path,
        f"{gender}_validate_hgs_data.csv")
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
    
    return df
###############################################################################
def load_preprocessed_data(
    population,
    mri_status,
    parkinson_type,
    gender,
):
    """
    Load train set after binned process.
    90 percent train from the binned data.
    Parameters
    ----------
    population: str
        Specifies the population.
    mri_status: str
        Specifies the MRI status.
    Return
    ----------    
    df : pandas.DataFrame
        DataFrame that should be save in specific folder.
    """
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        f"{mri_status}_{population}",
        "preprocessed_data",
        f"{parkinson_type}",
    )

    file_path = os.path.join(
        folder_path,
        f"{gender}_preprocessed_data.csv")
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
    
    return df