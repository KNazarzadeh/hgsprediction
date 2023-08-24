
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
# Load preprocessed data
def load_main_preprocessed_data(population, mri_status):
    """Get data from the preprocessed csv file.
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
        "preprocessed_data",
        f"{mri_status}_{population}",
        "original_preprocessed_data",
    )

    file_path = os.path.join(
        folder_path,
        "original_preprocessed_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
  
    return df

###############################################################################
# Load preprocessed data
def load_validated_hgs_data(population, mri_status, session_column):
    """Get data from the preprocessed csv file.
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
        "preprocessed_data",
        f"{mri_status}_{population}",
        f"{session_column}_data",
    )

    file_path = os.path.join(
        folder_path,
        f"{session_column}_validated_hgs_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
  
    return df

###############################################################################
# Load preprocessed data
def load_preprocessed_data(population, mri_status, session_column):
    """Get data from the preprocessed csv file.
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
        "preprocessed_data",
        f"{mri_status}_{population}",
        f"{session_column}_data",
    )

    file_path = os.path.join(
        folder_path,
        f"{session_column}_preprocessed_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
  
    return df

###############################################################################
def load_extracted_data(
    population,
    mri_status,
    session_column,
    feature_type,
    target,
    gender,
):
    """
    Save data to csv file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "extracted_data",
        f"{mri_status}_{population}",
        f"{session_column}_data",
        f"{feature_type}",
        f"{target}",
        f"{gender}",
    )

    file_path = os.path.join(
        folder_path,
        f"{session_column}_extracted_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df
