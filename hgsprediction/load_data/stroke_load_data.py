
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
    
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
  
    return df

###############################################################################
# Load preprocessed data
def load_main_preprocessed_data(population, mri_status, stroke_cohort):
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
        f"{stroke_cohort}_data",
        "primary_preprocess_data",
    )

    file_path = os.path.join(
        folder_path,
        f"primary_preprocess_{stroke_cohort}_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
  
    return df

###############################################################################
# Load preprocessed data
def load_validated_hgs_data(
    population,
    mri_status,
    session_column,
    stroke_cohort,
):
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
        f"{stroke_cohort}_data",
        f"{session_column}_data",
        "validated_hgs_data",
    )

    file_path = os.path.join(
        folder_path,
        f"{session_column}_validated_hgs_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
  
    return df

###############################################################################
# Load preprocessed data
def load_preprocessed_data(population, mri_status, session_column, stroke_cohort):
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
    f"{stroke_cohort}_data",
    f"{session_column}_data",
    "preprocessed_data"
)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_preprocessed_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
  
    return df

###############################################################################
def load_extracted_data(
    population,
    mri_status,
    stroke_cohort,
    visit_session,
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
    if visit_session == "1":
            session_column = f"1st_{stroke_cohort}_session"
    elif visit_session == "2":
        session_column = f"2nd_{stroke_cohort}_session"
    elif visit_session == "3":
        session_column = f"3rd_{stroke_cohort}_session"
    elif visit_session == "4":
        session_column = f"4th_{stroke_cohort}_session"
    
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
        "extracted_data",
    )

    file_path = os.path.join(
        folder_path,
        f"{gender}_{session_column}_extracted_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df
###############################################################################
def load_preprocessed_longitudinal_data(
    population,
    mri_status,
    session_column,
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
        "preprocessed_data",
        f"{mri_status}_{population}",
        "only_longitudinal-stroke_data",
        f"{session_column}_data",  
    )

    file_path = os.path.join(
        folder_path,
        f"{gender}_longitudinal_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)

    return df
