
import os
import pandas as pd
from ptpython.repl import embed


def save_main_preprocessed_data(
    df,
    population,
    mri_status,
    stroke_cohort,
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
        f"{stroke_cohort}_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{stroke_cohort}_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  
###############################################################################
def save_extracted_pre_post_data(
    df,
    population,
    mri_status,
    session_column,
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
        f"{session_column}_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_original_extracted_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)

###############################################################################
def save_validated_hgs_data(
    df,
    population,
    mri_status,
    session_column,
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
        f"{session_column}_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_validated_hgs_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  
###############################################################################
def save_computed_features_data(
    df,
    population,
    mri_status,
    session_column,
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
        f"{session_column}_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_computed_features_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)

###############################################################################
def save_computed_targets_data(
    df,
    population,
    mri_status,
    session_column,
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
        f"{session_column}_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_computed_targets_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  
###############################################################################
def save_extracted_features_data(
    df,
    population,
    mri_status,
    session_column,
    feature_type,
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
        f"{session_column}_data",
        "extracted_features",
        f"{feature_type}",
        f"{gender}",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_extracted_features_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)

###############################################################################
def save_extracted_target_data(
    df,
    population,
    mri_status,
    session_column,
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
        "preprocessed_data",
        f"{mri_status}_{population}",
        f"{session_column}_data",
        "extracted_target",
        f"{target}",
        f"{gender}",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_extracted_target_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  
###############################################################################