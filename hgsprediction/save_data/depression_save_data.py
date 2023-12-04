
import os
import pandas as pd
from ptpython.repl import embed


def save_main_preprocessed_data(
    df,
    population,
    mri_status,
    depression_cohort,
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
        f"{depression_cohort}_data",
        "primary_preprocess_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"primary_preprocess_{depression_cohort}_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  
###############################################################################
def save_primary_extracted_data(
    df,
    population,
    mri_status,
    session_column,
    depression_cohort,
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
        f"{depression_cohort}_data",
        f"{session_column}_data",
        "primary_extracted_data",
)

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_original_extracted_pre_post_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)

###############################################################################
def save_validated_hgs_data(
    df,
    population,
    mri_status,
    session_column,
    depression_cohort,
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
        f"{depression_cohort}_data",
        f"{session_column}_data",
        "validated_hgs_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_validated_hgs_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  
###############################################################################
def save_preprocessed_data(
    df,
    population,
    mri_status,
    session_column,
    depression_cohort,
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
        f"{depression_cohort}_data",
        f"{session_column}_data",
        "preprocessed_data"
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_preprocessed_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)

###############################################################################
###############################################################################
###############################################################################
def save_subgroups_only_extracted_data(
    df,
    population,
    mri_status,
    session_column,
    depression_cohort,
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
        f"{depression_cohort}_data",
        f"only_{depression_cohort}_no_longitudinal_data",
        f"{session_column}_data",
        "primary_extracted_data",
)

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_original_extracted_pre_post_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
###############################################################################
def save_subgroups_only_validated_hgs_data(
    df,
    population,
    mri_status,
    session_column,
    depression_cohort,
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
        f"{depression_cohort}_data",
        f"only_{depression_cohort}_no_longitudinal_data",
        f"{session_column}_data",
        "validated_hgs_data",
)

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{session_column}_original_extracted_pre_post_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
###############################################################################    
def save_subgroups_only_preprocessed_data(
    df,
    population,
    mri_status,
    depression_cohort,
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
        f"{depression_cohort}_data",
        f"only_{depression_cohort}_no_longitudinal_data",
        "primary_preprocess_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"primary_preprocess_{depression_cohort}_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
###############################################################################    
def save_extracted_data(
    df,
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
        "extracted_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{gender}_{session_column}_extracted_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)

###############################################################################