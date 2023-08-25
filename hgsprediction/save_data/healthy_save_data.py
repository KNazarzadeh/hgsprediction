import os
import pandas as pd
from ptpython.repl import embed


def save_validate_hgs_data(
    df,
    population,
    mri_status,
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
        "preprocessed_train_data",
        f"{mri_status}_{population}",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"validate_hgs_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  
###############################################################################
def save_preprocessed_train_data(
    df,
    population,
    mri_status,
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
        "preprocessed_train_data",
        f"{mri_status}_{population}",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"preprocessed_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  