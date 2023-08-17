
import os
import pandas as pd

def save_preprocessed_data(
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
        f"{mri_status}_{population}",
        "preprocessed_data",
        f"{stroke_cohort}_data",
    )

    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{stroke_cohort}_data.csv")
    
    df.to_csv(file_path, sep=',', index=True)
  
