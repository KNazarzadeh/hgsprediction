import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
def save_extracted_mri_data(
    df,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
):
    """
    Save results to csv file.

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
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "results_hgsprediction",
        f"{population}",
        f"{mri_status}",
        f"{gender}",
        f"{feature_type}",
        f"{target_label}",
        f"{confound}",
        f"data_ready_for_test",
    )
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"data_extracted.csv")
    # Save the dataframe to csv file path
    df.to_csv(file_path, sep=',', index="SubjectID")
###############################################################################
def save_tested_mri_data(
    df,
    population,
    mri_status,
    gender,
    feature_type,
    target,
    confound_status,
    model_name,
    n_repeats,
    n_folds,
):
    """
    Save results to csv file.

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
    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target

    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "results_hgsprediction",
        f"{population}",
        f"{mri_status}",
        f"{gender}",
        f"{feature_type}",
        f"{target_label}",
        f"{confound}",
        f"{model_name}",
        f"{n_repeats}",
        f"{n_folds}",
        f"mri_data_tested",
    )
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"mri_data_tested.csv")
    # Save the dataframe to csv file path
    df.to_csv(file_path, sep=',', index="SubjectID")
