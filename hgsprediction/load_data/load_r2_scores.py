#!/home/knazarzadeh/miniconda3/envs/disorderspredwp3/bin/python3
"""
Load R2 scores from csv files.
"""
import os
import pandas as pd


###############################################################################
# Load R2 results from csv file.
def load_r2_results(
    population,
    mri_status,
    confound,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    samplesize,
):
    """
    Load R2 results from csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    confound_status: int (binary values 0 or 1)
        Confound status the binary value to define use cofound removal or not.
    gender: str
        Gender to load data
    feature_type:
        The feature type
    target_label:
        The target type
    model: str
        The name of model
    n_repeats:
        The number of repeats that data will load from.
    n_folds
        The number of folds that data will load from.
    """

    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "results",
        "hgs_prediction",
        f"results_{population}",
        f"results_{mri_status}",
        f"results_{gender}_genders",
        f"{feature_type}_features",
        f"{target}_target",
        f"{model}",
        f"{confound}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"{samplesize}",
        "results_csv",
    )
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"cv_object_{mri_status}_{population}_{gender}_genders_{feature_type}_{target}_{model}_{confound}_{n_repeats}_repeats_{n_folds}_folds.csv")

    # load the dataframe to csv file path
    df = pd.read_csv(file_path, sep=',', index_col=False)

    return df
