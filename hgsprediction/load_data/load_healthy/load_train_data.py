
import os
import pandas as pd

###############################################################################
def load_preprocessed_train_df(
    population,
    mri_status,
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
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        "binned_data",
        f"{mri_status}",
        "train_set",
    )
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"preprocessed_train_{mri_status}_{population}.csv")
    # Load the dataframe from csv file path
    df_train = pd.read_csv(file_path, sep=',', index_col=0)
    # df_train = df_train.rename(columns={"eid": "SubjectID"})
    # df_train.set_index("SubjectID", inplace=True)
    
    return df_train

###############################################################################
###############################################################################
def load_train_set_df(
    population,
    mri_status,
):
    """
    Save results to csv file.

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
    """

    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        "data_hgs",
        f"{population}",
        "preprocessed_data",
        "binned_data",
        f"{mri_status}",
    )
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"train_set_{mri_status}_{population}.csv")
    # Save the dataframe to csv file path
    df_train = pd.read_csv(file_path, sep=',', index_col=False)
    
    return df_train