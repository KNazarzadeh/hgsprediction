
import os
import pandas as pd

###############################################################################
def load_original_binned_train_data(
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
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "splitted_data",
        f"{mri_status}",
        "train_set",
        "original_binned_train_data"
    )
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"original_binned_train_data.csv")
    # Load the dataframe from csv file path
    # Specify 'Name' as the index column
    df_train = pd.read_csv(file_path, sep=',', index_col=0)
        
    return df_train
###############################################################################
def load_original_preprocessed_train_data(
    population,
    gender,
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
        "project_hgsprediction",
        "data_hgs",
        f"{population}",
        "splitted_data",
        f"{mri_status}",
        "train_set",
        "original_preprocessed_train_data"
    )
    if gender == "both_gender":
        # Define the csv file path to load
        file_path = os.path.join(
            folder_path,
            f"original_preprocessed_train_data.csv")
        # Load the dataframe from csv file path
        # Specify 'Name' as the index column
        df_train = pd.read_csv(file_path, sep=',', index_col=0)
        
    return df_train
###############################################################################
def load_preprocessed_train_data(
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
        "project_hgsprediction",
        "results_hgsprediction",
        f"{population}",
        f"{mri_status}",
        "preprocessed_train_data",
    )
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"preprocessed_train_data.csv")
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df
###############################################################################
###############################################################################
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
        "project_hgsprediction",
        "results_hgsprediction",
        f"{population}",
        f"{mri_status}",
        "preprocessed_train_set",
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

def load_primary_mri_df(
    population,
    gender,
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
        "original_data",
        f"{mri_status}_healthy",
    )
    if gender == "both_gender":
        # Define the csv file path to load
        file_path = os.path.join(
            folder_path,
            f"{mri_status}_{population}.csv")
        # Load the dataframe from csv file path
        # Specify 'Name' as the index column
        # df_train = pd.read_csv(file_path, sep=',', index_col="SubjectID")
        df_train = pd.read_csv(file_path, sep=',')
    return df_train

###############################################################################
###############################################################################
###############################################################################
def load_preprocessed_train_data(
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
        "project_hgsprediction",
        "results_hgsprediction",
        f"{population}",
        f"{mri_status}",
        "preprocessed_train_set",
    )
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"preprocessed_train_data.csv")
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df