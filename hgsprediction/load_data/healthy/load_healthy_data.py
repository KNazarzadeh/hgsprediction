
import os
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
def load_original_binned_train_data(
    population,
    mri_status,
):
    """
    load results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be load in specific folder.
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
    df_train = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
        
    return df_train

###############################################################################
def load_validate_hgs_data(
    population,
    mri_status,
    session,
    gender,
    data_set,
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
        DataFrame that should be load in specific folder.
    """
    if mri_status == "nonmri":
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
            f"{data_set}",
            "validated_hgs_data",
            f"{session}_session_ukb"
            )
    else:
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
            "validated_hgs_data",
            f"{session}_session_ukb",
            )

    file_path = os.path.join(
        folder_path,
        f"{gender}_validate_hgs_data.csv")
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
    
    return df

###############################################################################
def load_preprocessed_data(
    population,
    mri_status,
    feature_type,
    session,
    gender,
    data_set,
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
        DataFrame that should be load in specific folder.
    """
    if mri_status == "nonmri":
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
            f"{data_set}",
            "preprocessed_data",
            f"{feature_type}",
            f"{session}_session_ukb"
            )
    else:
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
            "preprocessed_data",
            f"{feature_type}",
            f"{session}_session_ukb",
            )

    file_path = os.path.join(
        folder_path,
        f"{gender}_preprocessed_data.csv")
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
    
    return df

###############################################################################
def load_original_data(
    population,
    mri_status,
):
    """
    load results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be load in specific folder.
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
        "original_data",
        f"{mri_status}_{population}",
    )
    
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"{mri_status}_{population}.csv")
    
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
    
    return df

###############################################################################
def load_ready_training_data(
    population,
    mri_status,
    feature_type,
    target,
    confound_status,
    gender,
    
):
    """
    load results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be load in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    
    if confound_status == "0":
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
        
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "results_hgsprediction",
        f"{population}",
        f"{mri_status}",
        f"{feature_type}",
        f"{target}",
        f"{confound}",
        "data_ready_to_train_models",
        f"{gender}",
    )
    
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        "data_extracted_to_train_models.csv")
    
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
    
    return df

###############################################################################
def load_original_nonmri_test_data(
    population,
    mri_status,
):
    """
    load results to csv file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame that should be load in specific folder.
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
        "test_set",
        "original_test_data",
    )
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"test_set_{mri_status}_{population}.csv")
    
    # Load the dataframe from csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0, low_memory=False)
    
    return df

###############################################################################
def load_extracted_data_by_feature_and_target(
    population,
    mri_status,
    feature_type,
    target,
    session,
    gender,
    data_set,
):
    """
    load results to csv file.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame that should be load in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    if mri_status == "nonmri":
        folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",
        "results_hgsprediction",
        f"{population}",
        f"{mri_status}",
        f"{data_set}",
        f"{session}_session_ukb",
        f"{feature_type}",
        f"{target}",
        "extracted_data_by_feature_and_target",
        )
    else:
        folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "project_hgsprediction",
            "results_hgsprediction",
            f"{population}",
            f"{mri_status}",
            f"{session}_session_ukb",
            f"{feature_type}",
            f"{target}",
            "extracted_data_by_feature_and_target",
            )
        
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"{gender}_extracted_data_by_feature_and_target.csv")
    
    # load the dataframe to csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df
###############################################################################