#!/usr/bin/env python3
"""Load data from different locations.
    Based on different purposes for loading in different files.
"""

import os
import pandas as pd

###############################################################################
# Load specific data for specific session
def load_hgs_data_per_session(motor, population, mri_status, session):
    """
    load data from the relative csv file.

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
    pandas.DataFrame
        DataFrame of data specified.
    """
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        f"data_{motor}",
        population,
        "preprocessed_data",
    )
    # if population is healthy, we need to load data for session 0
    # with more available data.
    if population == "healthy":
        folder_path = os.path.join(
            folder_path,
            "hgs_availability_per_session",
            f"{mri_status}_{population}"
        )
    file_path = os.path.join(
        folder_path,
        f"{mri_status}_{population}_hgs_availability_session_{session}.csv")

    data = pd.read_csv(file_path, sep=',')
    
    # Focus on Session/Instance 0 only
    # data = data[data.columns[~data.columns.str.contains("-1.*|-2.*|-3.*")]]

    return data

###############################################################################
###############################################################################
# Load original data from the the original folder
# which fetched from UK Biobank data
def load_original_data(motor, population, mri_status):
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
    pandas.DataFrame
        DataFrame of data specified.
    """
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        f"data_{motor}",
        population,
        "original_data",
        f"{mri_status}_{population}"
    )

    file_path = os.path.join(
        folder_path,
        f"{mri_status}_{population}.csv")
    
    data = pd.read_csv(file_path, sep=',')

    return data

###############################################################################
# Load specific data for specific session
def load_original_data_per_session(motor, population, mri_status, session):
    """
    load data from the relative csv file.

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
    pandas.DataFrame
        DataFrame of data specified.
    """
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        f"data_{motor}",
        population,
        "original_data",
        f"{mri_status}_{population}",
        "original_data_per_sessions",
    )
    
    file_path = os.path.join(
        folder_path,
        f"{mri_status}_{population}_original_data_session_{session}.csv")

    data = pd.read_csv(file_path, sep=',')
    
    # Focus on Session/Instance 0 only
    # data = data[data.columns[~data.columns.str.contains("-1.*|-2.*|-3.*")]]

    return data
###############################################################################
# Load specific data for specific session
def load_hgs_availability_data_per_session(motor, population, mri_status, session):
    """
    load data from the relative csv file.

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
    pandas.DataFrame
        DataFrame of data specified.
    """
    folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        f"data_{motor}",
        population,
        "preprocessed_data",
    )
    # if population is healthy, we need to load data for session 0
    # with more available data.
    if population == "healthy":
        folder_path = os.path.join(
            folder_path,
            "hgs_availability_per_session",
            f"{mri_status}_{population}"
        )
    file_path = os.path.join(
        folder_path,
        f"{mri_status}_{population}_hgs_availability_session_{session}.csv")

    data = pd.read_csv(file_path, sep=',')
    
    # Focus on Session/Instance 0 only
    # data = data[data.columns[~data.columns.str.contains("-1.*|-2.*|-3.*")]]

    return data

###############################################################################
# Load data from the feature extraction
def load_hgs_disease_data(motor, population, mri_status):
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
    pandas.DataFrame
        DataFrame of data specified.
    """
    data_folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        f"data_{motor}",
        population,
        "preprocessed_data",
        "hgs_availability_all_sessions",
        f"{mri_status}_{population}"
    )
    
    data_file_path = os.path.join(
        data_folder_path,
        f"{mri_status}_{population}_hgs_availability.csv")

    data = pd.read_csv(data_file_path, sep=',')

    return data
