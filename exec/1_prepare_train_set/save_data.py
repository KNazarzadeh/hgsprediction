#!/usr/bin/env python3
"""Save data in different locations.
    Based on different purposes for saving in different files.
"""

import os


###############################################################################
def save_checked_hgs_availability_data(
    dataframe,
    motor,
    population,
    mri_status
):
    """
    Save data to csv file.

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
    save_folder_path = os.path.join(
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

    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{mri_status}_{population}_hgs_availability.csv")
    # Save the dataframe to csv file path
    print(save_file_path)
    dataframe.to_csv(save_file_path, sep=',', index=False)


###############################################################################
def save_checked_hgs_availability_per_sessions_data(
    data_list,
    motor,
    population,
    mri_status,
):
    """
    Save separate sessions data to csv files.

    Parameters
    ----------
    data_sessions_list : list of pandas.DataFrame
        List of dataFrames that should be save in specific folder.
    motor : str
        Name of the motor which to be analyse.
    population: str
        Name of the population which to  to be analyse
    mri_status: str
        MRI status which data to be  to be analyse.
    """
    sessions = 4
    save_folder_path = os.path.join(
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
        "hgs_availability_per_session",
        f"{mri_status}_{population}"
    )

    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # extract the dataframe from the list of dataframe
    for ses in range(0, sessions):
        # Define the csv file path to save
        save_file_path = os.path.join(
            save_folder_path,
            f"{mri_status}_{population}_hgs_availability_session_{ses}.csv")

        # Save the dataframe to csv file path
        data_list[ses].to_csv(save_file_path, sep=',', index=False)


###############################################################################
def save_original_data_per_session(
    dataframe,
    motor,
    population,
    mri_status,
    ses,
):
    """
    Save data to csv file.

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
    save_folder_path = os.path.join(
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

    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{mri_status}_{population}_original_data_session_{ses}.csv")
    # Save the dataframe to csv file path
    print(save_file_path)
    dataframe.to_csv(save_file_path, sep=',', index=False)
