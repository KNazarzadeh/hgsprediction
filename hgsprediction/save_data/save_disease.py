
import os



def save_prepared_disease_data(
    df,
    df_name,
    motor,
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
        "prepared_data",
        f"{mri_status}_{population}",
    )

    if(not os.path.isdir(save_folder_path)):
        os.makedirs(save_folder_path)

    # Define the csv file path to save
    save_file_path = os.path.join(
        save_folder_path,
        f"{df_name}_{mri_status}_{population}.csv")
    
    df.to_csv(save_file_path, sep=',')

