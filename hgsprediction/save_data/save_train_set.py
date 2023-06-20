import os
###############################################################################
def save_preprocessed_train_df(
    population,
    mri_status,
    df,
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
    df = df.rename(columns={"eid": "SubjectID"})
    df.set_index("SubjectID", inplace=True)
    df.to_csv(file_path, sep=',')
