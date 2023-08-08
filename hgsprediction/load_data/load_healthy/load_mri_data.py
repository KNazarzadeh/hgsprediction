
import os
import pandas as pd

def load_mri_data(
   population,
):
    session = 2
    mri_status = "mri"
    # Read CSV file from Juseless
    folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "GIT_repositories",
            "motor_ukb",
            "data_ukb",
            "data_hgs",
            population,
            "preprocessed_data",
        )

    folder_path = os.path.join(
        folder_path,
        "hgs_availability_per_session",
        f"{mri_status}_{population}"
    )
    file_path = os.path.join(
        folder_path,
        f"{mri_status}_{population}_hgs_availability_session_{session}.csv")

    data = pd.read_csv(file_path, sep=',')
    
    # data.rename(columns={'eid': 'SubjectID'}, inplace=True)
    # data = data.set_index('SubjectID')

    return data

def load_mri_data_for_anthropometrics(
   population,
):
    session = 2
    mri_status = "mri"
    # Read CSV file from Juseless
    folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "GIT_repositories",
            "hgsprediction",
            "results",
            f"{population}",
            f"{mri_status}",
            "preprocessed_data",
        )
    file_path = os.path.join(
        folder_path,
        f"{mri_status}_{population}_preprocessed_data.csv")

    data = pd.read_csv(file_path, sep=',', index_col=0)

    return data