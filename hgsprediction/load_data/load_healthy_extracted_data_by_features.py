import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
def load_extracted_data_by_features(
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
        f"data_ready_to_train_models",
        f"{gender}",
    )
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"data_extracted_to_train_models.csv")
    # Save the dataframe to csv file path
    df = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df

###############################################################################