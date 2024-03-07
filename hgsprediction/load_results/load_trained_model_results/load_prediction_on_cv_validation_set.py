import os
import numpy as np
import pandas as pd


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
###############################################################################
# This class define all required features from data:
def load_prediction_on_cv_validation_set(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
):
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
        f"{model_name}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        f"{gender}",
        "prediction_hgs_on_validation_set_trained",
        )

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"prediction_hgs_on_validation_set_trained_trained.pkl")

    df = pd.read_pickle(file_path)
    
    return df