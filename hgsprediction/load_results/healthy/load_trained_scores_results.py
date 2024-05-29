import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################     
def load_scores_trained(
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
):
    if confound_status == "0":
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"

    # Assuming that you have already trained and instantiated the model as `model`
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
            "scores_trained",
        )
    
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"scores_trained.pkl")
        
    # load the scores to pickle format
    df = pd.read_pickle(file_path)

    return df
###############################################################################     
def load_test_scores_trained(
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
):
    if confound_status == "0":
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"

    # Assuming that you have already trained and instantiated the model as `model`
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
            "test_scores_trained",
        )

    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"test_r2_scores_trained.pkl")
    
    # load the scores to pickle format
    r2_df = pd.read_pickle(file_path)
        
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"test_pearson_r_scores_trained.pkl")
    
    # load the scores to pickle format
    r_df = pd.read_pickle(file_path)
    
    return r2_df, r_df