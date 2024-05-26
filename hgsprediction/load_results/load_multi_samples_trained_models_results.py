
import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
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
    samplesize,
):
    if confound_status == "0":
        confound = "without_confound_removal"
    elif confound_status == "1":
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
            "multi_samplesize_results",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            f"{gender}",
            f"results_samples_{samplesize}",
            "scores_trained",
        )

    # Define the file path to load
    file_path = os.path.join(
        folder_path,
        f"scores_trained.pkl")

    # Load the scores to pickle format
    df = pd.read_pickle(file_path)
    
    return df
############################################################################### 