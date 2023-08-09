
import pandas as pd
import pickle
import os
from hgsprediction.LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
def load_best_model_trained(
    population,
    gender,
    feature_type,
    target,
    confound_status,
    model_name,
    n_repeats,
    n_folds,
):

    if confound_status == 0:
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    if target == "L+R":
        target_label = "L_plus_R"
    else:
        target_label = target
    # Assuming that you have already trained and instantiated the model as `model`
    folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "project_hgsprediction",
            "results_hgsprediction",
            f"{population}",
            "nonmri",
            f"{gender}",
            f"{feature_type}",
            f"{target_label}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            "best_model_trained",
        )

    # Define the csv file path
    file_path = os.path.join(
        folder_path,
        f"best_model_trained.pkl")

    # Load the model
    with open(file_path, 'rb') as f:
        model_trained = pickle.load(f)
        
    return model_trained
