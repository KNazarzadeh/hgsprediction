
import os
import pandas as pd
import pickle
from hgsprediction.LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def load_best_model_trained(
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
    if confound_status == 0:
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
            f"{gender}",
            f"{feature_type}",
            f"{target}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            "best_model_trained",
        )

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"best_model_trained.pkl")
    
    # Load the model
    with open(file_path, 'rb') as f:
        model_trained = pickle.load(f)
        
    return model_trained

###############################################################################
def load_estimators_trained(
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
    if confound_status == 0:
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
            f"{gender}",
            f"{feature_type}",
            f"{target}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            "estimators_trained",
        )
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"estimators_trained.pkl")
    # Open the pickle file for reading
    with open(file_path, 'rb') as pickle_file:
        loaded_estimators_cells = []
        while True:
            try:
                cell_value = pickle.load(pickle_file)
                loaded_estimators_cells.append(cell_value)
            except EOFError:  # End of file
                break

    return loaded_estimators_cells
############################################################################### 