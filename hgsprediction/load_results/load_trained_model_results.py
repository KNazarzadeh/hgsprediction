import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
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
    session,
    data_set,
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
            f"{data_set}",
            f"{session}_session_ukb",
            f"{feature_type}",
            f"{target}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            f"{gender}",
            "best_model_trained",
        )
        

    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"best_model_trained.pkl")
    # load the model to disk
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
    session,
    data_set,
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
            f"{data_set}",
            f"{session}_session_ukb",
            f"{feature_type}",
            f"{target}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            f"{gender}",
            "estimators_trained",
        )

    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"estimators_trained.pkl")
    # load the model to disk
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
    session,
    data_set,
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
            f"{data_set}",
            f"{session}_session_ukb",
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
    r2_df,
    r_df,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
    session,
    data_set,
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
            f"{data_set}",
            f"{session}_session_ukb",
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
        
###############################################################################     
def load_prediction_hgs_on_validation_set(
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
    session,
    data_set,
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
            f"{data_set}",
            f"{session}_session_ukb",
            f"{feature_type}",
            f"{target}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            f"{gender}",
            "prediction_hgs_on_validation_set_trained",
        )
        
    # Define the csv file path to load
    file_path = os.path.join(
        folder_path,
        f"prediction_hgs_on_validation_set_trained_trained.pkl")
    
    # load the scores to pickle format
    df = pd.read_pickle(file_path)
    
    return df