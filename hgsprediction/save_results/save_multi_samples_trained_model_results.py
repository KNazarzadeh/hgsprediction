
import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
def save_best_model_trained(
    model_trained,
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
            "best_model_trained",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"best_model_trained.pkl")
    # Save the model to disk
    with open(file_path, 'wb') as f:
        pickle.dump(model_trained, f)

###############################################################################
def save_estimators_trained(
    df_estimators,
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
            "estimators_trained",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"estimators_trained.pkl")
    # Save the model to disk
    # Flatten the DataFrame and save cells as pickle
    cells = df_estimators.values.flatten()  # Flatten the DataFrame into a 1D array
    with open(file_path, 'wb') as pickle_file:
        for cell_value in cells:
            pickle.dump(cell_value, pickle_file)

###############################################################################     
def save_scores_trained(
    df,
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
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"scores_trained.pkl")
    
    # Save the scores to pickle format
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)
     
###############################################################################     
def save_test_scores_trained(
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
            "test_scores_trained",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"test_r2_scores_trained.pkl")
    
    # Save the scores to pickle format
    with open(file_path, 'wb') as f:
        pickle.dump(r2_df, f)
        
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"test_pearson_r_scores_trained.pkl")
    
    # Save the scores to pickle format
    with open(file_path, 'wb') as f:
        pickle.dump(r_df, f)
        
###############################################################################     
def save_prediction_hgs_on_validation_set(
    df,
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
            "prediction_hgs_on_validation_set_trained",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"prediction_hgs_on_validation_set_trained_trained.pkl")
    
    # Save the scores to pickle format
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)