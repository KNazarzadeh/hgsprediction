import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
def save_extracted_nonmri_data(
    df,
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
        f"{gender}",
        f"{feature_type}",
        f"{target}",
        f"{confound}",
        f"data_ready_for_prediction",
    )
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"data_extracted.csv")
    # Save the dataframe to csv file path
    df.to_csv(file_path, sep=',', index="SubjectID")

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
    estimators_trained,
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
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"estimators_trained.pkl")
    # Save the model to disk
    with open(file_path, 'wb') as f:
        pickle.dump(estimators_trained, f)
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
            "test_scores_trained",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"test_scores_trained.pkl")
    
    # Save the scores to pickle format
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)
        
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