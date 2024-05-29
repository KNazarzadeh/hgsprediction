import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def save_disorder_matched_control_samples_correlation_results(
    df_corr,
    df_p_values,
    df_r2_values,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
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
            f"{session_column}",
            f"{feature_type}",
            f"{target}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            "matched_control_samples_results",
            f"1_to_{n_samples}_samples",
            "matched_control_samples_correlation_results",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_matched_control_correlations.csv")
    
    df_corr.to_csv(file_path, sep=',', index=True)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_matched_control_p_values.csv")
    
    df_p_values.to_csv(file_path, sep=',', index=True)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_matched_control_r2_values.csv")
    
    df_r2_values.to_csv(file_path, sep=',', index=True)
