import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def load_prediction_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,    
    correlation_type,
    data_set,
):
    if confound_status == "0":
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    # Assuming that you have already trained and instantiated the model as `model`
    if mri_status == "nonmri":
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
                "hgs_prediction_correlation_results",
            )
    else:
        folder_path = os.path.join(
                "/data",
                "project",
                "stroke_ukb",
                "knazarzadeh",
                "project_hgsprediction",  
                "results_hgsprediction",
                f"{population}",
                f"{mri_status}",
                f"{session}_session_ukb",
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                f"{model_name}",
                f"{n_repeats}_repeats_{n_folds}_folds",            
                "hgs_prediction_correlation_results",
            )

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_{correlation_type}_correlation_values.csv")
    
    df_correlation_values = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_r2_values.csv")
    
    df_r2_values = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_MAE_values.csv")
    
    df_mae_values = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df_correlation_values, df_r2_values, df_mae_values
##############################################################################
