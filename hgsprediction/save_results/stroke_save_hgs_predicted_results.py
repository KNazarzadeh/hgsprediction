import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def stroke_save_hgs_predicted_results(
    df,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
):
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
            f"{model_name}",
            "hgs_predicted_results",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_predicted_results.csv")
    
    df.to_csv(file_path, sep=',', index=True)

##############################################################################    
def stroke_save_hgs_predicted_results_mri_records_sessions_only(
    df,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
):
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
            "mri_records_sessions_only",
            f"{feature_type}",
            f"{target}",
            f"{model_name}",
            "hgs_predicted_results",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_predicted_results.csv")
    
    df.to_csv(file_path, sep=',', index=True)