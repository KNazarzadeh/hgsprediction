import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def save_spearman_results(
    df_corr,
    df_pvalue,
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
            f"{gender}",
            "spearman_hgs_correlations",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"correlation_coefficient.csv")
    
    df_corr.to_csv(file_path, sep=',', index=True)
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"p_values.csv")
    df_pvalue.to_csv(file_path, sep=',', index=True)
    