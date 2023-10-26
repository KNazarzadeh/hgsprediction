import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def parkinson_save_spearman_correlation_results(
    df_corr,
    df_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    parkinson_type,
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
            f"{parkinson_type}",
            f"{feature_type}",
            f"{target}",
            f"{model_name}",
            "spearman_hgs_correlations",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_correlation_coefficient.csv")
    
    df_corr.to_csv(file_path, sep=',', index=True)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_p_values.csv")
    
    df_pvalue.to_csv(file_path, sep=',', index=True)
##############################################################################
