import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def load_spearman_correlation_results(
    population,
    mri_status,
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
            f"{feature_type}",
            f"{target}",
            f"{model_name}",
            "spearman_hgs_correlations",
        )
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_correlation_coefficient.csv")
    
    df_corr = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_p_values.csv")
    df_pvalue = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df_corr, df_pvalue
