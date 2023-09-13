import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def load_hgs_predicted_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
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
            f"{session}_session_ukb",
            f"{feature_type}",
            f"{target}",
            f"{model_name}",
            "hgs_predicted_results",
        )

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_hgs_predicted_results.csv")
    
    df = pd.read_csv (file_path, sep=',', index_col=0)
    
    return df
