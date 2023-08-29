import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def save_correlations_plot(
    x,
    y,
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
            "plots",
            f"{population}",
            f"{mri_status}",
            f"{session_column}",
            f"{model_name}",
            f"{feature_type}",
            f"{target}",
            f"{gender}",
            "hgs_correlations",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{y}_vs_{x}.png")
    
    return file_path