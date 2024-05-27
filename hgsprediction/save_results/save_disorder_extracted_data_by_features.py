import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def save_disorder_extracted_data_by_features(
    df,
    population,
    mri_status,
    session_column,
    feature_type,
    target,
    gender,
    first_event,
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
            f"{first_event}",
            f"{mri_status}",
            f"{session_column}",
            f"{feature_type}",
            f"{target}",
            "extracted_data_by_features",
            f"{gender}",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_extracted_data_by_features.csv")
    
    df.to_csv(file_path, sep=',', index=True)

##############################################################################    
