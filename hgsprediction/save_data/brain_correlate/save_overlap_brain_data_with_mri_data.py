import os
import pandas as pd
import numpy as np


def save_overlap_brain_data_with_mri_data(
    df,
    feature_type,
    brain_data_type,
    schaefer,
    session,
    gender,
):
    
    folder_path = os.path.join(
                "/data",
                "project",
                "stroke_ukb",
                "knazarzadeh",
                "project_hgsprediction",  
                "results_hgsprediction",
                "brain_correlation_results",
                f"{brain_data_type.upper()}_subcorticals_cerebellum",
                f"schaefer{schaefer}",            
                "barin_data_overlap_with_mri_healthy_data",
                f"{session}_session_ukb",
                f"{feature_type}",
            )
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_brain_{brain_data_type}_Schaefer{schaefer}_overlap_with_mri_healthy_data.csv")

    df.to_csv(file_path, sep=',', index=True)