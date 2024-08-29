import os
import pandas as pd
import numpy as np


def load_overlap_brain_data_with_mri_data(
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
            )
    if(os.path.isdir(folder_path)):
        # Define the csv file path to load
        file_path = os.path.join(
            folder_path,
            f"{gender}_brain_{brain_data_type}_Schaefer{schaefer}_overlap_with_mri_healthy_data.csv")

        df = pd.read_csv(file_path, sep=',', index_col=0)
    else:
        df = pd.DataFrame()
        
    return df