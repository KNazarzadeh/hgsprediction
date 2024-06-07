import os
import pandas as pd
import numpy as np


def load_original_brain_data(
    brain_data_type,
    schaefer,
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
                "brain_ready_data",
            )
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"brain_{brain_data_type}_Schaefer{schaefer}_data.csv")

    df_brain = pd.read_csv(file_path, sep=',', index_col=0)

    df_brain.index = 'sub-' + df_brain.index.astype(str)
    
    return df_brain