import os
import pandas as pd
import numpy as np


def load_removed_tiv_from_brain_data(
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
                "brain_data_without_tiv",
                "session_2_and_3_ukb",
            )
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"brain_{brain_data_type}_Schaefer{schaefer}_without_tiv_data.csv")

    df = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df