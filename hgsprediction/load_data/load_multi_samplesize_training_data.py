
import pandas as pd
import numpy as np
import os

def load_multi_samplesize_training_data(
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    session,
    data_set,
    samplesize,
):
    
        if confound_status == '0':
            confound = "without_confound_removal"
        elif confound_status == '1':
            confound = "with_confound_removal"
        
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
                f"{data_set}",
                f"{session}_session_ukb",                
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                "multi_samplesize_results",
                "data_ready_to_train_models",
                f"results_samples_{samplesize}",
            )
        
        # Define the csv file path to save
        file_path = os.path.join(
            folder_path,
            f"{gender}_ready_training_data.csv")
        
        df_sample = pd.read_csv(file_path, sep=',', index_col=0)
        
        return df_sample