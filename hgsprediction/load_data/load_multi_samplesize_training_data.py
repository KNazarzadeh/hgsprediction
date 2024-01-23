
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
    model_name,
    n_repeats,
    n_folds,
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
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                f"{model_name}",
                f"{n_repeats}_repeats_{n_folds}_folds",
                f"{gender}",
                "multi_samplesize_results",
                f"results_samples_{samplesize}",
                f"ready_training_data",
            )

        # Define the csv file path to save
        file_path = os.path.join(
            folder_path,
            f"ready_training_data.csv")
        
        df_sample = pd.read_csv(file_path, sep=',', index_col=0)
        
        return df_sample