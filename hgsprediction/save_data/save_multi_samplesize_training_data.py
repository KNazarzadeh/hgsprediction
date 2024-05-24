
import pandas as pd
import numpy as np
import os

def save_multi_samplesize_training_data(
    df_sample,
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
                "multi_samplesize_results",
                "data_ready_to_train_models",
                f"results_samples_{samplesize}",
            )
            
        if(not os.path.isdir(folder_path)):
            os.makedirs(folder_path)

        # Define the csv file path to save
        file_path = os.path.join(
            folder_path,
            f"{gender}_ready_training_data.csv")
        
        df_sample.to_csv(file_path, sep=',', index=True)