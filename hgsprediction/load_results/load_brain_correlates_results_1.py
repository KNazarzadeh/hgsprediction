
import os
import numpy as np
import pandas as pd
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


def load_brain_overlap_data_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
    brain_data_type,
    schaefer,
):

    if confound_status == "0":
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    folder_path = os.path.join(
                "/data",
                "project",
                "stroke_ukb",
                "knazarzadeh",
                "project_hgsprediction",  
                "results_hgsprediction",
                "brain_correlation_results",
                f"{population}",
                f"{mri_status}",
                f"{session}_session_ukb",
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                f"{model_name}",
                f"{n_repeats}_repeats_{n_folds}_folds",
                f"{brain_data_type.upper()}_subcorticals_cerebellum",
                f"schaefer{schaefer}",
                f"overlap_data_with_mri_data",         
            )
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_overlap_data_with_mri_data.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
    return df

############################################################################## 
    
def load_brain_hgs_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
    brain_data_type,
    schaefer,
    corr_target,
):


    if confound_status == "0":
        confound = "without_confound_removal"
    else:
        confound = "with_confound_removal"
    folder_path = os.path.join(
                "/data",
                "project",
                "stroke_ukb",
                "knazarzadeh",
                "project_hgsprediction",  
                "results_hgsprediction",
                "brain_correlation_results",
                f"{population}",
                f"{mri_status}",
                f"{session}_session_ukb",
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                f"{model_name}",
                f"{n_repeats}_repeats_{n_folds}_folds",
                f"{brain_data_type.upper()}_subcorticals_cerebellum",
                f"schaefer{schaefer}",           
                "hgs_correlation_results",
                f"{corr_target}",
            )

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        f"{gender}_brain_hgs_correlation_results.csv")
    
    df = pd.read_csv(file_path, sep=',', index_col=0)
    return df
##############################################################################    
