
import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def save_prepared_data_for_jamovi_software(
    df,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    confound_status,
    n_repeats,
    n_folds,
    n_samples,
    anova_target,
    first_event,
):
    if confound_status == "0":
        confound = "without_confound_removal"
    else:
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
        f"{first_event}",
        f"{mri_status}",
        f"{session_column}",
        f"{feature_type}",
        f"{target}",
        f"{confound}",
        f"{model_name}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        "matched_control_samples_results",
        f"1_to_{n_samples}_samples",
        "ANOVA_results",
        "jamovi_software",
        f"{anova_target}",
        "ready_data",
        )
    
    if(not os.path.isdir(folder_path)):
            os.makedirs(folder_path)
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "anova_gender_group_time_point_data.csv")

    df.to_csv(file_path, sep=',', index=True)
