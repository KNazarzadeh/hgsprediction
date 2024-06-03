import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def save_posthoc_results(
    df_pairwise_posthoc,
    df_posthoc_summary,
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
    gender,
    anova_type,
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
            f"{anova_target}",
            f"{anova_type}",
        )
        
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    file_path = os.path.join(
        folder_path,
        f"{gender}_pairwise_posthoc_result_table.pkl")
        
    df_pairwise_posthoc.to_csv(file_path, sep=',', index=True)

    file_path = os.path.join(
        folder_path,
        f"{gender}_tukeyhsd_posthoc_result_table.pkl")
        
    with open(file_path, 'wb') as f:
        pickle.dump(df_posthoc_summary, f)