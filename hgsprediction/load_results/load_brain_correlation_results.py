import os
import numpy as np
import pandas as pd
import pickle
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

def load_brain_correlation_overlap_data_with_mri(
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
            "data_overlap_with_mri_healthy",
        )
    
    
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "both_gender_data_overlap_with_mri_healthy.csv")
    
    df_both = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "female_data_overlap_with_mri_healthy.csv")
    
    df_female = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "male_data_overlap_with_mri_healthy.csv")
    
    df_male = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df_both, df_female, df_male
    
###############################################################################
def load_brain_hgs_correlation_results(
    brain_data_type,
    schaefer,
    corr_target,
    
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
            "hgs_correlation_results",
            f"{corr_target}", 
        )
    
    
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "female_hgs_correlation_results.csv")
    
    df_female = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "male_hgs_correlation_results.csv")
    
    df_male = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "both_gender_hgs_correlation_results.csv")
    
    df_corr = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df_female, df_male, df_corr


###############################################################################
def load_brain_hgs_correlation_results_for_plot(
    brain_data_type,
    schaefer,
    corr_target, 
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
            "hgs_correlation_results_for_plots",
            f"{corr_target}", 
        )
    
    
    if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)

    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "female_hgs_correlation_results_for_plots.csv")
    
    df_female = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "male_hgs_correlation_results_for_plots.csv")
    
    df_male = pd.read_csv(file_path, sep=',', index_col=0)
    
    # Define the csv file path to save
    file_path = os.path.join(
        folder_path,
        "both_gender_hgs_correlation_results_for_plots.csv")
    
    df_corr = pd.read_csv(file_path, sep=',', index_col=0)
    
    return df_female, df_male, df_corr
    