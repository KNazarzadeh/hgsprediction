#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
gender = sys.argv[10]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
if confound_status == '0':
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
            f"{population}",
            f"{mri_status}",
            f"{feature_type}",
            f"{target}",
            f"{confound}",
            f"{model_name}",
            f"{n_repeats}_repeats_{n_folds}_folds",
            f"{gender}",
            "prediction_hgs_on_validation_set_trained",
        )
# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"prediction_hgs_on_validation_set_trained_trained.pkl")

df = pd.read_pickle(file_path)

print("===== Done! =====")
embed(globals(), locals())
###############################################################################
def calculate_correlations(df, n_folds, target):
    df_correlations = pd.DataFrame(columns=["cv_fold", 
                                            "r_values_true_predicted", "r2_values_true_predicted",
                                            "r_values_true_delta", "r2_values_true_delta",
                                            "r_values_true_corrected_predicted", "r2_values_true_corrected_predicted",
                                            "r_values_true_corrected_delta", "r2_values_true_corrected_delta"])

    model = LinearRegression()
    
    df_train = df[df['cv_fold'] != n_folds-2]
    # Beheshti Method:
    X = df_train.loc[:, f"{target}"].values.reshape(-1, 1)
    # y = df.loc[:, f"{target}_delta(true-predicted)"].values
    y = df_train.loc[:, f"{target}_delta(true-predicted)"].values

    model.fit(X, y)

    slope = model.coef_[0]
    intercept = model.intercept_   
    df_test = df[df['cv_fold'] == n_folds-1]
        
    # Beheshti Method:
    # Calculate Slope and Intercept:
    df_test.loc[:, f"{target}_corrected_predicted"] = (df_test.loc[:, f"{target}_predicted"] + ((slope * df_test.loc[:, f"{target}"]) + intercept))
    # Calculate new Delta based on corrected HGS:
    df_test.loc[:, f"{target}_corrected_delta(true-predicted)"] =  df_test.loc[:, f"{target}"] - df_test.loc[:, f"{target}_corrected_predicted"]

    r_values_true_predicted = pearsonr(df_test.loc[:, f"{target}"], df_test.loc[:,f"{target}_predicted"])[0]
    r2_values_true_predicted = r2_score(df_test.loc[:, f"{target}"], df_test.loc[:,f"{target}_predicted"])

    r_values_true_delta = pearsonr(df_test.loc[:, f"{target}"], df_test.loc[:,f"{target}_delta(true-predicted)"])[0]
    r2_values_true_delta = r2_score(df_test.loc[:, f"{target}"], df_test.loc[:,f"{target}_delta(true-predicted)"])

    r_values_true_corrected_predicted = pearsonr(df_test.loc[:, f"{target}"], df_test.loc[:, f"{target}_corrected_predicted"])[0]
    r2_values_true_corrected_predicted = r2_score(df_test.loc[:, f"{target}"], df_test.loc[:, f"{target}_corrected_predicted"])

    r_values_true_corrected_delta = pearsonr(df_test.loc[:, f"{target}"], df_test.loc[:,f"{target}_corrected_delta(true-predicted)"])[0]
    r2_values_true_corrected_delta = r2_score(df_test.loc[:, f"{target}"], df_test.loc[:,f"{target}_corrected_delta(true-predicted)"])

    df_correlations.loc["cv_fold"] = n_folds-1
    df_correlations.loc["r_values_true_predicted"] = r_values_true_predicted
    df_correlations.loc["r2_values_true_predicted"] = r2_values_true_predicted
    df_correlations.loc["r_values_true_delta"] = r_values_true_delta
    df_correlations.loc["r2_values_true_delta"] = r2_values_true_delta
    df_correlations.loc["r_values_true_corrected_predicted"] = r_values_true_corrected_predicted
    df_correlations.loc[ "r2_values_true_corrected_predicted"] = r2_values_true_corrected_predicted
    df_correlations.loc["r_values_true_corrected_delta"] = r_values_true_corrected_delta
    df_correlations.loc["r2_values_true_corrected_delta"] = r2_values_true_corrected_delta

    
    df_correlations = df_correlations.set_index("cv_fold")
    
    return df_test, df_correlations


###############################################################################

df_test, df_correlations = calculate_correlations(df, n_folds, target)

###############################################################################
main_folder_path = os.path.join(
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
        )

subfolders = ["corrected_predictions", "corrected_correlations"]

for subfolder in subfolders:
    subfolder_path = os.path.join(main_folder_path, subfolder)
    
    if not os.path.isdir(subfolder_path):
        os.makedirs(subfolder_path)

    file_path = os.path.join(subfolder_path, f"{gender}_{subfolder}.csv")
    
    if subfolder == "corrected_predictions":
        df_test.to_csv(file_path, sep=',', index=True)
        
    elif subfolder == "corrected_correlations":
        df_correlations.to_csv(file_path, sep=',', index=True)

print("===== Done! =====")
embed(globals(), locals())