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
from hgsprediction.load_results.healthy import load_trained_model_results

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
data_set = sys.argv[10]
gender = sys.argv[11]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df = load_trained_model_results.load_prediction_hgs_on_validation_set(
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    n_repeats,
    n_folds,
    session,
    data_set,
)

###############################################################################
def calculate_correlations(df, n_folds, target):
    df_corrected = pd.DataFrame()
    df_correlations = pd.DataFrame(columns=["cv_fold", 
                                                    "r_values_true_predicted", "r2_values_true_predicted",
                                                    "r_values_true_delta", "r2_values_true_delta",
                                                    "r_values_true_corrected_predicted", "r2_values_true_corrected_predicted",
                                                    "r_values_true_corrected_delta", "r2_values_true_corrected_delta"])

    model = LinearRegression()
    for fold in range(int(n_folds)):
        df_tmp = df[df['cv_fold']==fold]
        df_half = df_tmp.sample(frac=0.5, random_state=47)

        # Beheshti Method:
        X = df_half.loc[:, f"{target}"].values.reshape(-1, 1)
        # y = df.loc[:, f"{target}_delta(true-predicted)"].values
        y = df_half.loc[:, f"{target}_delta(true-predicted)"].values

        model.fit(X, y)

        slope = model.coef_[0]
        intercept = model.intercept_   
        df_half_rest = df_tmp[~df_tmp.index.isin(df_half.index)]
        
        # Beheshti Method:
        df_half_rest.loc[:, f"{target}_corrected_predicted"] = (df_half_rest.loc[:, f"{target}_predicted"] + ((slope * df_half_rest.loc[:, f"{target}"]) + intercept))
        df_half_rest.loc[:, f"{target}_corrected_delta(true-predicted)"] =  df_half_rest.loc[:, f"{target}"] - df_half_rest.loc[:, f"{target}_corrected_predicted"]

        r_values_true_predicted = pearsonr(df_half_rest.loc[:, f"{target}"], df_half_rest.loc[:,f"{target}_predicted"])[0]
        r2_values_true_predicted = r2_score(df_half_rest.loc[:, f"{target}"], df_half_rest.loc[:,f"{target}_predicted"])

        r_values_true_delta = pearsonr(df_half_rest.loc[:, f"{target}"], df_half_rest.loc[:,f"{target}_delta(true-predicted)"])[0]
        r2_values_true_delta = r2_score(df_half_rest.loc[:, f"{target}"], df_half_rest.loc[:,f"{target}_delta(true-predicted)"])

        r_values_true_corrected_predicted = pearsonr(df_half_rest.loc[:, f"{target}"], df_half_rest.loc[:, f"{target}_corrected_predicted"])[0]
        r2_values_true_corrected_predicted = r2_score(df_half_rest.loc[:, f"{target}"], df_half_rest.loc[:, f"{target}_corrected_predicted"])

        r_values_true_corrected_delta = pearsonr(df_half_rest.loc[:, f"{target}"], df_half_rest.loc[:,f"{target}_corrected_delta(true-predicted)"])[0]
        r2_values_true_corrected_delta = r2_score(df_half_rest.loc[:, f"{target}"], df_half_rest.loc[:,f"{target}_corrected_delta(true-predicted)"])

        df_correlations.loc[fold, "cv_fold"] = fold
        df_correlations.loc[fold, "r_values_true_predicted"] = r_values_true_predicted
        df_correlations.loc[fold, "r2_values_true_predicted"] = r2_values_true_predicted
        df_correlations.loc[fold, "r_values_true_delta"] = r_values_true_delta
        df_correlations.loc[fold, "r2_values_true_delta"] = r2_values_true_delta
        df_correlations.loc[fold, "r_values_true_corrected_predicted"] = r_values_true_corrected_predicted
        df_correlations.loc[fold, "r2_values_true_corrected_predicted"] = r2_values_true_corrected_predicted
        df_correlations.loc[fold, "r_values_true_corrected_delta"] = r_values_true_corrected_delta
        df_correlations.loc[fold, "r2_values_true_corrected_delta"] = r2_values_true_corrected_delta


        df_corrected = pd.concat([df_corrected, df_half_rest], axis=0)
    
    df_correlations = df_correlations.set_index("cv_fold")
    return df_corrected, df_correlations


###############################################################################

df_corrected, df_correlations = calculate_correlations(df, n_folds, target)

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
            f"{confound_status}",
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
        df_corrected.to_csv(file_path, sep=',', index=True)
        
    elif subfolder == "corrected_correlations":
        df_correlations.to_csv(file_path, sep=',', index=True)

print("===== Done! =====")
embed(globals(), locals())