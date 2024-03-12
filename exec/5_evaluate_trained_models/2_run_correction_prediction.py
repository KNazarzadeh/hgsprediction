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
from hgsprediction.prediction_corrector_model import prediction_corrector_model
from hgsprediction.load_results.healthy.load_hgs_predicted_results import load_hgs_predicted_results
from hgsprediction.save_results.healthy.save_corrected_prediction_results import save_corrected_prediction_results
from hgsprediction.save_results.healthy.save_corrected_prediction_correlation_results import save_corrected_prediction_correlation_results

from scipy.stats import pearsonr
from sklearn.metrics import r2_score

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
slope, intercept = prediction_corrector_model(
    population,
    "nonmri",
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
)

print(slope)
print(intercept)
###############################################################################

df = load_hgs_predicted_results(
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
)

###############################################################################

#Beheshti Method:
df.loc[:, f"{target}_corrected_predicted"] = (df.loc[:, f"{target}_predicted"] + ((slope * df.loc[:, f"{target}"]) + intercept))
# de Lange Method:
# df.loc[:, f"{target}_corrected_predicted"] = df.loc[:, f"{target}_predicted"] + (df.loc[:, f"{target}"] - ((slope * df.loc[:, f"{target}"]) + intercept))
# Cole Mathod:
# df.loc[:, f"{target}_corrected_predicted"] = (df.loc[:, f"{target}_predicted"] - intercept) / slope
# Calculate Corrected Delta
df.loc[:, f"{target}_corrected_delta(true-predicted)"] =  df.loc[:, f"{target}"] - df.loc[:, f"{target}_corrected_predicted"]

save_corrected_prediction_results(
    df,
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
)
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
df_correlations = pd.DataFrame()
df_p_values = pd.DataFrame()
df_r2_values = pd.DataFrame()

df_correlations.loc[0, "r_values_true_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])[0]
df_correlations.loc[0, "r_values_true_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])[0]
df_correlations.loc[0, "r_values_true_corrected_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])[0]
df_correlations.loc[0, "r_values_true_corrected_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])[0]

df_p_values.loc[0, "p_values_true_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])[1]
df_p_values.loc[0, "p_values_true_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])[1]
df_p_values.loc[0, "p_values_true_corrected_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])[1]
df_p_values.loc[0, "p_values_true_corrected_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])[1]


df_r2_values.loc[0, "r2_values_true_predicted"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])
df_r2_values.loc[0, "r2_values_true_delta"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])
df_r2_values.loc[0, "r2_values_true_corrected_predicted"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])
df_r2_values.loc[0, "r2_values_true_corrected_delta"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])



save_corrected_prediction_correlation_results(
    df_correlations,
    df_p_values,
    df_r2_values,
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
)
print("===== Done! End =====")
embed(globals(), locals())