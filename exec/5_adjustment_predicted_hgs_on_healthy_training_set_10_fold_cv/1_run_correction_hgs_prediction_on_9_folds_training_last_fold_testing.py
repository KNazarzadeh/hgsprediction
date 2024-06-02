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
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from hgsprediction.load_results.healthy import load_trained_model_results
from hgsprediction.correction_predicted_hgs import beheshti_correction_method
from hgsprediction.save_results.healthy.save_corrected_prediction_results import save_corrected_prediction_results

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
correlation_type = sys.argv[11]
gender = sys.argv[12]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
if correlation_type == "pearson":
    correlation_func = pearsonr
elif correlation_type == "spearman":
    correlation_func = spearmanr
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
model = LinearRegression()
###############################################################################
df_train = df[df['cv_fold'] != int(n_folds)-1]
# Beheshti Method:
X = df_train.loc[:, f"{target}"].values.reshape(-1, 1)
# y = df.loc[:, f"{target}_delta(true-predicted)"].values
y = df_train.loc[:, f"{target}_delta(true-predicted)"].values

model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_  
###############################################################################     
df_test = df[df['cv_fold'] == int(n_folds)-1]
        
# Beheshti Method:
true_hgs = df_test.loc[:, f"{target}"]
predicted_hgs = df_test.loc[:, f"{target}_predicted"]
df_corrected_hgs = beheshti_correction_method(df_test.copy(), target, true_hgs, predicted_hgs, slope, intercept)


print("===== Done! =====")
embed(globals(), locals())
###############################################################################
save_corrected_prediction_results(
    df_corrected_hgs,
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
    data_set,
)

print("===== Done! =====")
embed(globals(), locals())