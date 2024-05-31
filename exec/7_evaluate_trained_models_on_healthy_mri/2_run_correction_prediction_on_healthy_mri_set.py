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
from hgsprediction.correction_predicted_hgs import prediction_corrector_model
from hgsprediction.load_results.healthy.load_hgs_predicted_results import load_hgs_predicted_results
from hgsprediction.correction_predicted_hgs.correction_method import beheshti_correction_method
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
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
data_set = sys.argv[9]
gender = sys.argv[10]

###############################################################################
slope, intercept = prediction_corrector_model(
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
)

print(slope)
print(intercept)
###############################################################################
for session in ["0", "1", "2", "3"]:
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
        data_set,
    )
    print(df)
    ###############################################################################
    #Beheshti Method:
    true_hgs = df.loc[:, f"{target}"]
    predicted_hgs = df.loc[:, f"{target}_predicted"]

    df_corrected_hgs = beheshti_correction_method(
        df.copy(),
        target,
        true_hgs,
        predicted_hgs,
        slope, 
        intercept,
    )
    print(df_corrected_hgs)

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

