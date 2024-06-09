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
from sklearn.linear_model import LinearRegression
from hgsprediction.load_results.healthy import load_corrected_prediction_results
from hgsprediction.load_data.brain_correlate import load_original_brain_data
from hgsprediction.load_data.brain_correlate.load_removed_tiv_from_brain_data import load_removed_tiv_from_brain_data
from hgsprediction.brain_correlate.remove_true_hgs_varaince_from_regions import remove_true_hgs_variance_from_regions
from hgsprediction.save_data.brain_correlate.save_overlap_brain_data_with_mri_data import save_overlap_brain_data_with_mri_data
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
data_set =sys.argv[10]
brain_data_type = sys.argv[11]
tiv_status = sys.argv[12]
schaefer = sys.argv[13]
gender = sys.argv[14]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

df_mri = load_corrected_prediction_results(
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

###############################################################################
df_brain = load_original_brain_data(brain_data_type, schaefer)

if tiv_status == "without_tiv":
    df_brain_without_tiv = load_removed_tiv_from_brain_data(session, brain_data_type, schaefer)
    
##############################################################################
# Remove true HGS variance from regions:
true_hgs = df_mri.loc[:, f'{target}']

df = pd.merge(df_brain_without_tiv, true_hgs, left_index=True, right_index=True, how='inner')
##############################################################################
brain_regions = df_brain.columns

df_residuals = remove_true_hgs_variance_from_regions(df, target, brain_regions)

##############################################################################
df_merged = pd.merge(df_residuals, df_mri, left_index=True, right_index=True, how='inner')
##############################################################################
save_overlap_brain_data_with_mri_data(
    df_merged,
    session,
    brain_data_type,
    schaefer,
    gender,
)

print("===== Done! =====")
embed(globals(), locals())
