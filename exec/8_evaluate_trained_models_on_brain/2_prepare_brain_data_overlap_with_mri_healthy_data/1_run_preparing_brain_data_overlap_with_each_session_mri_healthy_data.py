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
from hgsprediction.load_data.brain_correlate.load_removed_tiv_from_brain_data import load_removed_tiv_from_brain_data
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
if tiv_status == "without_tiv":
    df_brain_without_tiv = load_removed_tiv_from_brain_data(brain_data_type, schaefer)

##############################################################################
df_merged = pd.merge(df_brain_without_tiv, df_mri, left_index=True, right_index=True, how='inner')

print(df_merged)
print("===== Done! =====")
embed(globals(), locals())
##############################################################################
save_overlap_brain_data_with_mri_data(
    df_merged,
    brain_data_type,
    schaefer,
    session,
    gender,
)

print("===== Done! =====")
embed(globals(), locals())
