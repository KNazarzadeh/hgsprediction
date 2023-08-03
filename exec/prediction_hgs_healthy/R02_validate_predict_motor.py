#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""
import os
import pandas as pd
import numpy as np
import sys
from hgsprediction.features_extraction import ExtractFeatures
from define_features import define_features
from define_target import define_target
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Define motor, population and mri status to run the code:
filename = sys.argv[0]
motor = sys.argv[1]
population = sys.argv[2]
mri_status = sys.argv[3]
feature_type = "cognitive"
target = "L+R"
session = 2
##############################################################################
# Read CSV file from Juseless

folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "GIT_repositories",
        "motor_ukb",
        "data_ukb",
        f"data_{motor}",
        population,
        "preprocessed_data",
    )
    # if population is healthy, we need to load data for session 0
    # with more available data.
folder_path = os.path.join(
    folder_path,
    "hgs_availability_per_session",
    f"{mri_status}_{population}"
)
file_path = os.path.join(
    folder_path,
    f"{mri_status}_{population}_hgs_availability_session_{session}.csv")

data = pd.read_csv(file_path, sep=',')

##############################################################################
# Check the availa
hgs_left = "46"  # Handgrip_strength_(left)
hgs_right = "47"  # Handgrip_strength_(right)
df_tmp_mri = pd.DataFrame()
ses = 2
df = data
df_tmp_mri = df[
        ((~df[f'{hgs_left}-{ses}.0'].isna()) &
            (df[f'{hgs_left}-{ses}.0'] !=  0))
        & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
            (df[f'{hgs_right}-{ses}.0'] !=  0))
        ]
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# Remove columns that all values are NaN
nan_cols = df_tmp_mri.columns[df_tmp_mri.isna().all()].tolist()
df_train_set = df_tmp_mri.drop(nan_cols, axis=1)

# Define features and target
X = define_features(feature_type, df_train_set)
# Target: HGS(L+R)
y = define_target(target)
print("===== Done! =====")
embed(globals(), locals())
##############################################################################
