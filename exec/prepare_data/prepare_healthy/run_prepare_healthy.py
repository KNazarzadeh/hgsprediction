#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""

import pandas as pd
import numpy as np
import sys
from hgsprediction.preprocess import check_hgs_availability
from hgsprediction.load_data import load_original_data
# from save_results import save_binned_df, save_train_set_df, save_test_set_df, save_data_summary

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Define motor, population and mri status to run the code:
filename = sys.argv[0]
motor = sys.argv[1]
population = sys.argv[2]
# mri_status = sys.argv[3]

##############################################################################
# Read CSV file from Juseless
data_original_mri = load_original_data(motor, population, mri_status='mri')
data_original_non = load_original_data(motor, population, mri_status='nonmri')

print("===== Done! =====")
embed(globals(), locals())
##############################################################################
# Check the availa
df_hgs, df_session0, df_session1, df_session2, df_session3 = check_hgs_availability(data_original_non)
hgs_left = "46"  # Handgrip_strength_(left)
hgs_right = "47"  # Handgrip_strength_(right)
df_hgs = pd.DataFrame()
ses = 0
df = df_session0
df_tmp_non = df[
        ((~df[f'{hgs_left}-{ses}.0'].isna()) &
            (df[f'{hgs_left}-{ses}.0'] !=  0))
        & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
            (df[f'{hgs_right}-{ses}.0'] !=  0))
        ]
df_hgs_mri, df_session0_mri, df_session1_mri, df_session2_mri, df_session3_mri = check_hgs_availability(data_original_mri)
ses = 2
df = df_session2_mri
df_tmp_mri = df[
        ((~df[f'{hgs_left}-{ses}.0'].isna()) &
            (df[f'{hgs_left}-{ses}.0'] !=  0))
        & ((~df[f'{hgs_right}-{ses}.0'].isna()) &
            (df[f'{hgs_right}-{ses}.0'] !=  0))
        ]
##############################################################################
# Save the data with HGS availability
