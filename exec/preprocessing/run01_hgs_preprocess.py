#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""

import sys
import pandas as pd
from hgsprediction.load_data import load_hgs_data_per_session
from hgsprediction.extract_features import ExtractFeatures

# from add_new_columns import PreprocessData
# from save_results import save_preprocessed_train_df


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Define motor, population and mri status to run the code:
filename = sys.argv[0]
motor = sys.argv[1] 
population = sys.argv[2]
mri_status = sys.argv[3]
if mri_status == "mri":
    session = 2
###############################################################################
df = load_hgs_data_per_session(motor, population, mri_status, session)

extract_features = ExtractFeatures(df, motor, population)
extracted_data = extract_features.extract_features()
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# # Add new culomns to data
# add_new_cols = PreprocessData(extracted_data, session)
# df_hgs = add_new_cols.validate_handgrips(extracted_data)
# print("===== Done! =====")
# embed(globals(), locals())
# # Preprocess data
# df_train_set = add_new_cols.preprocess_behaviours(df_hgs)
# df_train_set = add_new_cols.calculate_qualification(df_train_set)
# df_train_set = add_new_cols.calculate_waist_to_hip_ratio(df_train_set)
# df_train_set = add_new_cols.calculate_neuroticism_score(df_train_set)
# df_train_set = add_new_cols.calculate_anxiety_score(df_train_set)
# df_train_set = add_new_cols.calculate_depression_score(df_train_set)
# df_train_set = add_new_cols.calculate_cidi_score(df_train_set)

# save_preprocessed_train_df(df_train_set,
#                            population,
#                            mri_status)