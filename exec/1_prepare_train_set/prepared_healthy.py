#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""

import pandas as pd
import os
import numpy as np
import sys
from arguments_input import parse_args
from load_data import load_data_per_session
from binning_data import binning_data
from split_train_test import split_train_test_sets
from preprocess import check_hgs_availability_per_session
from add_new_target_columns import PrepareData
from save_results import save_binned_df, save_train_set_df, save_test_set_df, save_data_summary

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Define motor, population and mri status to run the code:
filename = sys.argv[0]
motor = sys.argv[1]
population = sys.argv[2]
mri_status = sys.argv[3]

##############################################################################
# Read CSV file from Juseless
data_loaded = load_data_per_session(motor, population, mri_status, session=0)

data_hgs = check_hgs_availability_per_session(data_loaded, session=0)

data = data_hgs.copy()

###############################################################################
# Add new culomns to data
# data = extracted_data.copy()
add_new_cols = PrepareData(data, session=0)
data = add_new_cols.sum_handgrips(data)
data = add_new_cols.dominant_handgrip(data)
data = add_new_cols.nondominant_handgrip(data)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
binned_data = binning_data(data)
save_binned_df(
    data,
    population,
    mri_status,
)
###############################################################################
df_train, df_test = split_train_test_sets(binned_data)

save_train_set_df(
    df_train,
    population,
    mri_status,
)

save_test_set_df(
    df_test,
    population,
    mri_status,
)

###############################################################################
# Save Summary of data:
df_train_female = df_train[df_train['31-0.0']==0]
df_train_male = df_train[df_train['31-0.0']==1]
df_test_female = df_test[df_test['31-0.0']==0]
df_test_male = df_test[df_test['31-0.0']==1]

# initialize data of lists.
summary_data = {'original_data': [len(data_loaded)],
                   'hgs_available_data': [len(data_hgs)],
                   'binned_data': [len(binned_data)],
                   'train_set': [len(df_train)],
                   'test_set': [len(df_test)],
                   'female_train_set': [len(df_train_female)],
                   'female_test_set': [len(df_test_female)],
                   'male_train_set': [len(df_train_male)],
                   'male_test_set': [len(df_test_male)],
                   }

# Create DataFrame
df_summary = pd.DataFrame(summary_data)

save_data_summary(
    df_summary,
    population,
    mri_status,
)

print("============================ Done! ============================")
embed(globals(), locals())
