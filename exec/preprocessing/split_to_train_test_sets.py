#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""

import sys
import pandas as pd
import os
import numpy as np
from hgsprediction.input_arguments import parse_args
from hgsprediction.preprocess import PreprocessData
from hgsprediction.load_data.load_data import load_data_per_session
from hgsprediction.preprocess.preprocess_healthy import check_hgs_availability_per_session
from hgsprediction.features_extraction import ExtractFeatures

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
# Define motor, population and mri status to run the code:
motor = args.motor
population = args.population
mri_status = args.mri_status
feature_type = args.feature_type
target = args.target
gender = args.gender
model = args.model
confound_status = args.confound_status
n_repeats = args.repeat_number
n_folds = args.fold_number

# Print all input
print("================== Inputs ==================")
if motor == "hgs":
    print("Motor = handgrip strength")
else:
    print("Motor =", motor)
print("Population =", population)
print("MRI status =", mri_status)
print("Feature type =", feature_type)
print("Target =", target)
if gender == "both":
    print("Gender = both genders")
else:
    print("Gender =", gender)
if model == "rf":
    print("Model = random forest")
else:
    print("Model =", model)
if confound_status == 0:
    print("Confound_status = Without Confound Removal")
else:
    print("Confound_status = With Confound Removal")

print("Repeat Numbers =", n_repeats)
print("Fold Numbers = ", n_folds)

print("============================================")

###############################################################################
# Read CSV file from Juseless
data_loaded = load_data_per_session(motor, population, mri_status, session=0)

data_hgs = check_hgs_availability_per_session(data_loaded, session=0)

extract_features = ExtractFeatures(data_hgs, motor, population)
extracted_data = extract_features.extract_features()

###############################################################################
# Add new culomns to data
data = extracted_data.copy()
add_new_cols = PreprocessData(data, session=0)
# data = add_new_cols.dominant_handgrip(data)
data = add_new_cols.validate_handgrips(data)
# Preprocess data
data = add_new_cols.preprocess_behaviours(data)
data = add_new_cols.calculate_qualification(data)
data = add_new_cols.calculate_waist_to_hip_ratio(data)
data = add_new_cols.sum_handgrips(data)
data = add_new_cols.calculate_neuroticism_score(data)
data = add_new_cols.calculate_anxiety_score(data)
data = add_new_cols.calculate_depression_score(data)
data = add_new_cols.calculate_cidi_score(data)

###############################################################################

nonmri_healthy = data.copy()

###############################################################################
# bin data based on age and hgs bins:
if (feature_type == "bodysize") | (feature_type == "bodysize+gender"):
    subs_id = pd.read_csv("nonmri_cognitive.csv", sep=',', index_col=False)
    nonmri_healthy = nonmri_healthy[nonmri_healthy.loc[:, 'eid'].isin(subs_id.loc[:, 'eid'])]

###############################################################################
binned_data = binning_data(nonmri_healthy, gender)
###############################################################################
df_train, df_test = split_train_test_sets(binned_data, gender)
if gender == "both":
    df_train_set = df_train
    df_train_female = df_train_set[df_train['31-0.0']==0]
    df_train_male = df_train_set[df_train['31-0.0']==1]
    df_test_set = df_test
    df_test_female = df_test_set[df_test['31-0.0']==0]
    df_test_male = df_test_set[df_test['31-0.0']==1]
elif gender == "female":
    df_train_female = df_train[df_train['31-0.0']==0]
    df_train_set = df_train_female
elif gender == "male":
    df_train_male = df_train[df_train['31-0.0']==1]
    df_train_set = df_train_male
# initialize data of lists.
train_test_data = {'original_data': [len(binned_data)],
                   'train_set': [len(df_train_set)],
                   'test_set': [len(df_test_set)],
                   'female_train_set': [len(df_train_female)],
                   'female_test_set': [len(df_test_female)],
                   'male_train_set': [len(df_train_male)],
                   'male_test_set': [len(df_test_male)],
                   }

# Create DataFrame
df_train_test = pd.DataFrame(train_test_data)

save_train_test_length_results(
    df_train_test,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
)


