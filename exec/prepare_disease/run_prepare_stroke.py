


import pandas as pd
import numpy as np
import os
from hgsprediction.load_data import load_original_data
from hgsprediction.input_arguments import parse_args
from hgsprediction.prepare_data.prepare_disease import PrepareDisease

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse the input arguments by function parse_args.
args = parse_args()
# Define the following parameters to run the code:
# motor, population name, mri status and confound removal status, 
# Type of feature, target and gender 
# Model, the number of repeats and folds for run_cross_validation
# motor is hgs or handgrip_strength(str)
motor = args.motor
# populations are: healthy, stroke or parkinson(str)
population = args.population
# MRI status: mri or nonmri(str)
mri_status = args.mri_status
# Type of features:(str)
# cognitive, cognitive+gnder
# bodysize, bodysize+gender
# bodysize+cognitive, bodysize+cognitive+gender
feature_type = args.feature_type
# Target(str): L+R(for HGS(Left +Right)), dominant_hgs or nondominant_hgs
target = args.target
# Type of genders: both (female+male), female and male
gender = args.gender
# Type of models(str): linear_svm, random forest(rf)
model = args.model
# 0 means without confound removal(int)
# 1 means with confound removal(int)
confound_status = args.confound_status
# Number of repeats for run_cross_validation(int)
n_repeats = args.repeat_number
# Number of folds for run_cross_validation(int)
n_folds = args.fold_number
###############################################################################
# Print summary of all inputs
print("================== Inputs ==================")
# print Motor type
if motor == "hgs":
    print("Motor = handgrip strength")
else:
    print("Motor =", motor)
# print Population type
print("Population =", population)
# print MRI status
print("MRI status =", mri_status)
# print Feature type
print("Feature type =", feature_type)
# print Target type
print("Target =", target)
# print Gender type
if gender == "both":
    print("Gender = both genders")
else:
    print("Gender =", gender)
# print Model type
if model == "rf":
    print("Model = random forest")
else:
    print("Model =", model)
# print Confound status 
if confound_status == 0:
    print("Confound status = Without Confound Removal")
else:
    print("Confound status = With Confound Removal")
# print Number of repeats for run_cross_validation
print("Repeat Numbers =", n_repeats)
# print Number of folds for run_cross_validation
print("Fold Numbers = ", n_folds)
print("============================================")

###############################################################################

data_original = load_original_data(motor, population, mri_status)

###############################################################################

prepare_data = PrepareDisease(data_original)
df_available_disease_dates = prepare_data.remove_missing_disease_dates(data_original, population)
df_available_hgs = prepare_data.remove_missing_hgs(df_available_disease_dates)
df_followup_days = prepare_data.define_followup_days(df_available_hgs, population)
pre_disease_df = prepare_data.extract_pre_disease(df_followup_days)
post_disease_df = prepare_data.extract_post_disease(df_followup_days)
longitudinal_df = prepare_data.extract_longitudinal_disease(df_followup_days)


# initialize data length of lists.
index_list = ['original_data_both_gender',
              'original_data_female',
              'original_data_male',
              'available_disease_dates_both_gender',
              'available_disease_dates_both_gender',
              'available_disease_dates_both_gender',
              'available_hgs_both_gender',
              'available_hgs_both_gender',
              'available_hgs_both_gender',
              'pre_disease_both_gender',
              'pre_disease_female',
              'pre_disease_male',
              'post_disease_both_gender',
              'post_disease_female',
              'post_disease_male',
              'longitudinal_disease_both_gender',
              'longitudinal_disease_female',
              'longitudinal_disease_male',
                ]

# summary_data = pd.DataFrame(columns=['length_of_data'], index=index_list)

# summary_data.loc[size]['train_set'] = df['train_set'][0]
# df_train_test_length.loc[size]['female_train_set'] = df['female_train_set'][0]
# df_train_test_length.loc[size]['male_train_set'] = df['male_train_set'][0]
# summary_data = pd.DataFrame({'original_data_both_gender': [len(data_original)],
#                    'available_disease_dates_both_gender': [len(df_available_disease_dates)],
#                    'available_hgs_both_gender': [len(df_available_hgs)],
#                    'pre_disease_both_gender': [len(pre_disease_df)],
#                    'pre_disease_female': [len(pre_disease_df)],
#                    'pre_disease_male': [len(pre_disease_df)],
#                    'post_disease_both_gender': [len(post_disease_df)],
#                    'post_disease_female': [len(post_disease_df)],
#                    'post_disease_male': [len(post_disease_df)],
#                    'longitudinal_disease_both_gender': [len(longitudinal_df)],
#                    'longitudinal_disease_female': [len(longitudinal_df)],
#                    'longitudinal_disease_male': [len(longitudinal_df)]
                #    })



print("===== Done! =====")
embed(globals(), locals())