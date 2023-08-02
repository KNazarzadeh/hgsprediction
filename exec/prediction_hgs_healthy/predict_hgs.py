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
from hgsprediction.input_arguments import parse_args, input_arguments
from hgsprediction.load_data.load_healthy import load_preprocessed_train_df
from hgsprediction.prediction_preparing import remove_nan_columns


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, gender, model, \
    confound_status, cv_repeats_number, cv_folds_number = input_arguments(args)

###############################################################################
# Read CSV file from Juseless
df_train = load_preprocessed_train_df(population, mri_status)

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Remove columns that all values are NaN
df_train_set = remove_nan_columns(df_train)
print("===== Done! =====")
embed(globals(), locals())
# Define features and target
X = define_features(feature_type, df_train_set)
# Target: HGS(L+R)
y = define_target(target)

print("===== Done! =====")
embed(globals(), locals())

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
    subs_id = pd.read_csv("nonmri_cognitive_dominant_hgs.csv", sep=',', index_col=False)
    nonmri_healthy = nonmri_healthy[nonmri_healthy.loc[:, 'eid'].isin(subs_id.loc[:, 'eid'])]

# print("============================ Done! ============================")
# embed(globals(), locals())
# ###############################################################################
# # Remove columns that all values are NaN
# nan_cols = nonmri_healthy.columns[nonmri_healthy.isna().all()].tolist()
# nonmri_healthy = nonmri_healthy.drop(nan_cols, axis=1)

# # Define features and target
# X = define_features(feature_type, nonmri_healthy)
# # Target: HGS(L+R)
# y = define_target(target)

# ###############################################################################

# # Remove Missing data from Features and Target
# nonmri_healthy = nonmri_healthy.dropna(subset=y)
# nonmri_healthy = nonmri_healthy.dropna(subset=X)

# if confound_status == 1:
#     confound = define_confound(confound_status)
#     nonmri_healthy = nonmri_healthy.dropna(subset=confound)

###############################################################################
binned_data = binning_data(nonmri_healthy, gender)
print("============================ Done! ============================")
embed(globals(), locals())
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
samplesize = "same_samples"
# initialize data of lists.
# train_test_data = {'original_data': [len(binned_data)],
#                    'train_set': [len(df_train_set)],
#                    'test_set': [len(df_test_set)],
#                    'female_train_set': [len(df_train_female)],
#                    'female_test_set': [len(df_test_female)],
#                    'male_train_set': [len(df_train_male)],
#                    'male_test_set': [len(df_test_male)],
#                    }

# # Create DataFrame
# df_train_test = pd.DataFrame(train_test_data)

# save_train_test_length_results(
#     df_train_test,
#     population,
#     mri_status,
#     confound_status,
#     gender,
#     feature_type,
#     target,
#     model,
#     samplesize,
# )
###############################################################################
# Remove columns that all values are NaN
nan_cols = df_train_set.columns[df_train_set.isna().all()].tolist()
df_train_set = df_train_set.drop(nan_cols, axis=1)

# Define features and target
X = define_features(feature_type, nonmri_healthy)
# Target: HGS(L+R)
y = define_target(target)

###############################################################################

# Remove Missing data from Features and Target
df_train_set = df_train_set.dropna(subset=y)
df_train_set = df_train_set.dropna(subset=X)

if confound_status == 1:
    confound = define_confound(confound_status)
    df_train_set = df_train_set.dropna(subset=confound)



print("============================ Done! ============================")
embed(globals(), locals())
###############################################################################
# Define model and model parameters:
model_name, model_params = model_parameters(model)

###############################################################################
# When cv=None, it define as follows:
cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=47)

###############################################################################
# run run_cross_validation
if confound_status == 1:
    scores_trained, model_trained = run_cross_validation(
        X=X, y=y, data=df_train_set, cv=cv, seed=47,
        confounds=confound, preprocess_X=['zscore', 'remove_confound'],
        problem_type='regression',
        model=model_name, model_params=model_params,
        return_estimator='all', scoring='r2'
    )
    scores_trained_female, model_trained_female = run_cross_validation(
        X=X, y=y, data=df_train_female, cv=cv, seed=47,
        confounds=confound, preprocess_X=['zscore', 'remove_confound'],
        problem_type='regression',
        model=model_name, model_params=model_params,
        return_estimator='all', scoring='r2'
    )
    scores_trained_male, model_trained_male = run_cross_validation(
        X=X, y=y, data=df_train_male, cv=cv, seed=47,
        confounds=confound, preprocess_X=['zscore', 'remove_confound'],
        problem_type='regression',
        model=model_name, model_params=model_params,
        return_estimator='all', scoring='r2'
    )
    
else:
    scores_trained, model_trained = run_cross_validation(
        X=X, y=y, data=df_train_set, cv=cv, seed=47,
        preprocess_X='zscore', problem_type='regression',
        model=model_name, model_params=model_params,
        return_estimator='all', scoring='r2'
    )
    scores_trained_female, model_trained_female = run_cross_validation(
        X=X, y=y, data=df_train_female, cv=cv, seed=47,
        preprocess_X='zscore', problem_type='regression',
        model=model_name, model_params=model_params,
        return_estimator='all', scoring='r2'
    )
    scores_trained_male, model_trained_male = run_cross_validation(
        X=X, y=y, data=df_train_male, cv=cv, seed=47,
        preprocess_X='zscore', problem_type='regression',
        model=model_name, model_params=model_params,
        return_estimator='all', scoring='r2'
    )

###############################################################################
# Prediction on validation sets from CV
# Extract cv object
# Now we can get the estimator per fold and repetition:
df_estimators = scores_trained.set_index(
    ['repeat', 'fold'])['estimator'].unstack()
df_estimators.index.name = 'Repeats'
df_estimators.columns.name = 'K-fold splits'

print(df_estimators)

###############################################################################
# Now we can get the test_score per fold and repetition:
df_test_score = scores_trained.set_index(
    ['repeat', 'fold'])['test_score'].unstack()
df_test_score.index.name = 'Repeats'
df_test_score.columns.name = 'K-fold splits'

print(df_test_score)

# Now we can get the test_score per fold and repetition:
df_test_score_female = scores_trained_female.set_index(
    ['repeat', 'fold'])['test_score'].unstack()
df_test_score_female.index.name = 'Repeats'
df_test_score_female.columns.name = 'K-fold splits'

print(df_test_score_female)

df_test_score_male = scores_trained_male.set_index(
    ['repeat', 'fold'])['test_score'].unstack()
df_test_score_male.index.name = 'Repeats'
df_test_score_male.columns.name = 'K-fold splits'

print(df_test_score_male)
###############################################################################
# Predict each estimator on validation set per fold:
# To access train and validation sets, using split function on CV object.
# The result must be the same as the 'test_score' of run_cross_validation.
df_prediction_scores = pd.DataFrame()

for idx, (train_val_index, validation_index) \
        in enumerate(cv.split(df_train_set)):
    repeat = scores_trained['repeat'][idx]
    fold = scores_trained['fold'][idx]
    estimator = scores_trained['estimator'][idx]
    y_pred = pd.Series(
        estimator.predict(df_train_set.iloc[validation_index][X]))
    y_true = df_train_set.iloc[validation_index][y]
    # use 'r2_score' scoring
    score = r2_score(y_true, y_pred)
    df_prediction_scores.loc[f'{repeat}', f'{fold}'] = score
df_prediction_scores.index.name = 'Repeats'
df_prediction_scores.columns.name = 'K-fold splits'

print(df_prediction_scores)


###############################################################################
# Predict each estimator on validation set per fold:
# To access train and validation sets, using split function on CV object.
# The result must be the same as the 'test_score' of run_cross_validation.
df_prediction_female_scores = pd.DataFrame()

for idx, (train_val_index, validation_index) \
        in enumerate(cv.split(df_train_female)):
    repeat = scores_trained_female['repeat'][idx]
    fold = scores_trained_female['fold'][idx]
    estimator = scores_trained_female['estimator'][idx]
    y_pred = pd.Series(
        estimator.predict(df_train_female.iloc[validation_index][X]))
    y_true = df_train_female.iloc[validation_index][y]
    # use 'r2_score' scoring
    score = r2_score(y_true, y_pred)
    df_prediction_female_scores.loc[f'{repeat}', f'{fold}'] = score
df_prediction_female_scores.index.name = 'Repeats'
df_prediction_female_scores.columns.name = 'K-fold splits'

print(df_prediction_female_scores)


df_prediction_male_scores = pd.DataFrame()

for idx, (train_val_index, validation_index) \
        in enumerate(cv.split(df_train_male)):
    repeat = scores_trained_male['repeat'][idx]
    fold = scores_trained_male['fold'][idx]
    estimator = scores_trained_male['estimator'][idx]
    y_pred = pd.Series(
        estimator.predict(df_train_male.iloc[validation_index][X]))
    y_true = df_train_male.iloc[validation_index][y]
    # use 'r2_score' scoring
    score = r2_score(y_true, y_pred)
    df_prediction_male_scores.loc[f'{repeat}', f'{fold}'] = score
df_prediction_male_scores.index.name = 'Repeats'
df_prediction_male_scores.columns.name = 'K-fold splits'

print(df_prediction_male_scores)

###############################################################################
# save test_scores in a csv file:
# output_type = "julearn"
# save_prediction_results(df_test_score,
#     population,
#     mri_status,
#     confound_status,
#     gender,
#     feature_type,
#     target,
#     model,
#     n_repeats,
#     n_folds,
#     output_type,)

# # save test_scores in a csv file:
# output_type = "cv_object"
# save_prediction_results(df_prediction_scores,
#     population,
#     mri_status,
#     confound_status,
#     gender,
#     feature_type,
#     target,
#     model,
#     n_repeats,
#     n_folds,
#     output_type,)
###############################################################################
# save test_scores in a csv file:
output_type = "julearn"
save_same_samplesize_r2_results(df_test_score,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,)

# save test_scores in a csv file:
output_type = "cv_object"
save_same_samplesize_r2_results(df_prediction_scores,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,)
###############################################################################
# save test_scores in a csv file:
output_type = "cv_object_female"
save_same_samplesize_r2_results(df_prediction_female_scores,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,)


output_type = "cv_object_male"
save_same_samplesize_r2_results(df_prediction_male_scores,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model,
    n_repeats,
    n_folds,
    output_type,)

###############################################################################
# plots
regplot_genders_together(
    df_train_set,
    scores_trained,
    X,
    y,
    population,
    mri_status,
    gender,
    feature_type,
    target,
    model,
    confound_status,
    n_repeats,
    n_folds,
    cv,
)
###############################################################################
regplot_genders_seperate(
    df_train_set,
    scores_trained,
    X,
    y,
    population,
    mri_status,
    gender,
    feature_type,
    target,
    model,
    confound_status,
    n_repeats,
    n_folds,
    cv,
)

print("============================ Done! ============================")
embed(globals(), locals())
# %%
