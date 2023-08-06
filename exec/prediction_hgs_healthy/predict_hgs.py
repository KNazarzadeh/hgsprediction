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
import os
####### Load Train set #######
from hgsprediction.input_arguments import parse_args, input_arguments
# Load Primary Train set (after binning and splitting to Train and test)
from hgsprediction.load_data.load_healthy import load_primary_train_set_df
# Load Processed Train set (after data validation, feature engineering)
from hgsprediction.load_data.load_healthy import load_preprocessed_train_df
####### Prepocessing data #######
from hgsprediction.data_preprocessing import run_healthy_preprocessing, DataPreprocessor
from hgsprediction.compute_target import compute_target
from hgsprediction.data_extraction import data_extractor, run_data_extraction
from hgsprediction.extract_features import features_extractor
from hgsprediction.extract_target import target_extractor

from LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc
import pickle

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from julearn import run_cross_validation


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, gender, model_name, \
    confound_status, cv_repeats_number, cv_folds_number = input_arguments(args)

###############################################################################
# Read CSV file from Juseless
# df_train = load_primary_train_set_df(population,gender,mri_status)
###############################################################################
df_train = load_preprocessed_train_df(population, mri_status)

data = compute_target(df_train, mri_status, target)

data_extracted = run_data_extraction.data_extractor(data, mri_status, gender, feature_type, target)

X = features_extractor(data_extracted, mri_status, feature_type)
y = target_extractor(data, target)

###############################################################################
# Define model and model parameters:
if model_name == "linear_svm":
    model = svrhc(dual=False, loss='squared_epsilon_insensitive')

###############################################################################
# When cv=None, it define as follows:
cv = RepeatedKFold(n_splits=cv_folds_number, n_repeats=cv_repeats_number, random_state=47)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# run run_cross_validation
if confound_status == 0:
    scores_trained, model_trained = run_cross_validation(
        X=X, y=y, data=data_extracted, cv=cv, seed=47,
        preprocess_X='zscore', problem_type='regression',
        model=model,
        return_estimator='all', scoring='r2'
    )

df_estimators = scores_trained.set_index(
    ['repeat', 'fold'])['estimator'].unstack()
df_estimators.index.name = 'Repeats'
df_estimators.columns.name = 'K-fold splits'

print(df_estimators)


df_test_score = scores_trained.set_index(
    ['repeat', 'fold'])['test_score'].unstack()
df_test_score.index.name = 'Repeats'
df_test_score.columns.name = 'K-fold splits'

print(df_test_score)

df_prediction_scores = pd.DataFrame()
df_dict = {}
for idx, (train_val_index, validation_index) \
        in enumerate(cv.split(data_extracted)):
    repeat = scores_trained['repeat'][idx]
    fold = scores_trained['fold'][idx]
    estimator = scores_trained['estimator'][idx]
    y_pred = pd.Series(
        estimator.predict(data_extracted.iloc[validation_index][X]), name='y_pred')
    y_true = data_extracted.iloc[validation_index][y]
    # use 'r2_score' scoring
    score = r2_score(y_true, y_pred)
    df_prediction_scores.loc[f'{repeat}', f'{fold}'] = score
    df_tmp = data_extracted.iloc[validation_index].assign(hgs_pred=y_pred.values)
    df_dict[idx] = df_tmp
df_prediction_scores.index.name = 'Repeats'
df_prediction_scores.columns.name = 'K-fold splits'

print(df_prediction_scores)

###############################################################################
# save_model_trained(
#     model_trained,
#     model_trained_female,
#     model_trained_male,
#     population,
#     mri_status,
#     confound_status,
#     gender,
#     feature_type,
#     target,
#     model_name,
#     cv_repeats_number,
#     cv_folds_number,
# )
# print("===== Done! =====")
# embed(globals(), locals())
# ###############################################################################
# # save estimator in a csv file:
# output_type = "run_CV_scores_trained"
# save_prediction_scores_results(scores_trained,
#     population,
#     mri_status,
#     confound_status,
#     gender,
#     feature_type,
#     target,
#     model_name,
#     cv_repeats_number,
#     cv_folds_number,
#     output_type,)

# # save estimator in a csv file:
# # output_type = "run_CV_model_trained"
# # save_prediction_models_results(model_trained,
# #     population,
# #     mri_status,
# #     confound_status,
# #     gender,
# #     feature_type,
# #     target,
# #     model_name,
# #     cv_repeats_number,
# #     cv_folds_number,
# #     output_type,)

# ###############################################################################
# # Prediction on validation sets from CV
# # Extract cv object
# # Now we can get the estimator per fold and repetition:
# df_estimators = scores_trained.set_index(
#     ['repeat', 'fold'])['estimator'].unstack()
# df_estimators.index.name = 'Repeats'
# df_estimators.columns.name = 'K-fold splits'

# print(df_estimators)

# # save estimator in a csv file:
# output_type = "estimators"
# save_prediction_models_results(df_estimators,
#     population,
#     mri_status,
#     confound_status,
#     gender,
#     feature_type,
#     target,
#     model_name,
#     cv_repeats_number,
#     cv_folds_number,
#     output_type,)

# ###############################################################################
# # Now we can get the test_score per fold and repetition:
# df_test_score = scores_trained.set_index(
#     ['repeat', 'fold'])['test_score'].unstack()
# df_test_score.index.name = 'Repeats'
# df_test_score.columns.name = 'K-fold splits'

# print(df_test_score)

# # save test_scores in a csv file:
# output_type = "julearn"
# save_prediction_scores_results(df_test_score,
#     population,
#     mri_status,
#     confound_status,
#     gender,
#     feature_type,
#     target,
#     model_name,
#     cv_repeats_number,
#     cv_folds_number,
#     output_type,)
# ###############################################################################
# # Predict each estimator on validation set per fold:
# # To access train and validation sets, using split function on CV object.
# # The result must be the same as the 'test_score' of run_cross_validation.
# df_prediction_scores = pd.DataFrame()
# df_dict = {}
# for idx, (train_val_index, validation_index) \
#         in enumerate(cv.split(data_extracted)):
#     repeat = scores_trained['repeat'][idx]
#     fold = scores_trained['fold'][idx]
#     estimator = scores_trained['estimator'][idx]
#     y_pred = pd.Series(
#         estimator.predict(data_extracted.iloc[validation_index][X]), name='y_pred')
#     y_true = data_extracted.iloc[validation_index][y]
#     # use 'r2_score' scoring
#     score = r2_score(y_true, y_pred)
#     df_prediction_scores.loc[f'{repeat}', f'{fold}'] = score
#     df_tmp = data_extracted.iloc[validation_index].assign(hgs_pred=y_pred.values)
#     df_dict[idx] = df_tmp
# df_prediction_scores.index.name = 'Repeats'
# df_prediction_scores.columns.name = 'K-fold splits'

# print(df_prediction_scores)
# # save test_scores in a csv file:
# output_type = "both_gender_validation_ypred"
# for idx in df_dict:
#     save_validation_ypred_results(df_dict[idx],
#         population,
#         mri_status,
#         confound_status,
#         gender,
#         feature_type,
#         target,
#         model_name,
#         cv_repeats_number,
#         cv_folds_number,
#         output_type,
#         idx)

# # save test_scores in a csv file:
# output_type = "cv_object"
# save_prediction_scores_results(df_prediction_scores,
#     population,
#     mri_status,
#     confound_status,
#     gender,
#     feature_type,
#     target,
#     model_name,
#     cv_repeats_number,
#     cv_folds_number,
#     output_type,)

# ###############################################################################
# print("===== Done! =====")
# embed(globals(), locals())