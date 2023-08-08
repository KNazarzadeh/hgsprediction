#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""
import pandas as pd
import numpy as np
####### Parse Input #######
from hgsprediction.input_arguments import parse_args, input_arguments
####### Load Train set #######
# Load Processed Train set (after data validation, feature engineering)
from hgsprediction.load_data.load_healthy import load_preprocessed_train_df
####### Data Extraction #######
from hgsprediction.data_extraction import data_extractor, run_data_extraction
####### Features Extraction #######
from hgsprediction.extract_features import features_extractor
########################### Target Extraction ###########################
from hgsprediction.extract_target import target_extractor
# Calculation of Heuristic C for Linear SVM model
from hgsprediction.LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc
# Save Python objects in pickle structure
import pickle
####### Julearn #######
from julearn import run_cross_validation
####### sklearn libraries #######
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
# IMPORT SAVE FUNCTIONS
import pickle
from hgsprediction.save_results import save_extracted_nonmri_data, \
                                       save_best_model_trained, \
                                       save_scores_trained
#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, gender, model_name, \
    confound_status, cv_repeats_number, cv_folds_number = input_arguments(args)

###############################################################################
df_train = load_preprocessed_train_df(population, mri_status)
print("===== Done! =====")
embed(globals(), locals())
data_extracted = run_data_extraction.data_extractor(df_train, mri_status, gender, feature_type, target)

X = features_extractor(data_extracted, mri_status, feature_type)
y = target_extractor(data_extracted, target)

###############################################################################
# Define model and model parameters:
if model_name == "linear_svm":
    model = svrhc(dual=False, loss='squared_epsilon_insensitive')
elif model_name == "random_forest":
    model = "rf"

###############################################################################
# When cv=None, it define as follows:
cv = RepeatedKFold(n_splits=cv_folds_number, n_repeats=cv_repeats_number, random_state=47)

###############################################################################
# run run_cross_validation
if confound_status == 0:
    scores_trained, model_trained = run_cross_validation(
        X=X, y=y, data=data_extracted, cv=cv, seed=47,
        preprocess_X='zscore', problem_type='regression',
        model=model,
        return_estimator='all', scoring='r2'
    )
###############################################################################
df_estimators = scores_trained.set_index(
    ['repeat', 'fold'])['estimator'].unstack()
df_estimators.index.name = 'Repeats'
df_estimators.columns.name = 'K-fold splits'

print(df_estimators)
###############################################################################
df_test_score = scores_trained.set_index(
    ['repeat', 'fold'])['test_score'].unstack()
df_test_score.index.name = 'Repeats'
df_test_score.columns.name = 'K-fold splits'

print(df_test_score)
###############################################################################
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
# SAVE THE RESULTS
###############################################################################
save_extracted_nonmri_data(
    data_extracted,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target)
###############################################################################
save_best_model_trained(
    model_trained,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    cv_repeats_number,
    cv_folds_number)

################################################################################
save_scores_trained(
    scores_trained,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    cv_repeats_number,
    cv_folds_number)
################################################################################
print("===== Done! =====")
embed(globals(), locals())