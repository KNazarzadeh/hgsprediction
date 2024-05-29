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
####### Parse Input #######
from hgsprediction.input_arguments import parse_args, input_arguments
####### Load Train set #######
# Load Processed Train set (after data validation, feature engineering)
from hgsprediction.load_data.healthy import load_healthy_data
####### Load results #######
from hgsprediction.save_results.healthy import save_trained_model_results
####### Features Extraction #######
from hgsprediction.define_features import define_features
# Calculation of Heuristic C for Linear SVM model
from hgsprediction.LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc
# Save Python objects in pickle structure
import pickle
####### Julearn #######
from julearn import run_cross_validation
from julearn.scoring import register_scorer
####### sklearn libraries #######
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from scipy.stats import pearsonr

#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
args = parse_args()
motor, population, mri_status, feature_type, target, model_name, \
    confound_status, cv_repeats_number, cv_folds_number, data_set, gender = input_arguments(args)

session="0"

###############################################################################
def pearson_corr(y_true, y_pred):
    """ Creates a Julearn-compatible pearson correlation scorer
    for cross-validation of the regression results.

    Parameters
    -----------
    y_true: array of float values
        A vector of true regression targets
    y_pred: array of float values with the same length of y_true
        A vector of predicted regression targets 

    Returns
    --------
    r: float
        pearson correlation between true and predicted y values

    """
    r = pearsonr(y_true, y_pred)[0]
    return r

pearson_scorer = make_scorer(pearson_corr)
register_scorer("pearson_corr", pearson_scorer)

###############################################################################
data_extracted = load_healthy_data.load_extracted_data_by_feature_and_target(
    population,
    mri_status,
    feature_type,
    target,
    session,
    gender,
    data_set,
)

features, extend_features = define_features(feature_type)

X = features
y = target
print(data_extracted)
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Define model and model parameters:
if model_name == "linear_svm":
    model = svrhc(dual=False, loss='squared_epsilon_insensitive')
elif model_name == "random_forest":
    model = "rf"
# print("===== Done! =====")
# embed(globals(), locals())
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
        return_estimator='all', scoring=['r2', 'pearson_corr']
    )
###############################################################################
df_estimators = scores_trained.set_index(
    ['repeat', 'fold'])['estimator'].unstack()
df_estimators.index.name = 'Repeats'
df_estimators.columns.name = 'K-fold splits'

print(df_estimators)
###############################################################################
df_test_r2_score = scores_trained.set_index(
    ['repeat', 'fold'])['test_r2'].unstack()
df_test_r2_score.index.name = 'Repeats'
df_test_r2_score.columns.name = 'K-fold splits'

print(df_test_r2_score)

df_test_pearson_r_score = scores_trained.set_index(
    ['repeat', 'fold'])['test_pearson_corr'].unstack()
df_test_pearson_r_score.index.name = 'Repeats'
df_test_pearson_r_score.columns.name = 'K-fold splits'

print(df_test_pearson_r_score)
###############################################################################
df_prediction_r2_scores = pd.DataFrame()
df_prediction_pearson_scores = pd.DataFrame()

# Define dataframe as the result of dataframe of list of dataframes 
df_validation_prediction_hgs = pd.DataFrame()
# Define list for list of dataframes
# list_of_dfs = []
for idx, (train_val_index, validation_index) \
        in enumerate(cv.split(data_extracted)):
    repeat = scores_trained['repeat'][idx]
    fold = scores_trained['fold'][idx]
    estimator = scores_trained['estimator'][idx]
    y_pred = pd.Series(
        estimator.predict(data_extracted.iloc[validation_index][X]), name=f"{target}_predicted")
    y_true = data_extracted.iloc[validation_index][y]
    # use 'r2_score' scoring
    r2score = r2_score(y_true, y_pred)
    pearson_score = pearsonr(y_true, y_pred)[0]
    df_prediction_r2_scores.loc[f'{repeat}', f'{fold}'] = r2score
    df_prediction_pearson_scores.loc[f'{repeat}', f'{fold}'] = pearson_score
    # Create a temporary DataFrame by selecting rows from 'data_extracted' using 'validation_index',
    # and add a new column 'hgs_pred' with values from 'y_pred'.
    df_tmp = data_extracted.iloc[validation_index].copy()
    df_tmp.loc[:, f"{target}_predicted"] = y_pred.values
    # df_tmp = data_extracted.iloc[validation_index].assign(hgs_pred=y_pred.values)
    df_tmp.loc[:, "cv_fold"] = fold
    df_tmp.loc[:, "cv_repeat"] = repeat
    df_tmp.loc[:, f"{target}_delta(true-predicted)"] =  df_tmp.loc[:, f"{target}"] - df_tmp.loc[:, f"{target}_predicted"]

    df_validation_prediction_hgs = pd.concat([df_validation_prediction_hgs,df_tmp], axis=0)
df_prediction_r2_scores.index.name = 'Repeats'
df_prediction_r2_scores.columns.name = 'K-fold splits'
df_prediction_pearson_scores.index.name = 'Repeats'
df_prediction_pearson_scores.columns.name = 'K-fold splits'
# For access to each dataframe use the following code:
# for example --> df_header1 = df_validation_prediction_hgs.xs('repeat:Repeat 0 - k-fold:Fold 0')
print(df_prediction_r2_scores)
print(df_prediction_pearson_scores)

###############################################################################
# SAVE THE RESULTS
###############################################################################
save_trained_model_results.save_best_model_trained(
    model_trained,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    cv_repeats_number,
    cv_folds_number,
    session,
    data_set,)
###############################################################################      
save_trained_model_results.save_estimators_trained(
    df_estimators,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    cv_repeats_number,
    cv_folds_number,
    session,
    data_set,)
################################################################################
save_trained_model_results.save_scores_trained(
    scores_trained,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    cv_repeats_number,
    cv_folds_number,
    session,
    data_set,)
################################################################################
save_trained_model_results.save_test_scores_trained(
    df_test_r2_score,
    df_test_pearson_r_score,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    cv_repeats_number,
    cv_folds_number,
    session,
    data_set,)
################################################################################
save_trained_model_results.save_prediction_hgs_on_validation_set(
    df_validation_prediction_hgs,
    population,
    mri_status,
    confound_status,
    gender,
    feature_type,
    target,
    model_name,
    cv_repeats_number,
    cv_folds_number,
    session,
    data_set,)
################################################################################
print("===== END Done! =====")
embed(globals(), locals())