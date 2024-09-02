import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results.healthy import load_trained_model_results
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.load_data.disorder import load_disorder_data
from hgsprediction.save_results.disorder.save_disorder_hgs_predicted_results import save_disorder_hgs_predicted_results

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
disorder_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
model_name = sys.argv[7]
confound_status = sys.argv[8]
n_repeats = sys.argv[9]
n_folds = sys.argv[10]
gender = sys.argv[11]
first_event = sys.argv[12]
###############################################################################
best_model_trained = load_trained_model_results.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                gender,
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
                                "0",
                                "training_set",
                            )

print(best_model_trained)
print("gender is :", gender)
##############################################################################
# Define main features and extra features:
features, extend_features = define_features(feature_type)
##############################################################################
# Define X as main features and y as target:
X = features
y = target
##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
# Load the extracted data based on various parameters
df_extracted = load_disorder_data.load_extracted_data_by_feature_and_target(
        population,
        mri_status,
        session_column,
        feature_type,
        target,
        gender,
        first_event,
    )
##############################################################################
# Iterate over the disorder subgroups
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    # Set the prefix for columns based on visit session    
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"
    # --------------------------------------- #
    # Select columns that contain the prefix    
    df_tmp = df_extracted.loc[:, [col for col in df_extracted.columns if prefix in col]]
    # --------------------------------------- #
    # Find columns that contain any of the strings in features or the target
    df_features = df_tmp.loc[:, [col for col in df_tmp.columns if any(item in col for item in features + [target])]]    
    # --------------------------------------- #
    # Remove prefix from column names
    df_features.rename(columns=lambda x: x.replace(prefix, ''), inplace=True)
    # --------------------------------------- #    
    # Predict using the trained model    
    df_predicted = predict_hgs(df_features, X, y, best_model_trained, target)
    # --------------------------------------- #
    # Add the prefix back to the column names in the predicted DataFrame
    df_predicted = df_predicted.rename(columns=lambda col: prefix + col)
    # --------------------------------------- #
    # Store the predicted DataFrame in the appropriate variable
    if disorder_subgroup == f"pre-{population}":
        print(disorder_subgroup)
        df_pre = df_predicted.copy()
    elif disorder_subgroup == f"post-{population}":
        print(disorder_subgroup)
        df_post = df_predicted.copy()
##############################################################################
# Reindex df_post to match the index order of df_pre
df_post = df_post.reindex(df_pre.index)
# Check if the indices of both dataframes are the same and in the same order
indices_are_same_and_in_same_order = df_pre.index.equals(df_post.index)
print("Indices are the same and in the same order:", indices_are_same_and_in_same_order)
##############################################################################
# Merge the pre and post DataFrames on the index
df_merged_pre_post = pd.merge(df_pre, df_post, left_index=True, right_index=True, how='inner')
##############################################################################
# Find columns that are not in the pre or post DataFrames
df_remaining_columns = df_extracted.loc[:, [col for col in df_extracted.columns if col not in df_pre and col not in df_post]]
##############################################################################
# Merge all DataFrames on the index without duplicating columns
df = pd.merge(df_merged_pre_post, df_remaining_columns, left_index=True, right_index=True, how='inner')

print(df)
# print("===== END Done! =====")
# embed(globals(), locals())

##############################################################################
save_disorder_hgs_predicted_results(
    df,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
    first_event,
)
print("===== END Done! =====")
embed(globals(), locals())
