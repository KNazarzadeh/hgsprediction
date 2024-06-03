import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.define_features import define_features
from hgsprediction.extract_data import disorder_extract_data
from hgsprediction.load_data.disorder import load_disorder_data
from hgsprediction.save_results.disorder.save_disorder_extracted_data_by_feature_and_target import save_disorder_extracted_data_by_feature_and_target
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
gender = sys.argv[7]
first_event = sys.argv[8]
##############################################################################
# Define main features and extra features:
features, extend_features = define_features(feature_type)
##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
if mri_status == "mri+nonmri":
    df_longitudinal_mri = load_disorder_data.load_preprocessed_data(population, "mri", session_column, disorder_cohort, first_event)
    df_longitudinal_nonmri = load_disorder_data.load_preprocessed_data(population, "nonmri", session_column, disorder_cohort, first_event)
    df_longitudinal = pd.concat([df_longitudinal_mri, df_longitudinal_nonmri]).dropna(axis=1, how='all')
else:
    df_longitudinal = load_disorder_data.load_preprocessed_data(population, mri_status, session_column, disorder_cohort, first_event)
##############################################################################
if gender == "female":
    df = df_longitudinal[df_longitudinal['gender'] == 0]
elif gender == "male":
    df = df_longitudinal[df_longitudinal['gender'] == 1]
##############################################################################
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    df_extracted = disorder_extract_data.extract_data(df, population, features, extend_features, target, disorder_subgroup, visit_session)
            
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"

    # Filter columns that require the prefix to be added
    filtered_columns = [col for col in df_extracted.columns if col in features + [target]]

    # Add the prefix to selected column names
    for col in filtered_columns:
        new_col_name = prefix + col
        df_extracted.rename(columns={col: new_col_name}, inplace=True)

    # Concatenate the DataFrames
    if disorder_subgroup == f"pre-{population}":
        print(disorder_subgroup)
        df_pre = df_extracted.copy()
        
    elif disorder_subgroup == f"post-{population}":
        print(disorder_subgroup)
        df_post = df_extracted.copy()
        # Merging DataFrames on index

# Reindex df_pre to match the index order of df_post
df_post = df_post.reindex(df_pre.index)

# Check if the indices of both dataframes are the same and in the same order
indices_are_same_and_in_same_order = df_pre.index.equals(df_post.index)
print("Indices are the same and in the same order:", indices_are_same_and_in_same_order)
# Finding common columns
common_cols = df_pre.columns.intersection(df_post.columns)
# Merge DataFrames on index without duplicating common columns
df_merged = pd.merge(df_pre.drop(columns=common_cols), df_post, left_index=True, right_index=True, how='inner')

df_merged = df_merged.drop(columns="handedness")
print(df_merged)

save_disorder_extracted_data_by_feature_and_target(
    df_merged,
    population,
    mri_status,
    session_column,
    feature_type,
    target,
    gender,
    first_event,
)
print("===== Done! =====")
embed(globals(), locals())