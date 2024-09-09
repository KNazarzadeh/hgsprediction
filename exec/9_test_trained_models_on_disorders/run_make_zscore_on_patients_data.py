import os
import pandas as pd
import numpy as np
import sys
from scipy.stats import zscore
from hgsprediction.define_features import define_features
from hgsprediction.load_data.disorder import load_disorder_data

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

##############################################################################
# Define main features and extra features:
features, extend_features = define_features(feature_type)
##############################################################################
# Define feature columns including the target
feature_columns = features + [target]
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
# print("===== Done! End =====")
# embed(globals(), locals())
##############################################################################
threshold = 2.3
outliers_index = []
# Iterate over the disorder subgroups
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    # Set the prefix for columns based on visit session    
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"
    # --------------------------------------- #
    # Add prefix to feature columns:
    features_items = [prefix + item for item in feature_columns]
    # Calculate z-scores for the selected features
    df_z_scores = zscore(df_extracted.loc[:, features_items])

    # Identify outliers based on z-scores exceeding the threshold
    outliers = (df_z_scores > threshold) | (df_z_scores < -threshold)
    # Remove outliers
    df_no_outliers = df_z_scores[~outliers.any(axis=1)]
    df_outliers = df_z_scores[outliers.any(axis=1)]

    outliers_index.extend(df_outliers.index.to_list())
    # # --------------------------------------- #
    print(outliers_index)

df = df_extracted[~df_extracted.index.isin(outliers_index)]
print(df)

print("===== Done! End =====")
embed(globals(), locals())