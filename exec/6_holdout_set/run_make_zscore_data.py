import math
import sys
import numpy as np
import pandas as pd
import os
from scipy.stats import zscore

from hgsprediction.define_features import define_features
from hgsprediction.load_results.load_corrected_prediction_results import load_corrected_prediction_results
from hgsprediction.save_results.save_zscore_results import save_zscore_results


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
gender = sys.argv[10]
###############################################################################
folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",  
        "results_hgsprediction",
        f"{population}",
        "nonmri_test_holdout_set",
        f"{feature_type}",
        f"{target}",
        f"{model_name}",
        "hgs_corrected_prediction_results",
    )

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_hgs_predicted_results.csv")

df = pd.read_csv(file_path, sep=',', index_col=0)

print("===== Done! =====")
embed(globals(), locals())
features, extend_features = define_features(feature_type)

# Define feature columns including the target
feature_columns = features + [target]

# Set the threshold for outlier detection
threshold = 3

# Calculate z-scores for the selected features
df_z_scores = zscore(df.loc[:, feature_columns])

# Identify outliers based on z-scores exceeding the threshold
outliers = (df_z_scores > threshold) | (df_z_scores < -threshold)
# Remove outliers
df_no_outliers = df_z_scores[~outliers.any(axis=1)]
df_outliers = df_z_scores[outliers.any(axis=1)]

df = df[df.index.isin(df_no_outliers.index)]

print("===== Done! End =====")
embed(globals(), locals())
folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",  
        "results_hgsprediction",
        f"{population}",
        "nonmri_test_holdout_set",
        f"{feature_type}",
        f"{target}",
        f"{model_name}",
        "zscore_results",
    )

if(not os.path.isdir(folder_path)):
    os.makedirs(folder_path)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"{gender}_zscore_data.csv")

df.to_csv(file_path, sep=',', index=True)

print("===== Done! End =====")
embed(globals(), locals())