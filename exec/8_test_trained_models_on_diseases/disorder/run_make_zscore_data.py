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
df = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
)

features, extend_features = define_features(feature_type)

# Filter columns that start with the specified prefix
filtered_columns = [col for col in df.columns if col in features + [target]]

# Remove the prefix from selected column names
for col in filtered_columns:
    new_col_name = col.replace("1st_scan", "")
    df.rename(columns={col: new_col_name}, inplace=True)


threshold = 3

z_scores = zscore(df[[features + [target]]])
df_z_scores = pd.DataFrame(z_scores, columns=[features + [target]])
outliers = (df_z_scores > threshold) | (df_z_scores < -threshold)
# Remove outliers
df_no_outliers = df_z_scores[~outliers.any(axis=1)]
df_outliers = df_z_scores[outliers.any(axis=1)]

df = df[df.index.isin(df_no_outliers.index)]

save_zscore_results(
    df,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
)

print("===== Done! =====")
embed(globals(), locals())