import math
import sys
import numpy as np
import pandas as pd
import os
from scipy.stats import zscore

from hgsprediction.define_features import define_features
from hgsprediction.load_results.load_corrected_prediction_results import load_corrected_prediction_results
from hgsprediction.save_results.healthy.save_zscore_results import save_zscore_results


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
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
gender = sys.argv[9]
# session = sys.argv[10]

###############################################################################
features, extend_features = define_features(feature_type)

# Define feature columns including the target
feature_columns = features + [target]
# feature_columns = features

###############################################################################
for session in ["0", "1", "2", "3"]:
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

    # Set the threshold for outlier detection
    threshold = 3.2

    # Calculate z-scores for the selected features
    df_z_scores = zscore(df.loc[:, feature_columns])

    # Identify outliers based on z-scores exceeding the threshold
    outliers = (df_z_scores > threshold) | (df_z_scores < -threshold)
    # Remove outliers
    df_no_outliers = df_z_scores[~outliers.any(axis=1)]
    df_outliers = df_z_scores[outliers.any(axis=1)]

    df = df[df.index.isin(df_no_outliers.index)]

    print(df)

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

print("===== Done! End =====")
embed(globals(), locals())