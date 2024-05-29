import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.load_data import load_healthy_data
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_pearson_hgs_correlation
from hgsprediction.save_results.healthy import save_correlation_results, \
                                               save_hgs_predicted_results

from scipy.stats import pearsonr

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
best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                gender,
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
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
# for session in ["0", "1", "2", "3"]: 
for session in ["0"]: 
    # load data
    df_original = load_healthy_data.load_preprocessed_data(population, mri_status, session, gender)
    ##############################################################################
    # Extract data based on main features, extra features, target for each session and mri status:
    data_extracted = healthy_extract_data.extract_data(df_original.copy(), features, extend_features, feature_type, target, mri_status, session)
    ##############################################################################
    # Predict Handgrip strength (HGS) on X and y in dataframe
    # With best trained model on non-MRI healthy controls data
    df = predict_hgs(data_extracted.copy(), X, y, best_model_trained, target)

    ##############################################################################
    # Print the final dataframe after adding predicted and delta HGS columns
    print(df)
    summary_stats = data_extracted.describe().apply(lambda x: round(x, 2))
    print(summary_stats)
    print("===== END Done! =====")
    embed(globals(), locals())
    ##############################################################################
    # Save dataframe in the specific location
    save_hgs_predicted_results(
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

print("===== END Done! =====")
embed(globals(), locals())

##############################################################################
y_axis = ["true", "predicted", "delta(true-predicted)"]
x_axis = ["true", "predicted"]
df_corr, df_pvalue = calculate_pearson_hgs_correlation(df, y_axis, x_axis)
print(df_corr)

save_correlation_results(
    df_corr,
    df_pvalue,
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
