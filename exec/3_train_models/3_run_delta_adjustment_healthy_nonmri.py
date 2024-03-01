import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.load_data import healthy_load_data
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
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]

###############################################################################
female_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                "female",
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
                            )

print(female_best_model_trained)

male_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                int(confound_status),
                                "male",
                                feature_type,
                                target,
                                "linear_svm",
                                n_repeats,
                                n_folds,
                            )
print(male_best_model_trained)

##############################################################################
# load data
df = healthy_load_data.load_preprocessed_data(population, mri_status, session, "both_gender")

features, extend_features = define_features(feature_type)

data_extracted = healthy_extract_data.extract_data(df, features, extend_features, feature_type, target, mri_status, session)

X = features
y = target
print("===== Done! =====")
embed(globals(), locals())
df_female = data_extracted[data_extracted["gender"] == 0]
df_male = data_extracted[data_extracted["gender"] == 1]

df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)

print(df_female)
print(df_male)

df_both_gender = pd.concat([df_female, df_male], axis=0)
print(df_both_gender)
print("===== Done! =====")
embed(globals(), locals())
save_hgs_predicted_results(
    df_both_gender,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
    confound_status,
    n_repeats,
    n_folds,
)

save_hgs_predicted_results(
    df_female,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,
)

save_hgs_predicted_results(
    df_male,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,
)
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
y_axis = ["true", "predicted", "delta(true-predicted)"]
x_axis = ["true", "predicted"]
df_corr, df_pvalue = calculate_pearson_hgs_correlation(df_both_gender, y_axis, x_axis)
df_female_corr, df_female_pvalue = calculate_pearson_hgs_correlation(df_female, y_axis, x_axis)
df_male_corr, df_male_pvalue = calculate_pearson_hgs_correlation(df_male, y_axis, x_axis)
print(df_corr)
print(df_female_corr)
print(df_male_corr)

save_correlation_results(
    df_corr,
    df_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
    confound_status,
    n_repeats,
    n_folds,
    )
save_correlation_results(
    df_female_corr,
    df_female_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,
    )
save_correlation_results(
    df_male_corr,
    df_male_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,
    )


print("===== Done! =====")
embed(globals(), locals())
