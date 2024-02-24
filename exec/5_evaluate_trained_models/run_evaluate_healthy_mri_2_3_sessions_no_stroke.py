import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.load_data import healthy_load_data
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results.healthy import save_spearman_correlation_results, \
                                               save_hgs_predicted_results

from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


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
                                0,
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
                                0,
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
df = healthy_load_data.load_preprocessed_data(population, mri_status, session, gender="both_gender")

df_female = df[df["gender"] == 0]
df_male = df[df["gender"] == 1]


features = define_features(feature_type)

print("===== Done! =====")
embed(globals(), locals())

df_female_extracted, df_female_extracted_full_data = healthy_extract_data.extract_data(df_female, features, target, session)
df_male_extracted, df_male_extracted_full_data = healthy_extract_data.extract_data(df_male, features, target, session)

X = features
y = target

df_female_tmp = predict_hgs(df_female_extracted, X, y, female_best_model_trained, target)
df_male_tmp = predict_hgs(df_male_extracted, X, y, male_best_model_trained, target)

# Prefix to add to column names
# if session == "2":
#     prefix = "1st_scan_"
# elif session == "3":
#     prefix = "2nd_scan_"

# excluded_columns = ["gender"]
# # Rename columns, adding the prefix only to columns not in the excluded list
# for df in [df_female, df_male]:
#     for col in df.columns:
#         if col not in excluded_columns:
#             df.rename(columns={col: prefix + col}, inplace=True)

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

##############################################################################
y_axis = ["actual", "predicted", "actual-predicted"]
x_axis = ["actual", "predicted"]
df_corr, df_pvalue = calculate_spearman_hgs_correlation(df_both_gender, y_axis, x_axis)
df_female_corr, df_female_pvalue = calculate_spearman_hgs_correlation(df_female, y_axis, x_axis)
df_male_corr, df_male_pvalue = calculate_spearman_hgs_correlation(df_male, y_axis, x_axis)
print(df_corr)
print(df_female_corr)
print(df_male_corr)
save_spearman_correlation_results(
    df_corr,
    df_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
)
save_spearman_correlation_results(
    df_female_corr,
    df_female_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
)
save_spearman_correlation_results(
    df_male_corr,
    df_male_pvalue,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
)


print("===== Done! =====")
embed(globals(), locals())
