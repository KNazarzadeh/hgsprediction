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

###############################################################################
female_best_model_trained = load_trained_models.load_best_model_trained(
                                "healthy",
                                "nonmri",
                                0,
                                "female",
                                feature_type,
                                target,
                                "linear_svm",
                                10,
                                5,
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
                                10,
                                5,
                            )
print(male_best_model_trained)
##############################################################################
# load data
df = healthy_load_data.load_preprocessed_data(population, mri_status, session, gender="both_gender")

features = define_features(feature_type)

df_extracted = healthy_extract_data.extract_data(df, mri_status, features, target, session)

X = features
y = target

df_female = df_extracted[df_extracted["gender"] == 0]
df_male = df_extracted[df_extracted["gender"] == 1]

df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)

print(df_female)
print(df_male)

df_both_gender = pd.concat([df_female, df_male], axis=0)
print(df_both_gender)

save_hgs_predicted_results(
    df_both_gender,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
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
