import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.extract_data import depression_extract_data
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results import depression_save_spearman_correlation_results
from hgsprediction.save_results import depression_save_hgs_predicted_results
from hgsprediction.load_data import depression_load_data
from hgsprediction.load_results import load_trained_models

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
depression_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
model_name = sys.argv[7]

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
if visit_session == "1":
    session_column = f"1st_{depression_cohort}_session"
if mri_status == "mri+nonmri":
    df_longitudinal_mri = depression_load_data.load_preprocessed_data(population, "mri", session_column, depression_cohort)
    df_longitudinal_nonmri = depression_load_data.load_preprocessed_data(population, "nonmri", session_column, depression_cohort)
    df_longitudinal = pd.concat([df_longitudinal_mri, df_longitudinal_nonmri])
else:
    df_longitudinal = depression_load_data.load_preprocessed_data(population, mri_status, session_column, depression_cohort)

features = define_features(feature_type)
X = features
y = target
# print("===== Done! =====")
# embed(globals(), locals())

df_merged = pd.DataFrame()
for depression_subgroup in ["pre-depression", "post-depression"]:
    df_extracted = df_longitudinal[[col for col in df_longitudinal.columns if depression_subgroup in col]]
    df_extracted = depression_extract_data.extract_data(df_extracted, depression_subgroup, visit_session, features, target)
    
    df_female = df_extracted[df_extracted["gender"] == 0]
    df_male = df_extracted[df_extracted["gender"] == 1]
    
    df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
    df_male = predict_hgs(df_male, X, y, male_best_model_trained, target) 
    if visit_session == "1":
        # Define the string to add
        prefix = f"1st_{depression_subgroup}_"
        # Add the suffix to all column names
        df_female.columns = [prefix + col for col in df_female.columns]
        df_male.columns = [prefix + col for col in df_male.columns]

    df_both_gender = pd.concat([df_female, df_male], axis=0)
    df_merged = pd.concat([df_merged, df_both_gender], axis=1)

df_merged = df_merged.dropna()
if df_merged['1st_pre-depression_gender'].astype(float).equals((df_merged['1st_post-depression_gender'].astype(float))):
    df_merged.insert(0, "gender", df_merged["1st_pre-depression_gender"])
    df_merged = df_merged.drop(columns=['1st_pre-depression_gender', '1st_post-depression_gender'])

df_female = df_merged[df_merged["gender"] == 0]
df_male = df_merged[df_merged["gender"] == 1]
# print("===== Done! =====")
# embed(globals(), locals())
depression_save_hgs_predicted_results(
    df_merged,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session_column,
)

depression_save_hgs_predicted_results(
    df_female,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session_column,
)

depression_save_hgs_predicted_results(
    df_male,
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session_column,
)

##############################################################################
y_axis = ["actual", "predicted", "actual-predicted"]
x_axis = ["actual", "predicted", "years"]
for depression_subgroup in ["pre-depression", "post-depression"]:
    df_extracted = df_merged[[col for col in df_merged.columns if depression_subgroup in col]]
    df_extracted.loc[:, "gender"] = df_merged.loc[:, "gender"]
    df_corr, df_pvalue = calculate_spearman_hgs_correlation(df_extracted, y_axis, x_axis)
    df_female_corr, df_female_pvalue = calculate_spearman_hgs_correlation(df_extracted[df_extracted["gender"]==0], y_axis, x_axis)
    df_male_corr, df_male_pvalue = calculate_spearman_hgs_correlation(df_extracted[df_extracted["gender"]==1], y_axis, x_axis)

    print(df_corr.applymap(lambda x: '{:.3f}'.format(x)))
    print(df_female_corr.applymap(lambda x: '{:.3f}'.format(x)))
    print(df_male_corr.applymap(lambda x: '{:.3f}'.format(x)))

    depression_save_spearman_correlation_results(
        df_corr,
        df_pvalue,
        population,
        mri_status,
        model_name,
        feature_type,
        target,
        "both_gender",
        session_column,)
    depression_save_spearman_correlation_results(
        df_female_corr,
        df_female_pvalue,
        population,
        mri_status,
        model_name,
        feature_type,
        target,
        "female",
        session_column,)
    depression_save_spearman_correlation_results(
        df_male_corr,
        df_male_pvalue,
        population,
        mri_status,
        model_name,
        feature_type,
        target,
        "male",
        session_column,)


print("===== Done! =====")
embed(globals(), locals())
