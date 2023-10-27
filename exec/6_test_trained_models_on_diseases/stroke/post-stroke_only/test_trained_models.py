import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.extract_data import stroke_extract_data
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results import save_spearman_correlation_results
from hgsprediction.save_results import save_hgs_predicted_results
from hgsprediction.load_data import stroke_load_data
from hgsprediction.load_results import load_trained_models


from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# from hgsprediction.plots import plot_correlations
from hgsprediction.save_plot.save_correlations_plot import stroke_save_correlations_plot


from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
visit_session = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]
model_name = sys.argv[6]

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
stroke_cohort = "longitudinal-stroke"
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"
if mri_status == "mri+nonmri":
    df_longitudinal_mri = stroke_load_data.load_preprocessed_data(population, "mri", session_column, stroke_cohort)
    df_longitudinal_nonmri = stroke_load_data.load_preprocessed_data(population, "nonmri", session_column, stroke_cohort)
    df_longitudinal = pd.concat([df_longitudinal_mri, df_longitudinal_nonmri])

stroke_cohort = "post-stroke"
if visit_session == "1":
    session_column = f"1st_{stroke_cohort}_session"
elif visit_session == "2":
    session_column = f"2nd_{stroke_cohort}_session"
elif visit_session == "3":
    session_column = f"3rd_{stroke_cohort}_session"
elif visit_session == "4":
    session_column = f"4th_{stroke_cohort}_session"
if mri_status == "mri+nonmri":
    df_post_mri = stroke_load_data.load_preprocessed_data(population, "mri", session_column, stroke_cohort)
    df_post_nonmri = stroke_load_data.load_preprocessed_data(population, "nonmri", session_column, stroke_cohort)
    df_post = pd.concat([df_post_mri, df_post_nonmri])
print("===== Done! =====")
embed(globals(), locals())
# # df_female = df_female[(df_female["1st_post-stroke_session"]==2.0) | (df_female["1st_post-stroke_session"]== 3.0)]
# # df_male = df_male[(df_male["1st_post-stroke_session"]==2.0) | (df_male["1st_post-stroke_session"]== 3.0)]
df_same_subjects = df_post[df_post.index.isin(df_longitudinal.index)]

features = define_features(feature_type)
df_extracted = stroke_extract_data.extract_data(df_same_subjects, stroke_cohort, visit_session, features, target)

X = features
y = target

df_female = df_extracted[df_extracted["gender"] == 0]
df_male = df_extracted[df_extracted["gender"] == 1]

df_female = predict_hgs(df_female, X, y, female_best_model_trained)
df_male = predict_hgs(df_male, X, y, male_best_model_trained)
print(df_female)
print(df_male)

df_both_gender = pd.concat([df_female, df_male], axis=0)
print(df_both_gender)

save_hgs_predicted_results(
    df_both_gender,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "both_gender",
)

save_hgs_predicted_results(
    df_female,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "female",
)

save_hgs_predicted_results(
    df_male,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "male",
)

##############################################################################
y_axis = ["actual", "predicted", "actual-predicted"]
x_axis = ["actual", "predicted", "years"]
df_corr, df_pvalue = calculate_spearman_hgs_correlation(df_both_gender, y_axis, x_axis)
df_female_corr, df_female_pvalue = calculate_spearman_hgs_correlation(df_female, y_axis, x_axis)
df_male_corr, df_male_pvalue = calculate_spearman_hgs_correlation(df_male, y_axis, x_axis)
pd.options.display.float_format = '{:.3f}'.format
print(df_corr)
print(df_female_corr)
print(df_male_corr)

save_spearman_correlation_results(
    df_corr,
    df_pvalue,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "both_gender")
save_spearman_correlation_results(
    df_female_corr,
    df_female_pvalue,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "female")
save_spearman_correlation_results(
    df_male_corr,
    df_male_pvalue,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "male")


print("===== Done! =====")
embed(globals(), locals())
