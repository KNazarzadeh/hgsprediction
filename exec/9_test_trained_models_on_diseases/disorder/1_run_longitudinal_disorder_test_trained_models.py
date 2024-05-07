import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.extract_data import disorder_extract_data

# from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
# from hgsprediction.save_results.stroke_save_spearman_correlation_results import stroke_save_spearman_correlation_results
from hgsprediction.save_results.save_disorder_hgs_predicted_results import save_disorder_hgs_predicted_results

from hgsprediction.load_data import disorder_load_data
from hgsprediction.load_results import load_trained_models

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
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

if mri_status == "mri+nonmri":
    df_longitudinal_mri = disorder_load_data.load_preprocessed_data(population, "mri", session_column, disorder_cohort)
    df_longitudinal_nonmri = disorder_load_data.load_preprocessed_data(population, "nonmri", session_column, disorder_cohort)
    df_longitudinal = pd.concat([df_longitudinal_mri, df_longitudinal_nonmri]).dropna(axis=1, how='all')
else:
    df_longitudinal = disorder_load_data.load_preprocessed_data(population, mri_status, session_column, disorder_cohort)

features, extend_features = define_features(feature_type)

X = features
y = target

df_both = pd.DataFrame()
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    df_extracted = disorder_extract_data.extract_data(df_longitudinal, population, features, extend_features, target, disorder_subgroup, visit_session)

    df_female = df_extracted[df_extracted["gender"] == 0]
    df_male = df_extracted[df_extracted["gender"] == 1]
    
    df_female = predict_hgs(df_female, X, y, female_best_model_trained, target)
    df_male = predict_hgs(df_male, X, y, male_best_model_trained, target)
    
    df_tmp = pd.concat([df_female, df_male], axis=0)
        
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}_"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}_"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}_"

    # Filter columns that require the prefix to be added
    filtered_columns = [col for col in df_tmp.columns if col in features + [target] + [f"{target}_predicted"] + [f"{target}_delta(true-predicted)"]]

    # Add the prefix to selected column names
    for col in filtered_columns:
        new_col_name = prefix + col
        df_tmp.rename(columns={col: new_col_name}, inplace=True)

    # Concatenate the DataFrames
    df_both = pd.concat([df_both, df_tmp], axis=1)

    # Drop duplicate columns
    df_both = df_both.loc[:,~df_both.columns.duplicated()]

df_female = df_both[df_both["gender"] == 0]
df_male = df_both[df_both["gender"] == 1]
print("===== Done! =====")
embed(globals(), locals())
save_disorder_hgs_predicted_results(
    df_both,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "both_gender",
    confound_status,
    n_repeats,
    n_folds,
)

save_disorder_hgs_predicted_results(
    df_female,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "female",
    confound_status,
    n_repeats,
    n_folds,
)

save_disorder_hgs_predicted_results(
    df_male,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "male",
    confound_status,
    n_repeats,
    n_folds,
)
print("===== END Done! =====")
embed(globals(), locals())
##############################################################################
# y_axis = ["actual", "predicted", "actual-predicted"]
# x_axis = ["actual", "predicted", "years"]
# for stroke_subgroup in ["pre-stroke", "post-stroke"]:
#     df_extracted = df_merged[[col for col in df_merged.columns if stroke_subgroup in col]]
#     df_extracted.loc[:, "gender"] = df_merged.loc[:, "gender"]
#     df_corr, df_pvalue = calculate_spearman_hgs_correlation(df_extracted, y_axis, x_axis)
#     df_female_corr, df_female_pvalue = calculate_spearman_hgs_correlation(df_extracted[df_extracted["gender"]==0], y_axis, x_axis)
#     df_male_corr, df_male_pvalue = calculate_spearman_hgs_correlation(df_extracted[df_extracted["gender"]==1], y_axis, x_axis)

#     print(df_corr.applymap(lambda x: '{:.3f}'.format(x)))
#     print(df_female_corr.applymap(lambda x: '{:.3f}'.format(x)))
#     print(df_male_corr.applymap(lambda x: '{:.3f}'.format(x)))

#     stroke_save_spearman_correlation_results(
#         df_corr,
#         df_pvalue,
#         population,
#         mri_status,
#         session_column,
#         model_name,
#         feature_type,
#         target,
#         stroke_subgroup,
#         "both_gender")
#     stroke_save_spearman_correlation_results(
#         df_female_corr,
#         df_female_pvalue,
#         population,
#         mri_status,
#         session_column,
#         model_name,
#         feature_type,
#         target,
#         stroke_subgroup,
#         "female")
#     stroke_save_spearman_correlation_results(
#         df_male_corr,
#         df_male_pvalue,
#         population,
#         mri_status,
#         session_column,
#         model_name,
#         feature_type,
#         target,
#         stroke_subgroup,
#         "male")


print("===== Done! =====")
embed(globals(), locals())
