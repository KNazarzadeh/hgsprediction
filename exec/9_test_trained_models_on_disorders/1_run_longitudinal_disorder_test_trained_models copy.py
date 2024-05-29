import os
import pandas as pd
import numpy as np
import sys

from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.extract_data import disorder_extract_data
from hgsprediction.save_results.save_disorder_hgs_predicted_results import save_disorder_hgs_predicted_results
from hgsprediction.load_data import load_disorder_data
from hgsprediction.load_results import load_trained_models
from hgsprediction.load_results.load_disorder_extracted_data_by_features import load_disorder_extracted_data_by_features

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
gender = sys.argv[11]
first_event = sys.argv[12]
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

# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
# Define main features and extra features:
features, extend_features = define_features(feature_type)
##############################################################################
# Define X as main features and y as target:
X = features
y = target
##############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

if mri_status == "mri+nonmri":
    df_longitudinal_mri = load_disorder_data.load_preprocessed_data(population, "mri", session_column, disorder_cohort, first_event)
    df_longitudinal_nonmri = load_disorder_data.load_preprocessed_data(population, "nonmri", session_column, disorder_cohort, first_event)
    df_longitudinal = pd.concat([df_longitudinal_mri, df_longitudinal_nonmri]).dropna(axis=1, how='all')
else:
    df_longitudinal = load_disorder_data.load_preprocessed_data(population, mri_status, session_column, disorder_cohort, first_event)

if gender == "female":
    df = df_longitudinal[df_longitudinal['gender'] == 0]
elif gender == "male":
    df = df_longitudinal[df_longitudinal['gender'] == 1]

for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    df_extracted = disorder_extract_data.extract_data(df, population, features, extend_features, target, disorder_subgroup, visit_session)

    df_tmp = predict_hgs(df_extracted, X, y, best_model_trained, target)
            
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"

    # Filter columns that require the prefix to be added
    filtered_columns = [col for col in df_tmp.columns if col in features + [target] + [f"{target}_predicted"] + [f"{target}_delta(true-predicted)"]]

    # Add the prefix to selected column names
    for col in filtered_columns:
        new_col_name = prefix + col
        df_tmp.rename(columns={col: new_col_name}, inplace=True)

    # Concatenate the DataFrames
    if disorder_subgroup == f"pre-{population}":
        print(disorder_subgroup)
        df_pre = df_tmp.copy()
        
    elif disorder_subgroup == f"post-{population}":
        print(disorder_subgroup)
        df_post = df_tmp.copy()
        # Merging DataFrames on index

# Reindex df_pre to match the index order of df_post
df_post = df_post.reindex(df_pre.index)

# Check if the indices of both dataframes are the same and in the same order
indices_are_same_and_in_same_order = df_pre.index.equals(df_post.index)
print("Indices are the same and in the same order:", indices_are_same_and_in_same_order)
# Finding common columns
common_cols = df_pre.columns.intersection(df_post.columns)
# Merge DataFrames on index without duplicating common columns
df_merged = pd.merge(df_pre.drop(columns=common_cols), df_post, left_index=True, right_index=True, how='inner')

print(df_merged)
print("===== END Done! =====")
embed(globals(), locals())
save_disorder_hgs_predicted_results(
    df_merged,
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    gender,
    confound_status,
    n_repeats,
    n_folds,
    first_event,
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
