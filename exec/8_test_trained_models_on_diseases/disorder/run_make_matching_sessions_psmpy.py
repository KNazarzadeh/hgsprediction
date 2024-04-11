import math
import sys
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from hgsprediction.load_results.load_zscore_results import load_zscore_results
from hgsprediction.load_results.load_disorder_corrected_prediction_results import load_disorder_corrected_prediction_results
from hgsprediction.define_features import define_features
from hgsprediction.save_results.save_disorder_matched_samples_results import save_disorder_matched_samples_results
from hgsprediction.save_results.save_disorder_matched_samples_correlation_results import save_disorder_matched_control_samples_correlation_results

from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
import seaborn as sns
import matplotlib.pyplot as plt
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
disorder_cohort = sys.argv[9]
visit_session = sys.argv[10]
gender = sys.argv[11]
n_samples = sys.argv[12]

##############################################################################
features, extend_features = define_features(feature_type)

# Define features and target for matching
X = features
y = "disorder"

extract_columns = X + [y]

disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

df_disorder = load_disorder_corrected_prediction_results(
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
)

df_disorder.loc[:, "disorder"] = 1

pre_sessions = df_disorder[f"1st_pre-{population}_session"]

post_sessions = df_disorder[f"1st_post-{population}_session"]

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Load z-score results for healthy individuals with MRI data
# And asssign to control dataframe
session = 0
df_control_sessin_0 = load_zscore_results(
    "healthy",
    "mri",
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
)
df_control_sessin_0.loc[:, "disorder"] = 0
##############################################################################
session = 1
# Load z-score results for healthy individuals with MRI data
# And asssign to control dataframe
df_control_sessin_1 = load_zscore_results(
    "healthy",
    "mri",
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
)
df_control_sessin_1.loc[:, "disorder"] = 0
##############################################################################
session = 2
# Load z-score results for healthy individuals with MRI data
# And asssign to control dataframe
df_control_sessin_2 = load_zscore_results(
    "healthy",
    "mri",
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
)
df_control_sessin_2.loc[:, "disorder"] = 0
##############################################################################
session = 3
# Load z-score results for healthy individuals with MRI data
# And asssign to control dataframe
df_control_sessin_3 = load_zscore_results(
    "healthy",
    "mri",
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
)
df_control_sessin_3.loc[:, "disorder"] = 0

###############################################################################
disorder_pre_subgroup = f"pre-{population}"
if visit_session == "1":
    pre_prefix = f"1st_{disorder_pre_subgroup}_"
elif visit_session == "2":
    pre_prefix = f"2nd_{disorder_pre_subgroup}_"
elif visit_session == "3":
    pre_prefix = f"3rd_{disorder_pre_subgroup}_"
elif visit_session == "4":
    pre_prefix = f"4th_{disorder_pre_subgroup}_"

disorder_post_subgroup = f"post-{population}"
if visit_session == "1":
    post_prefix = f"1st_{disorder_post_subgroup}_"
elif visit_session == "2":
    post_prefix = f"2nd_{disorder_post_subgroup}_"
elif visit_session == "3":
    post_prefix = f"3rd_{disorder_post_subgroup}_"
elif visit_session == "4":
    post_prefix = f"4th_{disorder_post_subgroup}_"
###############################################################################
# Step 1: Calculate propensity scores for each patient
# Assuming you have a column named 'propensity_scores' in patient_df representing propensity scores

df_control_matched = pd.DataFrame()

pre_ses_min = int(df_disorder[f"{pre_prefix}session"].min())
pre_ses_max = int(df_disorder[f"{pre_prefix}session"].max())
post_ses_min = int(df_disorder[f"{post_prefix}session"].min())
post_ses_max = int(df_disorder[f"{post_prefix}session"].max())
# print("===== Done! End =====")
# embed(globals(), locals())
for pre_ses in range(pre_ses_min, pre_ses_max+1):
    print("pre_ses=", pre_ses)
    # Iterate over the range of session values
    df_disorder_pre_sessions = df_disorder[df_disorder[f"{pre_prefix}session"]==pre_ses]
    # Assuming df is your DataFrame
    if not df_disorder_pre_sessions.empty:
        if pre_ses == 0.0:
            df_control_pre = df_control_sessin_0.copy()
        if pre_ses == 1.0:
            df_control_pre = df_control_sessin_1.copy()
        if pre_ses == 2.0:
            df_control_pre = df_control_sessin_2.copy()
        
        df_control_matched_tmp = pd.DataFrame()
        for post_ses in range(pre_ses+1, post_ses_max+1):
            print("post_ses=", post_ses)
            # Iterate over the range of session values
            df_disorder_post_sessions = df_disorder[df_disorder[f"{post_prefix}session"]==post_ses]
            # Assuming df is your DataFrame
            if not df_disorder_post_sessions.empty:
                if post_ses == 1.0:
                    df_control_post = df_control_sessin_1.copy()
                if post_ses == 2.0:
                    df_control_post = df_control_sessin_2.copy()
                if post_ses == 3.0:
                    if pre_ses == 2:
                        df_control_post = df_control_sessin_3.copy()
                    else:
                        break  # or whatever statement you use to exit the loop
            intersection_index = df_control_pre.index.intersection(df_control_post.index)
            df_control_pre = df_control_pre[df_control_pre.index.isin(intersection_index)]
            df_control_post = df_control_post[df_control_post.index.isin(intersection_index)]
            
            # Reindex the dataframes to have the same order of indices
            df_control_pre = df_control_pre.reindex(index=intersection_index)
            df_control_post = df_control_post.reindex(index=intersection_index)

            # Check if the indices are in the same order
            if df_control_pre.index.equals(df_control_post.index):
                print("The indices are in the same order.")
            else:
                print("The indices are not in the same order.")

            df_disorder_extract = df_disorder_pre_sessions[df_disorder_pre_sessions.index.isin(df_disorder_post_sessions.index)]
            df_disorder_pre = df_disorder_extract[[col for col in df_disorder_extract.columns if f"post-{population}" not in col]]

            features_columns = [col for col in df_disorder_pre.columns for item in extract_columns if item in col]

            # Remove the prefix from selected column names
            for col in features_columns:
                new_col_name = col.replace(pre_prefix, "")
                df_disorder_pre.rename(columns={col: new_col_name}, inplace=True)
    
            df = pd.concat([df_control_pre.loc[:, extract_columns], df_disorder_pre.loc[:, extract_columns]], axis=0)
            df.index.name = "SubjectID"
            df.loc[:, "SubjectID"] = df.index
###############################################################################
            psm = PsmPy(df, treatment='disorder', indx='SubjectID', exclude = [])
            psm.logistic_ps(balance = False)
            psm.knn_matched_12n(matcher='propensity_logit', how_many=10)
###############################################################################
            df_control_pre_tmp = psm.controldf
            df_disorder_tmp = psm.treatmentdf
            # Dictionary to store matched samples for each subject
            df_matched_tmp = pd.DataFrame()
            df_control_pre_matched = pd.DataFrame()
            df_control_post_matched = pd.DataFrame()
            
            for i in range(len(df_disorder_tmp)):
                matches = psm.matched_ids.iloc[i,:].to_list()
                df_matched_tmp = pd.concat([df_matched_tmp, df_control_pre[df_control_pre.index.isin(matches)]], axis=0)
            
            df_control_pre_matched = pd.concat([df_control_pre_matched, df_matched_tmp], axis=0)

            ###############################################################################
            print(psm.matched_ids)
            # print("===== Done! End =====")
            # embed(globals(), locals())
            ###############################################################################
            df_control_post_matched = df_control_post[df_control_post.index.isin(df_control_pre_matched.index)]
            df_control_post_matched["disorder_episode"] = f"post-{population}"
            # Reindex the dataframes to have the same order of indices
            df_control_post_matched = df_control_post_matched.reindex(index=df_control_pre_matched.index)

            # Check if the indices are in the same order
            if df_control_pre_matched.index.equals(df_control_post_matched.index):
                print("The indices are in the same order.")
            else:
                print("The indices are not in the same order.")    
            # print("===== Done! End =====")
            # embed(globals(), locals())        
            ###############################################################################
            # Add prefix to column names
            df_control_pre_matched.columns = [pre_prefix + col if col != 'gender' else col for col in df_control_pre_matched.columns]
            # Add the prefix to the column names excluding the gender column(s)
            df_control_post_matched.columns = [post_prefix + col if col != 'gender' else col for col in df_control_post_matched.columns]
            
            df_control_matched_pre_post = pd.concat([df_control_pre_matched, df_control_post_matched], axis=1)
            
            # Check if the indices are in the same order
            if df_control_matched_pre_post.index.equals(df_control_pre_matched.index):
                print("The indices are in the same order.")
            else:
                print("The indices are not in the same order.")
    
            # Remove the specified suffixes from the column names in df1
            df_control_matched_pre_post.columns = df_control_matched_pre_post.columns.str.replace(r'-[0-3]\.0$', '', regex=True)

            df_control_matched_tmp = pd.concat([df_control_matched_tmp, df_control_matched_pre_post], axis=0)
            print(df_control_matched_tmp)
            # print("===== Done! End =====")
            # embed(globals(), locals())
    df_control_matched = pd.concat([df_control_matched, df_control_matched_tmp], axis=0)
    print(df_control_matched)

print(df_control_matched)
print(df_disorder)
##############################################################################
df_check_matching_pre = pd.DataFrame(columns=["patinets_pre_episode", "controls_pre_episode", "differece_pre_episode"])
# Adding a new index'
age_mean_disorder = df_disorder[f"1st_pre-{population}_age"].mean()
age_mean_control = df_control_matched[f"1st_pre-{population}_age"].mean()
# Calculate the difference between the mean ages
age_difference = age_mean_disorder - age_mean_control
df_check_matching_pre.loc['age_mean'] = [age_mean_disorder, age_mean_control, age_difference]

bmi_mean_disorder = df_disorder[f"1st_pre-{population}_bmi"].mean()
bmi_mean_control = df_control_matched[f"1st_pre-{population}_bmi"].mean()
bmi_difference = bmi_mean_disorder - bmi_mean_control
df_check_matching_pre.loc['bmi_mean'] = [bmi_mean_disorder, bmi_mean_control, bmi_difference]

height_mean_disorder = df_disorder[f"1st_pre-{population}_height"].mean()
height_mean_control = df_control_matched[f"1st_pre-{population}_height"].mean()
height_difference = height_mean_disorder - height_mean_control
df_check_matching_pre.loc['height_mean'] = [height_mean_disorder, height_mean_control, height_difference]

WHR_mean_disorder = df_disorder[f"1st_pre-{population}_waist_to_hip_ratio"].mean()
WHR_mean_control= df_control_matched[f"1st_pre-{population}_waist_to_hip_ratio"].mean()
WHR_difference = WHR_mean_disorder - WHR_mean_control
df_check_matching_pre.loc['WHR_mean'] = [WHR_mean_disorder, WHR_mean_control, WHR_difference]


df_check_matching_post = pd.DataFrame(columns=["patinets_post_episode", "controls_post_episode", "differece_post_episode"])
# Adding a new index'
age_mean_disorder = df_disorder[f"1st_post-{population}_age"].mean()
age_mean_control = df_control_matched[f"1st_post-{population}_age"].mean()
# Calculate the difference between the mean ages
age_difference = age_mean_disorder - age_mean_control
df_check_matching_post.loc['age_mean'] = [age_mean_disorder, age_mean_control, age_difference]

bmi_mean_disorder = df_disorder[f"1st_post-{population}_bmi"].mean()
bmi_mean_control = df_control_matched[f"1st_post-{population}_bmi"].mean()
bmi_difference = bmi_mean_disorder - bmi_mean_control
df_check_matching_post.loc['bmi_mean'] = [bmi_mean_disorder, bmi_mean_control, bmi_difference]

height_mean_disorder = df_disorder[f"1st_post-{population}_height"].mean()
height_mean_control = df_control_matched[f"1st_post-{population}_height"].mean()
height_difference = height_mean_disorder - height_mean_control
df_check_matching_post.loc['height_mean'] = [height_mean_disorder, height_mean_control, height_difference]

WHR_mean_disorder = df_disorder[f"1st_post-{population}_waist_to_hip_ratio"].mean()
WHR_mean_control= df_control_matched[f"1st_post-{population}_waist_to_hip_ratio"].mean()
WHR_difference = WHR_mean_disorder - WHR_mean_control
df_check_matching_post.loc['WHR_mean'] = [WHR_mean_disorder, WHR_mean_control, WHR_difference]

print(df_check_matching_pre)
print(df_check_matching_post)
print("===== Done! End =====")
embed(globals(), locals())
##############################################################################
# Visualise new distribution
fig, ax = plt.subplots(1,2)
sns.kdeplot(data=df_control_matched, x=df, fill=True, color="green", ax=ax[1])
sns.kdeplot(data=df_disorder, x=f"1st_pre-{population}_propensity_scores", fill=True, color="pink", ax=ax[1])
sns.kdeplot(data=df_control_matched, x=f"1st_pre-{population}_propensity_scores", fill=True, color="green", ax=ax[1])
sns.kdeplot(data=df_disorder, x=f"1st_pre-{population}_propensity_scores", fill=True, color="pink", ax=ax[1])
plt.show()
plt.savefig("x_match_psm.png")
plt.close()
print("===== Done! End =====")
embed(globals(), locals())
sample_session = 1
save_disorder_matched_samples_results(
    df_control_matched,
    df_disorder,
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
    n_samples,
    sample_session,
)

print("===== Done! =====")
embed(globals(), locals())
###############################################################################
df_correlations = pd.DataFrame(index=[f"pre-{population}", f"post-{population}"])
df_p_values = pd.DataFrame(index=[f"pre-{population}", f"post-{population}"])
df_r2_values = pd.DataFrame(index=[f"pre-{population}", f"post-{population}"])

for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:

    df = df_control_matched[df_control_matched["disorder_episode"] == disorder_subgroup]

    df_correlations.loc[disorder_subgroup, "r_values_true_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])[0]
    df_correlations.loc[disorder_subgroup, "r_values_true_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])[0]
    df_correlations.loc[disorder_subgroup, "r_values_true_corrected_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])[0]
    df_correlations.loc[disorder_subgroup, "r_values_true_corrected_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])[0]

    df_p_values.loc[disorder_subgroup, "r_values_true_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])[1]
    df_p_values.loc[disorder_subgroup, "r_values_true_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])[1]
    df_p_values.loc[disorder_subgroup, "r_values_true_corrected_predicted"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])[1]
    df_p_values.loc[disorder_subgroup, "r_values_true_corrected_delta"] = pearsonr(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])[1]


    df_r2_values.loc[disorder_subgroup, "r2_values_true_predicted"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_predicted"])
    df_r2_values.loc[disorder_subgroup, "r2_values_true_delta"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_delta(true-predicted)"])
    df_r2_values.loc[disorder_subgroup, "r2_values_true_corrected_predicted"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_predicted"])
    df_r2_values.loc[disorder_subgroup, "r2_values_true_corrected_delta"] = r2_score(df.loc[:, f"{target}"],df.loc[:, f"{target}_corrected_delta(true-predicted)"])

save_disorder_matched_control_samples_correlation_results(
    df_correlations,
    df_p_values,
    df_r2_values,
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
    n_samples,
)

print("===== Done! End =====")
embed(globals(), locals())


