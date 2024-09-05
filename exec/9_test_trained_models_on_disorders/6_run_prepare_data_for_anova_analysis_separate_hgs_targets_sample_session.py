import math
import sys
import os
import numpy as np
import pandas as pd

from hgsprediction.load_results.disorder.load_disorder_matched_samples_results import load_disorder_matched_samples_results
from hgsprediction.save_results.anova.save_prepared_data_for_anova import save_prepare_data_for_anova

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
confound_status = sys.argv[5]
n_repeats = sys.argv[6]
n_folds = sys.argv[7]
disorder_cohort = sys.argv[8]
visit_session = sys.argv[9]
n_samples = sys.argv[10]
target = sys.argv[11]
first_event = sys.argv[12]
##############################################################################
main_extracted_columns = ["gender", "handedness", "hgs_dominant", "hgs_dominant_side", "hgs_nondominant", "hgs_nondominant_side", 
                          "age", "bmi", "height", "waist_to_hip_ratio", 
                          "group", "time_point", "hgs_target", 
                          "true_hgs", "hgs_predicted", "hgs_delta", "hgs_corrected_predicted", "hgs_corrected_delta", 
                          "patient_id"]
##############################################################################
df_disorder = pd.DataFrame()
df_control = pd.DataFrame()
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
df_disorder_matched_female, df_mathced_controls_female = load_disorder_matched_samples_results(
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
    n_samples,
    first_event,
)

df_disorder_matched_male, df_mathced_controls_male = load_disorder_matched_samples_results(
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
    n_samples,
    first_event,    
)
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
df_disorder_matched_female.loc[:, "group"] = f"{population}"
df_mathced_controls_female.loc[:, "group"] = "control"

df_disorder_matched_female.loc[:, "hgs_target"] = target
df_mathced_controls_female.loc[:, "hgs_target"] = target

df_disorder_matched_female.loc[:, "patient_id"] = df_disorder_matched_female.index
#-----------------------------------------------------------#
df_disorder_matched_male.loc[:, "group"] = f"{population}"
df_mathced_controls_male.loc[:, "group"] = "control"

df_disorder_matched_male.loc[:, "hgs_target"] = target
df_mathced_controls_male.loc[:, "hgs_target"] = target

df_disorder_matched_male.loc[:, "patient_id"] = df_disorder_matched_male.index
#-----------------------------------------------------------#
df_disorder_tmp = pd.concat([df_disorder_matched_female, df_disorder_matched_male], axis=0)
df_control_tmp = pd.concat([df_mathced_controls_female, df_mathced_controls_male], axis=0)
#-----------------------------------------------------------#
# Replace values in the column
pre_prefix = f"1st_pre-{population}"
df_control_tmp.loc[:, f"{pre_prefix}_time_point"] = df_control_tmp.loc[:, f"{pre_prefix}_time_point"].replace({f"pre-{population}": "pre-control"})

post_prefix = f"1st_post-{population}"
df_control_tmp.loc[:, f"{post_prefix}_time_point"] = df_control_tmp.loc[:, f"{post_prefix}_time_point"].replace({f"post-{population}": "post-control"})
#-----------------------------------------------------------#
df_control_tmp.rename(columns=lambda x: x.replace("delta(true-predicted)", "delta") if "delta(true-predicted)" in x else x, inplace=True)
df_disorder_tmp.rename(columns=lambda x: x.replace("delta(true-predicted)", "delta") if "delta(true-predicted)" in x else x, inplace=True)

##############################################################################
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"
    #-----------------------------------------------------------#  
    if disorder_subgroup == f"pre-{population}":
        df_control_extracted_pre = df_control_tmp[[col for col in df_control_tmp.columns if f"post-{population}" not in col]]        
        df_disorder_extracted_pre = df_disorder_tmp[[col for col in df_disorder_tmp.columns if f"post-{population}" not in col]]
        # Make a copy before setting a value to avoid SettingWithCopyWarning
        df_disorder_extracted_pre = df_disorder_extracted_pre.copy()
        df_disorder_extracted_pre.loc[:, "time_point"] = disorder_subgroup
        #-----------------------------------------------------------#  
        dataframes = [df_disorder_extracted_pre, df_control_extracted_pre]
        for df in dataframes:
            rename_columns = [col for col in df.columns if prefix in col]
            # Avoid inplace modification for rename
            df = df.rename(columns={col: col.replace(prefix, "") for col in rename_columns}, inplace=True)
        #-----------------------------------------------------------#  
    elif disorder_subgroup == f"post-{population}":
        df_control_extracted_post = df_control_tmp[[col for col in df_control_tmp.columns if f"pre-{population}" not in col]]
        df_disorder_extracted_post = df_disorder_tmp[[col for col in df_disorder_tmp.columns if f"pre-{population}" not in col]]
        # Make a copy before setting a value to avoid SettingWithCopyWarning
        df_disorder_extracted_post = df_disorder_extracted_post.copy()
        df_disorder_extracted_post.loc[:, "time_point"] = disorder_subgroup
        #-----------------------------------------------------------# 
        dataframes = [df_disorder_extracted_post, df_control_extracted_post]
        for df in dataframes:
            rename_columns = [col for col in df.columns if prefix in col]
            # Avoid inplace modification for rename
            df = df.rename(columns={col: col.replace(prefix, "") for col in rename_columns}, inplace=True)
##############################################################################        
# Check if the indices are in the same order for disorder DataFrames
if df_disorder_extracted_pre.index.equals(df_disorder_extracted_post.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")
#-----------------------------------------------------------#
# Check if the indices are in the same order for control DataFrames
if df_control_extracted_pre.index.equals(df_control_extracted_post.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")
##############################################################################
dataframes = [df_disorder_extracted_pre, 
              df_disorder_extracted_post, 
              df_control_extracted_pre, 
              df_control_extracted_post]

# Rename the target columns to "true_hgs"
for df in dataframes:
    df.rename(columns={f"{target}": "true_hgs"}, inplace=True)
#-----------------------------------------------------------#
# Replace the target part in column names with "hgs"
for df in dataframes:
    df.columns = [col.replace(f"{target}", "hgs") if f"{target}" in col else col for col in df.columns]
##############################################################################
# Concatenate dataframes for disorder data
df_disorder = pd.concat([df_disorder_extracted_pre.loc[:, main_extracted_columns], df_disorder_extracted_post.loc[:, main_extracted_columns]], axis=0)
# Concatenate dataframes for control data
df_control = pd.concat([df_control_extracted_pre.loc[:, main_extracted_columns], df_control_extracted_post.loc[:, main_extracted_columns]], axis=0)
##############################################################################
# Perform the ANOVA
# Merge disorder and control dataframes
df_merged_disorder_control = pd.concat([df_control, df_disorder], axis=0)
# Print the merged dataframe
print(df_merged_disorder_control)

# print("===== Done! End =====")
# embed(globals(), locals())
##############################################################################
save_prepare_data_for_anova(
        df_merged_disorder_control,
        population,
        mri_status,
        session_column,
        model_name,
        feature_type,
        target,
        confound_status,
        n_repeats,
        n_folds,
        n_samples,
        first_event,
    )

print("===== Done! End =====")
embed(globals(), locals())

        
     
