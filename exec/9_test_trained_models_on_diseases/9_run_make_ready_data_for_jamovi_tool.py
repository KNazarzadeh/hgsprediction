import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns

from hgsprediction.load_results.load_prepared_data_for_anova import load_prepare_data_for_anova

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
anova_target = sys.argv[12]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
##############################################################################
# Load data for ANOVA
df = load_prepare_data_for_anova(
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
)
df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)
# Replace values based on time_points
df.loc[df['time_point'].str.contains('pre-'), 'time_point'] = 'pre'
df.loc[df['time_point'].str.contains('post-'), 'time_point'] = 'post'


df = df[["gender", "group", "time_point", f"{anova_target}"]]


df_pre = df[df['time_point'] == "pre"]
df_pre.loc[:, "pre"] = df_pre.loc[:, f"{anova_target}"]

df_post = df[df['time_point'] == "post"]
df_post.loc[:, "post"] = df_post.loc[:, f"{anova_target}"]

df_pre = df_pre.drop(columns=[f"{anova_target}", "time_point"])
df_post = df_post.drop(columns=[f"{anova_target}", "time_point"])

df_merged = pd.merge(df_pre, df_post, on=["SubjectID", "gender", "group"], how="inner")


if confound_status == "0":
    confound = "without_confound_removal"
else:
    confound = "with_confound_removal"
# Assuming that you have already trained and instantiated the model as `model`
folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",  
        "results_hgsprediction",
        f"{population}",
        f"{mri_status}",
        f"{session_column}",
        f"{feature_type}",
        f"{target}",
        f"{confound}",
        f"{model_name}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        "matched_control_samples_results",
        f"1_to_{n_samples}_samples",
        "ANOVA_results",
        "jamovi_software",
        f"{anova_target}",
        "ready_data",
    )
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    "anova_gender_group_time_point_data.csv")

df_merged.to_csv(file_path, sep=',', index=0)

print("===== Done! =====")
embed(globals(), locals())




