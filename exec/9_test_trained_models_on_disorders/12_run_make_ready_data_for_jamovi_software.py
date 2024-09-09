import sys
import os
import numpy as np
import pandas as pd

from hgsprediction.load_results.anova.load_prepared_data_for_anova import load_prepare_data_for_anova
from hgsprediction.save_results.anova.save_prepared_data_for_jamovi_software import save_prepared_data_for_jamovi_software
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
anova_target = sys.argv[13]
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
    first_event,
)
#-----------------------------------------------------------#
df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)
# Replace values based on time_points
df.loc[df['time_point'].str.contains('pre-'), 'time_point'] = 'pre'
df.loc[df['time_point'].str.contains('post-'), 'time_point'] = 'post'
#-----------------------------------------------------------#
df = df[["gender", "group", "time_point", f"{anova_target}"]]
#-----------------------------------------------------------#
df_pre = df[df['time_point'] == "pre"]
df_pre.loc[:, "pre"] = df_pre.loc[:, f"{anova_target}"]

df_post = df[df['time_point'] == "post"]
df_post.loc[:, "post"] = df_post.loc[:, f"{anova_target}"]
#-----------------------------------------------------------#
df_pre = df_pre.drop(columns=[f"{anova_target}", "time_point"])
df_post = df_post.drop(columns=[f"{anova_target}", "time_point"])
#-----------------------------------------------------------#
df_merged = pd.merge(df_pre, df_post, on=["SubjectID", "gender", "group"], how="inner")
print(df_merged)
##############################################################################
save_prepared_data_for_jamovi_software(
    df_merged,
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
    anova_target,
    first_event,
)

print("===== Done! =====")
embed(globals(), locals())




