import math
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc

from hgsprediction.load_results.load_disorder_matched_samples_results import load_disorder_matched_samples_results
from hgsprediction.save_results.save_disorder_anova_results import save_disorder_anova_results
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
model_name = sys.argv[4]
session = sys.argv[5]
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
disorder_cohort = sys.argv[9]
visit_session = sys.argv[10]
n_samples = sys.argv[11]
target = sys.argv[12]
anova_target = sys.argv[13]
##############################################################################
if anova_target != "hgs":
    main_extracted_columns = ["gender", "treatment", "disorder_episode", "hgs_target", "hgs"]
else:
    main_extracted_columns = ["gender", "treatment", "disorder_episode", "hgs_target"]
    
extract_columns = main_extracted_columns + [anova_target]

df_disorder = pd.DataFrame()
df_control = pd.DataFrame()

disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
    
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
)

df_disorder_matched_female.loc[:, "treatment"] = f"{population}"
df_mathced_controls_female.loc[:, "treatment"] = "control"
df_disorder_matched_male.loc[:, "treatment"] = f"{population}"
df_mathced_controls_male.loc[:, "treatment"] = "control"

df_disorder_matched_female.loc[:, "hgs_target"] = target
df_mathced_controls_female.loc[:, "hgs_target"] = target
df_disorder_matched_male.loc[:, "hgs_target"] = target
df_mathced_controls_male.loc[:, "hgs_target"] = target

df_disorder_tmp = pd.concat([df_disorder_matched_female, df_disorder_matched_male], axis=0)
df_control_tmp = pd.concat([df_mathced_controls_female, df_mathced_controls_male], axis=0)

df_disorder_tmp.columns = [col.replace(f"{target}", "hgs") if f"{target}" in col else col for col in df_disorder_tmp.columns]    
df_control_tmp.columns = [col.replace(f"{target}", "hgs") if f"{target}" in col else col for col in df_control_tmp.columns]

df_control_tmp = df_control_tmp.rename(columns={"matched_disorder_subgroup":"disorder_episode"})
# Replace values in the column
df_control_tmp.loc[:, "disorder_episode"] = df_control_tmp.loc[:, "disorder_episode"].replace({f"pre-{population}": "pre-control", f"post-{population}": "post-control"})

df_control_tmp.rename(columns=lambda x: x.replace("delta(true-predicted)", "delta") if "delta(true-predicted)" in x else x, inplace=True)

##############################################################################
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}_"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}_"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}_"
        
    if disorder_subgroup == f"pre-{population}":
        df_extracted_pre = df_disorder_tmp[[col for col in df_disorder_tmp.columns if f"post-{population}" not in col]]
        df_extracted_pre.loc[:, "disorder_episode"] = disorder_subgroup
        rename_columns = [col for col in df_extracted_pre.columns if prefix in col]
        # Remove the prefix from selected column names
        for col in rename_columns:
            new_col_name = col.replace(prefix, "")
            df_extracted_pre = df_extracted_pre.rename(columns={col: new_col_name})
        
        df_extracted_pre.rename(columns=lambda x: x.replace("delta(true-predicted)", "delta") if "delta(true-predicted)" in x else x, inplace=True)

    elif disorder_subgroup == f"post-{population}":
        df_extracted_post = df_disorder_tmp[[col for col in df_disorder_tmp.columns if f"pre-{population}" not in col]]
        df_extracted_post.loc[:, "disorder_episode"] = disorder_subgroup
        rename_columns = [col for col in df_extracted_post.columns if prefix in col]
        # Remove the prefix from selected column names
        for col in rename_columns:
            new_col_name = col.replace(prefix, "")
            df_extracted_post = df_extracted_post.rename(columns={col: new_col_name})

        df_extracted_post.rename(columns=lambda x: x.replace("delta(true-predicted)", "delta") if "delta(true-predicted)" in x else x, inplace=True)

df_tmp = pd.concat([df_extracted_pre.loc[:, extract_columns], df_extracted_post.loc[:, extract_columns]], axis=0)

df_disorder = pd.concat([df_disorder, df_tmp], axis=0)
df_control = pd.concat([df_control, df_control_tmp.loc[:, extract_columns]], axis=0)

##############################################################################
# Perform the ANOVA
df = pd.concat([df_control, df_disorder], axis=0)
df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)

formula = (
    f'{anova_target} ~ '
    'C(gender) + C(treatment) + C(disorder_episode) + C(hgs_target) + '
    'C(gender):C(treatment) + C(gender):C(disorder_episode) + C(gender):C(hgs_target) + '
    'C(treatment):C(disorder_episode) + C(treatment):C(hgs_target) + '
    'C(disorder_episode):C(hgs_target) + '
    'C(treatment):C(disorder_episode):C(hgs_target) + '
    'C(disorder_episode):C(hgs_target):C(gender) + '
    'C(treatment):C(disorder_episode):C(hgs_target):C(gender)'
)

model = ols(formula, data=df).fit()
df_anova_result = sm.stats.anova_lm(model)

print(df_anova_result)

# Perform post-hoc tests on significant interactions (Tukey's HSD)
# interaction_groups =  df.treatment.astype(str) + "_"+ df.disorder_episode.astype(str)+ "_" + df.hgs_target.astype(str)
interaction_groups =  df.disorder_episode.astype(str) + " | " + df.hgs_target.astype(str)
comp = mc.MultiComparison(df[f"{anova_target}"], interaction_groups)
df_post_hoc_result_without_gender = comp.tukeyhsd()
print(df_post_hoc_result_without_gender.summary())

interaction_groups =  df.gender.astype(str) + " | " + df.disorder_episode.astype(str) + " | " + df.hgs_target.astype(str)
comp = mc.MultiComparison(df[f"{anova_target}"], interaction_groups)
df_post_hoc_result_with_gender = comp.tukeyhsd()
print(df_post_hoc_result_with_gender.summary())

# print("===== Done! =====")
# embed(globals(), locals())

save_disorder_anova_results(
    df,
    df_anova_result,
    df_post_hoc_result_without_gender,
    df_post_hoc_result_with_gender,
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
)


print("===== Done! =====")
embed(globals(), locals())

