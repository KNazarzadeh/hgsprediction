import math
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc


from hgsprediction.load_results.load_disorder_corrected_prediction_results import load_disorder_corrected_prediction_results
from hgsprediction.define_features import define_features
from hgsprediction.load_results.load_disorder_matched_samples_results import load_disorder_matched_samples_results

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
target_1 = sys.argv[12]
target_2 = sys.argv[13]

##############################################################################
df_disorder = pd.Dataframe()
df_control = pd.Dataframe()
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
    
for target in [f"{target_1}", f"{target_2}"]:
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
    
    df_disorder_matched_female.loc[:, "disorder_episode"] = target
    df_mathced_controls_female.loc[:, "disorder_episode"] = target
    df_disorder_matched_male.loc[:, "disorder_episode"] = target
    df_mathced_controls_male.loc[:, "disorder_episode"] = target
    
    df_disorder_tmp = pd.concat([df_disorder_matched_female, df_disorder_matched_male], axis=0)
    df_control_tmp = pd.concat([df_mathced_controls_female, df_mathced_controls_male], axis=0)
    
    df_disorder = pd.concat([df_disorder, df_disorder_tmp], axis=0)
    df_control = pd.concat([df_control, df_control_tmp], axis=0)
print("===== Done! =====")
embed(globals(), locals())
##############################################################################
# Perform the ANOVA
main_extracted_columns = ["disorder", "gender", "matched_disorder_subgroup", "treatment", f"{target}"]
anova_target_extracted_columns = [f"{target}_corrected_delta(true-predicted)"]

extract_columns = main_extracted_columns + anova_target_extracted_columns

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
        df_extracted_pre = df_disorder[[col for col in df_disorder.columns if f"post-{population}" not in col]]
        df_extracted_pre.loc[:, "matched_disorder_subgroup"] = disorder_subgroup
        rename_columns = [col for col in df_extracted_pre.columns if prefix in col]
        # Remove the prefix from selected column names
        for col in rename_columns:
            new_col_name = col.replace(prefix, "")
            df_extracted_pre.rename(columns={col: new_col_name}, inplace=True)

    elif disorder_subgroup == f"post-{population}":
        df_extracted_post = df_disorder[[col for col in df_disorder.columns if f"pre-{population}" not in col]]
        df_extracted_post.loc[:, "matched_disorder_subgroup"] = disorder_subgroup
        rename_columns = [col for col in df_extracted_post.columns if prefix in col]
        # Remove the prefix from selected column names
        for col in rename_columns:
            new_col_name = col.replace(prefix, "")
            df_extracted_post.rename(columns={col: new_col_name}, inplace=True)
            

# df_extracted_control = df_control[df_control["matched_disorder_subgroup"] == f"pre-{population}"]
    
# df = pd.concat([df_extracted_control.loc[:, extract_columns], df_extracted.loc[:, extract_columns]], axis=0)

df_d = pd.concat([df_extracted_pre.loc[:, extract_columns], df_extracted_post.loc[:, extract_columns]], axis=0)
df = pd.concat([df_control.loc[:, extract_columns], df_d.loc[:, extract_columns]], axis=0)

df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)
    
    
df.rename(columns={f"{target}": "hgs_combined", f"{target}_corrected_delta(true-predicted)":"hgs_combined_corrected_delta"}, inplace=True)

formula = (
    'hgs_combined_corrected_delta ~ '
    'C(treatment):C(disorder) + C(treatment):C(hgs_combined) + '
    'C(treatment):C(gender) + C(disorder):C(hgs_combined) + '
    'C(hgs_combined):C(gender) + '
    'C(treatment):C(disorder):C(hgs_combined) + '
    'C(treatment):C(hgs_combined):C(gender) + '
    'C(disorder):C(hgs_combined):C(gender) + '
    'C(treatment):C(disorder):C(hgs_combined):C(gender)'
    )

model = ols(formula, data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print(anova_table)
    # print("===== Done! =====")
    # embed(globals(), locals())
    # Perform post-hoc tests on significant interactions (Tukey's HSD)
interaction_groups =  df.matched_disorder_subgroup.astype(str) + "_" + df.hgs_combined.astype(str)
# interaction_groups =  b.disease_time.astype(str) + "_" + b.group.astype(str) + "_" + b.hgs_target.astype(str) + "_" + b.gender.astype(str)
comp = mc.MultiComparison(df["hgs_combined_corrected_delta"], interaction_groups)
post_hoc_res = comp.tukeyhsd()
print(post_hoc_res.summary())
    
print("===== Done! =====")
embed(globals(), locals())

