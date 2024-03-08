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
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
disorder_cohort = sys.argv[10]
visit_session = sys.argv[11]
n_samples = sys.argv[12]

##############################################################################
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

df_disorder_matched_female.loc[:, "samples"] = f"{population}"
df_mathced_controls_female.loc[:, "samples"] = "control"
df_disorder_matched_male.loc[:, "samples"] = f"{population}"
df_mathced_controls_male.loc[:, "samples"] = "control"

df_disorder = pd.concat([df_disorder_matched_female, df_disorder_matched_male], axis=0)
df_control = pd.concat([df_mathced_controls_female, df_mathced_controls_male], axis=0)

##############################################################################
# Perform the ANOVA
main_extracted_columns = ["disorder", "gender", "matched_disorder_subgroup", "samples", f"{target}"]
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
        df_extracted = df_disorder[[col for col in df_disorder.columns if f"post-{population}" not in col]]

    elif disorder_subgroup == f"post-{population}":
        df_extracted = df_disorder[[col for col in df_disorder.columns if f"pre-{population}" not in col]]
    
    rename_columns = [col for col in df_extracted.columns if prefix in col]
    
    # Remove the prefix from selected column names
    for col in rename_columns:
        new_col_name = col.replace(prefix, "")
        df_extracted.rename(columns={col: new_col_name}, inplace=True)
        
    df_extracted.loc[:, "matched_disorder_subgroup"] = disorder_subgroup
    
    df_extracted_control = df_control[df_control["matched_disorder_subgroup"] == disorder_subgroup]
    
    df = pd.concat([df_extracted_control.loc[:, extract_columns], df_extracted.loc[:, extract_columns]], axis=0)
    
    df["gender"].replace(0, "female", inplace=True)
    df["gender"].replace(1, "male", inplace=True)
    
    if target == "hgs_L+R":
        df.rename(columns={f"{target}": "hgs_combined", f"{target}_corrected_delta(true-predicted)":"hgs_combined_corrected_delta"}, inplace=True)
        
        formula = (
            'hgs_combined_corrected_delta ~ '
            'C(samples):C(disorder) + C(samples):C(hgs_combined) + '
            'C(samples):C(gender) + C(disorder):C(hgs_combined) + '
            'C(hgs_combined):C(gender) + '
            'C(samples):C(disorder):C(hgs_combined) + '
            'C(samples):C(hgs_combined):C(gender) + '
            'C(disorder):C(hgs_combined):C(gender) + '
            'C(samples):C(disorder):C(hgs_combined):C(gender)'
            )
    else:
        formula = (
            f'{anova_target_extracted_columns} ~ '
            f'C(samples) + C(disorder) + C({target}) + C(gender) + '
            f'C(samples):C(disorder) + C(samples):C({target}) + '
            f'C(samples):C(gender) + C(disorder):C({target}) + '
            f'C(disorder):C(gender) + C({target}):C(gender) + '
            f'C(samples):C(disorder):C({target}) + '
            f'C(samples):C(disorder):C(gender) + '
            f'C(samples):C({target}):C(gender) + '
            f'C(disorder):C({target}):C(gender) + '
            f'C(samples):C(disorder):C({target}):C(gender)'
            )

    model = ols(formula, data=df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)

    print(anova_results)
    
    # Perform post-hoc tests on significant interactions (Tukey's HSD)
    interaction_groups =  df.matched_disorder_subgroup.astype(str) + "_"+ df.samples.astype(str)+ "_" + df.hgs_combined.astype(str)
    # interaction_groups =  b.disease_time.astype(str) + "_" + b.group.astype(str) + "_" + b.hgs_target.astype(str) + "_" + b.gender.astype(str)
    comp = mc.MultiComparison(df["hgs_combined_corrected_delta"], interaction_groups)
    post_hoc_res = comp.tukeyhsd()
    print(post_hoc_res.summary())
    
    print("===== Done! =====")
    embed(globals(), locals())

