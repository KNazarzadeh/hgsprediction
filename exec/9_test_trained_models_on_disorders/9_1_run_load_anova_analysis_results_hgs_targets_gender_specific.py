import math
import sys
import os
import numpy as np
import pandas as pd

from hgsprediction.load_results.anova.load_disorder_posthoc_results import load_disorder_posthoc_results
from hgsprediction.load_results.anova.load_disorder_anova_results import load_disorder_anova_results

import pingouin
from pingouin import mixed_anova
import statsmodels.stats.multicomp as mc

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
# Pinguin Mixed ANOVA

df_female, anova_female = load_disorder_anova_results(
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
    "female",
    "pingouin",
    first_event,    
)

df_male, anova_male = load_disorder_anova_results(
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
    "male",
    "pingouin",
    first_event,
)

print("=================================================================================")
print("\n Female Pinguin ANOVA Result:")
# Applying 2 decimal format to the DataFrame
anova_female_df = anova_female.applymap(lambda x: "{:.3f}".format(x) if isinstance(x, (int, float)) else x)
print(anova_female_df)

print("#-----------------------------------------------------------#")

print("\n Male Pinguin ANOVA Result:")
# Applying 2 decimal format to the DataFrame
anova_male_df = anova_male.applymap(lambda x: "{:.3f}".format(x) if isinstance(x, (int, float)) else x)
print(anova_male_df)

print("=================================================================================")
################################################################################
print("\n Female Post-Hoc ANOVA Result:")
df_pairwise_posthoc_female, df_posthoc_summary_female = load_disorder_posthoc_results(
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
    "female",
    "pingouin",
    first_event,)
print("\n df_pairwise_posthoc_female:\n")
print(df_pairwise_posthoc_female)
print("\n df_posthoc_summary_female:\n")
print(df_posthoc_summary_female)
print("=================================================================================")
print("\n Male Post-Hoc ANOVA Result:")
df_pairwise_posthoc_male, df_posthoc_summary_male = load_disorder_posthoc_results(
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
    "male",
    "pingouin",
    first_event,)
print("\n df_pairwise_posthoc_male:\n")
print(df_pairwise_posthoc_male)
print("\n df_posthoc_summary_male:\n")
print(df_posthoc_summary_male)
################################################################################
print("===== Done! End =====")
embed(globals(), locals())