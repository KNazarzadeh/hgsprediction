import math
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM

from hgsprediction.load_results.load_prepared_data_for_anova import load_prepare_data_for_anova
from hgsprediction.save_results.save_anova_results import save_anova_results
from pingouin import mixed_anova
import statsmodels.stats.multicomp as multi
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
anova_target = sys.argv[12]
first_event = sys.argv[13]

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
df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)
# Replace values based on time_points
df.loc[df['time_point'].str.contains('pre-'), 'time_point'] = 'pre'
df.loc[df['time_point'].str.contains('post-'), 'time_point'] = 'post'

df["Subject"] = df.index

mixedlm_formula = f"{anova_target} ~ group * time_point"
mixedlm_model_fit = smf.mixedlm(formula=mixedlm_formula, data=df, groups="Subject").fit()
print("ANOVA Result:\n")
print(mixedlm_model_fit.summary())

interaction = df.group.astype(str) + " | " + df.time_point.astype(str)
comp = mc.MultiComparison(df[f"{anova_target}"], interaction)
df_post_hoc_result = comp.tukeyhsd()
print("\n Post-Hoc Result:\n")
print(df_post_hoc_result.summary())

print("===== Done! End =====")
embed(globals(), locals())
#################################################################################
# Linear Mixed Models mixedlm for Group, Gender and Time-point factors:

mixedlm_formula = f"{anova_target} ~ group * gender * time_point"
mixedlm_model_fit = smf.mixedlm(formula=mixedlm_formula, data=df, groups="Subject").fit()
print("ANOVA Result:\n")
print(mixedlm_model_fit.summary())

interaction = df.group.astype(str) + " | " + df.time_point.astype(str)
comp = mc.MultiComparison(df[f"{anova_target}"], interaction)
df_post_hoc_result = comp.tukeyhsd()
print("\n Post-Hoc Result:\n")
print(df_post_hoc_result.summary())

save_anova_results(
    df,
    mixedlm_model_fit,
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
    "non",
    "mixedlm_both_gender",
    first_event,
)


print("===== Done! End =====")
embed(globals(), locals())