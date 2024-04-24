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

df_female = df[df["gender"]=="female"]
df_male = df[df["gender"]=="male"]


##############################################################################
data = df[["gender", "treatment", "disorder_episode", anova_target]]
data.loc[data['disorder_episode'].str.contains('pre'), 'disorder_episode'] = 'pre'
data.loc[data['disorder_episode'].str.contains('post'), 'disorder_episode'] = 'post'

df_pre = data[data["disorder_episode"]=="pre"]
df_post = data[data["disorder_episode"]=="post"]

df_pre["pre_hgs"] = df_pre[anova_target]
df_post["post_hgs"] = df_post[anova_target]

df_merge = df_pre.merge(df_post['post_hgs'], left_index=True, right_index=True, how='left')

df_merge = df_merge.drop(columns="disorder_episode")
print("===== Done! End =====")
embed(globals(), locals())
##############################################################################

from scipy.stats import wilcoxon, mannwhitneyu

# Wilcoxon Signed-Rank Test within each group
patients = df_merge[df_merge['treatment'] == 'stroke']
controls = df_merge[df_merge['treatment'] == 'control']

stat, p = wilcoxon(patients['pre_hgs'], patients['post_hgs'])
print("Patients - Wilcoxon Test: stat =", stat, "p-value =", p)

stat, p = wilcoxon(controls['pre_hgs'], controls['post_hgs'])
print("Controls - Wilcoxon Test: stat =", stat, "p-value =", p)

# Mann-Whitney U Test between groups at each time point
stat, p = mannwhitneyu(patients['pre_hgs'], controls['pre_hgs'])
print("Between Groups Before Diagnosis - Mann-Whitney U: stat =", stat, "p-value =", p)

stat, p = mannwhitneyu(patients['post_hgs'], controls['post_hgs'])
print("Between Groups After Diagnosis - Mann-Whitney U: stat =", stat, "p-value =", p)

print("===== Done! End =====")
embed(globals(), locals())