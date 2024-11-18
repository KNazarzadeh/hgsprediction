import sys
import os
import numpy as np
import pandas as pd
import math
import scipy.stats as stats
from scipy.stats import ttest_ind
from hgsprediction.load_results.disorder.load_disorder_corrected_prediction_results import load_disorder_corrected_prediction_results

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
firts_event = sys.argv[12]
anova_target = sys.argv[13]
##############################################################################
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

##############################################################################
# Load data for ANOVA
df_disorder_female = load_disorder_corrected_prediction_results(
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
    firts_event,
)

df_disorder_male = load_disorder_corrected_prediction_results(
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
    firts_event,
)

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Check normality of the distributions
male_normal = stats.shapiro(df_disorder_male["1st_post-stroke_years"]).pvalue > 0.05
female_normal = stats.shapiro(df_disorder_female["1st_post-stroke_years"]).pvalue > 0.05

print(f"Normality test (Male): {male_normal}, Normality test (Female): {female_normal}")

# Levene's test for equality of variances
levene_pvalue = stats.levene(df_disorder_male["1st_post-stroke_years"], df_disorder_female["1st_post-stroke_years"]).pvalue
equal_variance = levene_pvalue > 0.05

print(f"Levene's test p-value: {levene_pvalue}, Equal variance: {equal_variance}")
print("===== Done! End =====")
embed(globals(), locals())
###############################################################################

stat, p_value = stats.mannwhitneyu(df_disorder_male["1st_post-stroke_years"], df_disorder_female["1st_post-stroke_years"], alternative='two-sided')

###############################################################################
print("===== Done! End =====")
embed(globals(), locals())