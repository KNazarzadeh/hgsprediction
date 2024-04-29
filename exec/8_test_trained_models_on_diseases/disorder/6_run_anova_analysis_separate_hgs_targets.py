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
from scipy.stats import levene, shapiro

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

male_pre_controls = df[(df["gender"]=="male") & (df["disorder_episode"]=="pre-control")][anova_target]
# print(male_pre_controls.var())

female_pre_controls = df[(df["gender"]=="female") & (df["disorder_episode"]=="pre-control")][anova_target]
# print(female_pre_controls.var())

male_post_controls = df[(df["gender"]=="male") & (df["disorder_episode"]=="post-control")][anova_target]
# print(male_post_controls.var())

female_post_controls = df[(df["gender"]=="female") & (df["disorder_episode"]=="post-control")][anova_target]
# print(female_post_controls.var())


male_pre_patients = df[(df["gender"]=="male") & (df["disorder_episode"]==f"pre-{population}")][anova_target]
female_pre_patients = df[(df["gender"]=="female") & (df["disorder_episode"]==f"pre-{population}")][anova_target]
male_post_patients = df[(df["gender"]=="male") & (df["disorder_episode"]==f"post-{population}")][anova_target]
female_post_patients = df[(df["gender"]=="female") & (df["disorder_episode"]==f"post-{population}")][anova_target]
# print("===== Done! End =====")
# embed(globals(), locals())
##############################################################################
# Between male controls and female controls (pre-condition)
stat1, p_value1 = levene(male_pre_controls, female_pre_controls)
print("Levene's test between male and female controls (pre-condition):", p_value1)

# Between male controls and female controls (post-condition)
stat2, p_value2 = levene(male_post_controls, female_post_controls)
print("Levene's test between male and female controls (post-condition):", p_value2)

# Between male patients and female patients (pre-condition)
stat3, p_value3 = levene(male_pre_patients, female_pre_patients)
print("Levene's test between male and female patients (pre-condition):", p_value3)

# Between male patients and female patients (post-condition)
stat4, p_value4 = levene(male_post_patients, female_post_patients)
print("Levene's test between male and female patients (post-condition):", p_value4)

# Between pre-condition and post-condition for male controls
stat5, p_value5 = levene(male_pre_controls, male_pre_patients)
print("Levene's test between male controls and male patients pre condition:", p_value5)

# Between pre-condition and post-condition for female controls
stat6, p_value6 = levene(female_pre_controls, female_pre_patients)
print("Levene's test between female controls and female patients pre condition:", p_value6)

# Between pre-condition and post-condition for male patients
stat7, p_value7 = levene(male_post_controls, male_post_patients)
print("Levene's test between male controls and male patients post condition:", p_value7)

# Between pre-condition and post-condition for female patients
stat8, p_value8 = levene(female_post_controls, female_post_patients)
print("Levene's test between female controls and female patients post condition:", p_value8)

###############################################################################
# Check Normality:

# Perform Shapiro-Wilk tests and print results
stat1, p_value1 = shapiro(male_pre_controls)
print("Shapiro-Wilk test for male pre-HGS controls:", p_value1)

stat2, p_value2 = shapiro(male_post_controls)
print("Shapiro-Wilk test for male post-HGS controls:", p_value2)

# I notice there seems to be a mistake with your repeated request, adjusting to check pre-HGS control against pre-HGS patient as well
stat3, p_value3 = shapiro(female_pre_controls)
print("Shapiro-Wilk test for female pre-HGS controls:", p_value3)

stat4, p_value4 = shapiro(female_post_controls)
print("Shapiro-Wilk test for female post-HGS controls:", p_value4)

# Perform Shapiro-Wilk tests and print results
stat1, p_value1 = shapiro(male_pre_patients)
print("Shapiro-Wilk test for male pre-HGS patients:", p_value1)

stat2, p_value2 = shapiro(male_post_patients)
print("Shapiro-Wilk test for male post-HGS patients:", p_value2)

# I notice there seems to be a mistake with your repeated request, adjusting to check pre-HGS control against pre-HGS patient as well
stat3, p_value3 = shapiro(female_pre_patients)
print("Shapiro-Wilk test for female pre-HGS patients:", p_value3)

stat4, p_value4 = shapiro(female_post_patients)
print("Shapiro-Wilk test for female post-HGS patients:", p_value4)



###############################################################################
# Mixedlm for female and male separately:

df.loc[df['disorder_episode'].str.contains('pre'), 'disorder_episode'] = 'pre'
df.loc[df['disorder_episode'].str.contains('post'), 'disorder_episode'] = 'post'
df_female = df[df["gender"]=="female"]
df_male = df[df["gender"]=="male"]

# conduct ANOVA using mixedlm for males
df_male_tmp = df_male[["treatment", "disorder_episode", anova_target]]
df_female_tmp = df_male[["treatment", "disorder_episode", anova_target]]
my_model_fit_male = smf.mixedlm("hgs ~ treatment * disorder_episode", df_male_tmp, groups=df_male_tmp.index).fit()
# get fixed effects
print("Male Mixedlm:")
print(my_model_fit_male.summary())

# conduct ANOVA using mixedlm for females
df_male = df_male[["treatment", "disorder_episode", "hgs"]]
my_model_fit_female = smf.mixedlm("hgs ~ treatment * disorder_episode", df_female_tmp, groups=df_female_tmp.index).fit()
# get fixed effects
print("Female Mixedlm:")
print(my_model_fit_female.summary())

###############################################################################


print("===== Done! End =====")
embed(globals(), locals())