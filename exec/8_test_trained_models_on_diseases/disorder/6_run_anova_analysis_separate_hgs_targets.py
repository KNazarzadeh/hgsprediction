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
from scipy.stats import levene, shapiro
from pingouin import mixed_anova

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

male_pre_controls = df[(df["gender"]=="male") & (df["condition"]=="pre-control")][anova_target]
# print(male_pre_controls.var())

female_pre_controls = df[(df["gender"]=="female") & (df["condition"]=="pre-control")][anova_target]
# print(female_pre_controls.var())

male_post_controls = df[(df["gender"]=="male") & (df["condition"]=="post-control")][anova_target]
# print(male_post_controls.var())

female_post_controls = df[(df["gender"]=="female") & (df["condition"]=="post-control")][anova_target]
# print(female_post_controls.var())


male_pre_patients = df[(df["gender"]=="male") & (df["condition"]==f"pre-{population}")][anova_target]
female_pre_patients = df[(df["gender"]=="female") & (df["condition"]==f"pre-{population}")][anova_target]
male_post_patients = df[(df["gender"]=="male") & (df["condition"]==f"post-{population}")][anova_target]
female_post_patients = df[(df["gender"]=="female") & (df["condition"]==f"post-{population}")][anova_target]
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
# Pingouin mixed_anova for female and male separately:
data = df[["gender", "treatment", "condition", anova_target]]
# Replace values based on conditions
data.loc[data['condition'].str.contains('pre-'), 'condition'] = 'pre'
data.loc[data['condition'].str.contains('post-'), 'condition'] = 'post'
data["Subject"] = data.index

df_female_tmp = data[data["gender"]=="female"]
df_male_tmp = data[data["gender"]=="male"]

###############################################################################
aov_female = mixed_anova(dv=anova_target, between='treatment', within='condition', subject='Subject', data=df_female_tmp)
aov_male = mixed_anova(dv=anova_target, between='treatment', within='condition', subject='Subject', data=df_male_tmp)

print("Female Pinguin ANOVA Result:")
print(aov_female)
save_anova_results(
    data,
    aov_female,
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
)
print("Male Pinguin ANOVA Result:")
print(aov_male)
save_anova_results(
    data,
    aov_male,
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
)
################################## Interpret ##################################
# General Conclusions:

# Disorder Episode is a significant factor affecting outcomes differently in both genders, particularly more impactful in males.
# Treatment itself does not significantly affect the outcome for either gender.
# Interaction between treatment and disorder episode is not significant in either gender, 
# indicating the effect of treatment is consistent across different levels of the disorder episode.

# This analysis indicates that interventions might need to be more focused on the disorder episode itself rather than the type of treatment, and 
# this might require different strategies for male and female groups considering the stronger effect in males.

###############################################################################
# Linear Mixed Models mixedlm for female and male separately:
mixedlm_formula = f"{anova_target} ~ treatment * condition"
mixedlm_model_fit_female = smf.mixedlm(formula=mixedlm_formula, data=df_female_tmp, groups="Subject").fit()
mixedlm_model_fit_male = smf.mixedlm(formula=mixedlm_formula, data=df_male_tmp, groups="Subject").fit()

import statsmodels.stats.multicomp as multi

# Combine the predictions with the original data for reference
df_female_tmp['pred'] = mixedlm_model_fit_female.fittedvalues
df_male_tmp['pred'] = mixedlm_model_fit_male.fittedvalues

# Perform pairwise comparisons for each group
# Note: Modify the code according to your specific levels in 'treatment' and 'condition'
tukey_hsd_female = multi.pairwise_tukeyhsd(endog=df_female_tmp['pred'], groups=df_female_tmp['treatment'] + "_" + df_female_tmp['condition'])
tukey_hsd_male = multi.pairwise_tukeyhsd(endog=df_male_tmp['pred'], groups=df_male_tmp['treatment'] + "_" + df_male_tmp['condition'])

print(tukey_hsd_female.summary())
print(tukey_hsd_male.summary())


print("===== Done! End =====")
embed(globals(), locals())
# get fixed effects
print("Female MixedLM ANOVA Result:")
print(mixedlm_model_fit_female.summary())
save_anova_results(
    data,
    mixedlm_model_fit_female,
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
    "mixedlm_gender_separated",
)
print("Male MixedLM ANOVA Result:")
print(mixedlm_model_fit_male.summary())
save_anova_results(
    data,
    mixedlm_model_fit_male,
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
    "mixedlm_gender_separated",
)
###############################################################################
# Linear Mixed Models mixedlm for female and male separately:
mixedlm_formula = f"{anova_target} ~ treatment * gender * condition"
mixedlm_model_fit = smf.mixedlm(formula=mixedlm_formula, data=data, groups="Subject").fit()

# get fixed effects
print("MixedLM ANOVA Result:")
print(mixedlm_model_fit.summary())

save_anova_results(
    data,
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
)
###############################################################################
print("===== Done! End =====")
embed(globals(), locals())