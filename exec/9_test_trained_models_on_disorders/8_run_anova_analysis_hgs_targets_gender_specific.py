import math
import sys
import os
import numpy as np
import pandas as pd

from hgsprediction.load_results.anova.load_prepared_data_for_anova import load_prepare_data_for_anova
from hgsprediction.save_results.anova.save_anova_results import save_anova_results
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

df["Subject"] = df.index

df_female = df[df["gender"]=="female"]
df_male = df[df["gender"]=="male"]
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# Scenario (C):
# In order to account for gender as a factor in ANOVA, individual analyses of variance (ANOVA) 
# must be conducted on males and females, given that each group satisfies all ANOVA criteria for 
# the treatment and condition (pre and post). In this case, ANOVA can be applied for smaller datasets 
# also because of the ANOVA assumptions (e.g., homogeneity of variance met separately in male and female data). 
# mixed ANOVA was conducted utilizing the Pingouin library.  
###############################################################################
# Pinguin Mixed ANOVA
aov_female = mixed_anova(dv=anova_target, between='group', within='time_point', subject='Subject', data=df_female)
aov_male = mixed_anova(dv=anova_target, between='group', within='time_point', subject='Subject', data=df_male)

print("Female Pinguin ANOVA Result:")
print(aov_female)
# save_anova_results(
#     df_female,
#     aov_female,
#     population,
#     mri_status,
#     session_column,
#     model_name,
#     feature_type,
#     target,
#     confound_status,
#     n_repeats,
#     n_folds,
#     n_samples,
#     anova_target,
#     "female",
#     "pingouin",
# )
print("\n Male Pinguin ANOVA Result:")
print(aov_male)
# save_anova_results(
#     df_male,
#     aov_male,
#     population,
#     mri_status,
#     session_column,
#     model_name,
#     feature_type,
#     target,
#     confound_status,
#     n_repeats,
#     n_folds,
#     n_samples,
#     anova_target,
#     "male",
#     "pingouin",
# )
print("=================================================================================")
# ################################################################################
# Linear Mixed Models mixedlm for female and male separately:
mixedlm_formula = f"{anova_target} ~ group * time_point"
mixedlm_model_fit_female = smf.mixedlm(formula=mixedlm_formula, data=df_female, groups="Subject").fit()
mixedlm_model_fit_male = smf.mixedlm(formula=mixedlm_formula, data=df_male, groups="Subject").fit()
# get fixed effects
print("\n Female MixedLM ANOVA Result:")
print(mixedlm_model_fit_female.summary())
# save_anova_results(
#     df_female,
#     mixedlm_model_fit_female,
#     population,
#     mri_status,
#     session_column,
#     model_name,
#     feature_type,
#     target,
#     confound_status,
#     n_repeats,
#     n_folds,
#     n_samples,
#     anova_target,
#     "female",
#     "mixedlm_gender_separated",
# )
print("\n Male MixedLM ANOVA Result:")
print(mixedlm_model_fit_male.summary())
# save_anova_results(
#     df_female,
#     mixedlm_model_fit_male,
#     population,
#     mri_status,
#     session_column,
#     model_name,
#     feature_type,
#     target,
#     confound_status,
#     n_repeats,
#     n_folds,
#     n_samples,
#     anova_target,
#     "male",
#     "mixedlm_gender_separated",
# )

################################################################################
interaction_female =  df_female.group.astype(str) + " | " + df_female.time_point.astype(str)
comp_female = mc.MultiComparison(df_female[f"{anova_target}"], interaction_female)
df_post_hoc_result_female = comp_female.tukeyhsd()
print("\n Female Post-Hoc Result:\n")
print(df_post_hoc_result_female.summary())

interaction_male =  df_male.group.astype(str) + " | " + df_male.time_point.astype(str)
comp_male = mc.MultiComparison(df_male[f"{anova_target}"], interaction_male)
df_post_hoc_result_male = comp_male.tukeyhsd()
print("\n Male Post-Hoc Result:\n")
print(df_post_hoc_result_male.summary())

print("===== Done! End =====")
embed(globals(), locals())