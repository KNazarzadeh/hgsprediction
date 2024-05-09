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

###############################################################################
# Scenario (A):
# Applying ANOVA to larger datasets with two factors (group and time_point (pre and post)) 
# while ignoring the gender factor (stroke and depression). 
# This is because I explained that ANOVA cannot be used when both genders are considered as a main factor. 
###############################################################################
if population == "parkinson":
    print("We cannot use ANOVA as parkinson data is small")
else:
    # Pingouin mixed_anova for female and male separately:
    data = df[["gender", "group", "time_point", anova_target]]
    # Replace values based on time_points
    data.loc[data['time_point'].str.contains('pre-'), 'time_point'] = 'pre'
    data.loc[data['time_point'].str.contains('post-'), 'time_point'] = 'post'
    data["Subject"] = data.index
    ###############################################################################
    aov = mixed_anova(dv=anova_target, between='group', within='time_point', subject='Subject', data=data)

    print("Pinguin ANOVA Result:")
    print(aov)
    # save_anova_results(
    #     data,
    #     aov,
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
    #     "non",
    #     "pingouin",
    # )
    ###############################################################################
    # Scenario (D): Sanity check:
    # Linear mixed model technique on the males and females separately as a sanity check 
    # to be compared with the analysis in (A). It was identified as two-way ANOVA with 
    # 1 between-group factor (group) and 1 within-group factor (pre- and post-time_point as repeated measures) using mixedlm(). 
    # The results of scenarios C and D were very similar. 
    # So, checking these two methods made me sure of the respective results.
    # Linear Mixed Models mixedlm for data ignoring gender:
    mixedlm_formula = f"{anova_target} ~ group * time_point"
    mixedlm_model_fit = smf.mixedlm(formula=mixedlm_formula, data=data, groups="Subject").fit()

    # get fixed effects
    print("MixedLM ANOVA Result:")
    print(mixedlm_model_fit.summary())

    # save_anova_results(
    #     data,
    #     mixedlm_model_fit,
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
    #     "non",
    #     "mixedlm_both_gender",
    # )
    print("===== Done! =====")
    embed(globals(), locals())
###############################################################################


# import statsmodels.stats.multicomp as multi

# # Combine the predictions with the original data for reference
# data['pred'] = mixedlm_model_fit.fittedvalues

# # Perform pairwise comparisons for each group
# # Note: Modify the code according to your specific levels in 'group' and 'time_point'
# tukey_hsd = multi.pairwise_tukeyhsd(endog=data['pred'], groups=data['group'] + "_" + data['time_point'])

# print(tukey_hsd.summary())


# import statsmodels.stats.multicomp as mc
# interaction_groups =  data.group.astype(str) + " | " + data.time_point.astype(str)
# comp = mc.MultiComparison(data[f"{anova_target}"], interaction_groups)
# df_post_hoc_result_without_gender = comp.tukeyhsd()
# print(df_post_hoc_result_without_gender.summary())
# print("===== Done! End =====")
# embed(globals(), locals())
# # General Conclusions:

# # Disorder Episode is a significant factor affecting outcomes differently in both genders, particularly more impactful in males.
# # group itself does not significantly affect the outcome for either gender.
# # Interaction between group and disorder episode is not significant in either gender, 
# # indicating the effect of group is consistent across different levels of the disorder episode.

# # This analysis indicates that interventions might need to be more focused on the disorder episode itself rather than the type of group, and 
# # this might require different strategies for male and female groups considering the stronger effect in males.
# print("===== Done! End =====")
# embed(globals(), locals())
# ###############################################################################
# # Linear Mixed Models mixedlm for female and male separately:
# mixedlm_formula = f"{anova_target} ~ group * time_point"
# mixedlm_model_fit_female = smf.mixedlm(formula=mixedlm_formula, data=df_female_tmp, groups="Subject").fit()
# mixedlm_model_fit_male = smf.mixedlm(formula=mixedlm_formula, data=df_male_tmp, groups="Subject").fit()

# import statsmodels.stats.multicomp as multi

# # Combine the predictions with the original data for reference
# df_female_tmp['pred'] = mixedlm_model_fit_female.fittedvalues
# df_male_tmp['pred'] = mixedlm_model_fit_male.fittedvalues

# # Perform pairwise comparisons for each group
# # Note: Modify the code according to your specific levels in 'group' and 'time_point'
# tukey_hsd_female = multi.pairwise_tukeyhsd(endog=df_female_tmp['pred'], groups=df_female_tmp['group'] + "_" + df_female_tmp['time_point'])
# tukey_hsd_male = multi.pairwise_tukeyhsd(endog=df_male_tmp['pred'], groups=df_male_tmp['group'] + "_" + df_male_tmp['time_point'])

# print(tukey_hsd_female.summary())
# print(tukey_hsd_male.summary())


# print("===== Done! End =====")
# embed(globals(), locals())
# # get fixed effects
# print("Female MixedLM ANOVA Result:")
# print(mixedlm_model_fit_female.summary())
# save_anova_results(
#     data,
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
# print("Male MixedLM ANOVA Result:")
# print(mixedlm_model_fit_male.summary())
# save_anova_results(
#     data,
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
# ###############################################################################
# # Linear Mixed Models mixedlm for female and male separately:
# mixedlm_formula = f"{anova_target} ~ group * time_point"
# mixedlm_model_fit = smf.mixedlm(formula=mixedlm_formula, data=data, groups="Subject").fit()

# # get fixed effects
# print("MixedLM ANOVA Result:")
# print(mixedlm_model_fit.summary())

# save_anova_results(
#     data,
#     mixedlm_model_fit,
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
#     "non",
#     "mixedlm_both_gender",
# )
# ###############################################################################
# # Scenario (C)
# ###############################################################################
# aov_female = mixed_anova(dv=anova_target, between='group', within='time_point', subject='Subject', data=df_female_tmp)
# aov_male = mixed_anova(dv=anova_target, between='group', within='time_point', subject='Subject', data=df_male_tmp)

# print("Female Pinguin ANOVA Result:")
# print(aov_female)
# save_anova_results(
#     data,
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
# print("Male Pinguin ANOVA Result:")
# print(aov_male)
# save_anova_results(
#     data,
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
# ################################## Interpret ##################################
# print("===== Done! End =====")
# embed(globals(), locals())