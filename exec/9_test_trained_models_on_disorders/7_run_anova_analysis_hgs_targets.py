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
    
    save_anova_results(
        data,
        aov,
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
        "pingouin",
    )
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

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Find rows where column 'A' < 0.05 and return values from column 'B'
# if aov[aov['Source']=='Interaction'] < 0.05:  # Assuming you're checking the uncorrected p-value
data['interaction_groups']  =  data.group.astype(str) + " | " + data.time_point.astype(str)
comp = mc.MultiComparison(data[f"{anova_target}"], data['interaction_groups'])
post_hoc_result_without_gender = comp.tukeyhsd()
print(post_hoc_result_without_gender.summary())

    # save_post_hoc_results(
    #     post_hoc_result_without_gender,
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
        
    # )

print("===== Done! End =====")
embed(globals(), locals())

# ###############################################################################
