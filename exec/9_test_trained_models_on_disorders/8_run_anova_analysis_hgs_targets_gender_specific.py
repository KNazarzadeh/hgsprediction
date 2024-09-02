import math
import sys
import os
import numpy as np
import pandas as pd

from hgsprediction.load_results.anova.load_prepared_data_for_anova import load_prepare_data_for_anova
from hgsprediction.save_results.anova.save_disorder_anova_results import save_disorder_anova_results
from hgsprediction.save_results.anova.save_disorder_posthoc_results import save_disorder_posthoc_results

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
# print("===== Done! End =====")
# embed(globals(), locals())
###############################################################################
# Scenario (A):
# In order to account for gender as a factor in ANOVA, individual analyses of variance (ANOVA) 
# must be conducted on males and females, given that each group satisfies all ANOVA criteria for 
# the treatment and condition (pre and post). In this case, ANOVA can be applied for smaller datasets 
# also because of the ANOVA assumptions (e.g., homogeneity of variance met separately in male and female data). 
# mixed ANOVA was conducted utilizing the Pingouin library.  
###############################################################################
# Pinguin Mixed ANOVA
aov_female = mixed_anova(dv=anova_target, between='group', within='time_point', subject='Subject', data=df_female)
aov_male = mixed_anova(dv=anova_target, between='group', within='time_point', subject='Subject', data=df_male)

save_disorder_anova_results(
    df_female,
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
    first_event,    
)

save_disorder_anova_results(
    df_male,
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
    first_event,
)


print("\n Female Pinguin ANOVA Result:")
# Applying 2 decimal format to the DataFrame
aov_female_df = aov_female.applymap(lambda x: "{:.3f}".format(x) if isinstance(x, (int, float)) else x)
print(aov_female_df)

print("#-----------------------------------------------------------#")

print("\n Male Pinguin ANOVA Result:")
# Applying 2 decimal format to the DataFrame
aov_male_df = aov_male.applymap(lambda x: "{:.3f}".format(x) if isinstance(x, (int, float)) else x)
print(aov_male_df)

print("=================================================================================")
################################################################################
# Perform post-hoc tests if the ANOVA is significant
interaction_female = aov_female[aov_female['Source'] == "Interaction"]
# Perform post-hoc tests
df_pairwise_posthoc_female = pingouin.pairwise_ttests(dv=anova_target, between='group', within='time_point', subject='Subject', data=df_female, padjust='bonf')
print(df_pairwise_posthoc_female)
print("#-----------------------------------------------------------#")
interaction_female_posthoc =  df_female.group.astype(str) + " | " + df_female.time_point.astype(str)
comp_female = mc.MultiComparison(df_female[f"{anova_target}"], interaction_female_posthoc)
df_posthoc_summary_female = comp_female.tukeyhsd()
print("\n Female Post-Hoc Result:\n")
print(df_posthoc_summary_female.summary())
#-----------------------------------------------------------#
save_disorder_posthoc_results(
    df_pairwise_posthoc_female,
    df_posthoc_summary_female,
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
if interaction_female['p-unc'].iloc[0] < 0.05:
    print(f"There are significant interactions for female {anova_target}")
else:
    print(f"Not significant interactions for {anova_target}")
    
print("=================================================================================")

interaction_male = aov_male[aov_male['Source'] == "Interaction"]

# Perform post-hoc tests
df_pairwise_posthoc_male = pingouin.pairwise_ttests(dv=anova_target, between='group', within='time_point', subject='Subject', data=df_male, padjust='bonf')
print(df_pairwise_posthoc_male)
print("#-----------------------------------------------------------#")
interaction_male_posthoc =  df_male.group.astype(str) + " | " + df_male.time_point.astype(str)
comp_male = mc.MultiComparison(df_male[f"{anova_target}"], interaction_male_posthoc)
df_posthoc_summary_male = comp_male.tukeyhsd()
print("\n Male Post-Hoc Result:\n")
print(df_posthoc_summary_male.summary())
#-----------------------------------------------------------#
save_disorder_posthoc_results(
    df_pairwise_posthoc_male,
    df_posthoc_summary_male,
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
if interaction_male['p-unc'].iloc[0] < 0.05:
    print(f"There are significant interactions for males {anova_target}")
else:
    print(f"Not significant interactions for {anova_target}")

################################################################################
print("===== Done! End =====")
embed(globals(), locals())