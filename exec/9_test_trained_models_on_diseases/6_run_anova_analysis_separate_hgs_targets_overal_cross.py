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
from scipy.stats import levene, shapiro, kstest
from pingouin import mixed_anova
from scipy import stats

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

##############################################################################
df_c = df[df['treatment']=='control']
df_c_pre = df_c[df_c['condition']=='pre-control']
df_c_post = df_c[df_c['condition']=='post-control']


df_dis = df[df['treatment']==f'{population}']
df_dis_pre = df_dis[df_dis['condition']==f'pre-{population}']
df_dis_post = df_dis[df_dis['condition']==f'post-{population}']


stat_c_dis_pre, p_value_c_dis_pre = levene(df_c_pre[anova_target], df_dis_pre[anova_target])
stat_c_dis_post, p_value_c_dis_post = levene(df_c_post[anova_target], df_dis_post[anova_target])
print("Homogeneity of variance between pre-control and pre-disease:", p_value_c_dis_pre)
print("Homogeneity of variance between post-control and post-disease:", p_value_c_dis_post)

###############################################################################
# Check Normality:
# Perform Shapiro-Wilk tests and print results
stat1, p_value1 = kstest(stats.norm.rvs(df_c_pre[anova_target]), stats.norm.cdf)
print("Shapiro-Wilk test for pre-HGS controls:", p_value1)

stat2, p_value2 = kstest(stats.norm.rvs(df_c_post[anova_target]), stats.norm.cdf)
print("Shapiro-Wilk test for post-HGS controls:", p_value2)

stat3, p_value3 = kstest(stats.norm.rvs(df_dis_pre[anova_target]), stats.norm.cdf)
print("Shapiro-Wilk test for pre-HGS patients:", p_value3)

stat4, p_value4 = kstest(stats.norm.rvs(df_dis_post[anova_target]), stats.norm.cdf)
print("Shapiro-Wilk test for post-HGS patients:", p_value4)

###############################################################################
fig, ax = plt.subplots(1, 2, figsize=(15, 8))
sns.kdeplot(data=df_c_pre, x=f"{anova_target}", fill=True, color="grey", ax=ax[0], label='Control')
sns.kdeplot(data=df_dis_pre, x=f"{anova_target}", fill=True, color="pink", ax=ax[0], label='disorder')

sns.kdeplot(data=df_c_post, x=f"{anova_target}", fill=True, color="grey", ax=ax[1], label='Control')
sns.kdeplot(data=df_dis_post, x=f"{anova_target}", fill=True, color="pink", ax=ax[1], label='disorder')
ax[0].set_xlabel("Pre-HGS")
ax[1].set_xlabel("Post-HGS")

plt.show()
plt.savefig(f"kde_{population}_{anova_target}.png")
plt.close()

###############################################################################
# Pingouin mixed_anova for female and male separately:
data = df[["gender", "treatment", "condition", anova_target]]
# Replace values based on conditions
data.loc[data['condition'].str.contains('pre-'), 'condition'] = 'pre'
data.loc[data['condition'].str.contains('post-'), 'condition'] = 'post'
data["Subject"] = data.index
###############################################################################
aov = mixed_anova(dv=anova_target, between='treatment', within='condition', subject='Subject', data=data)

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


import statsmodels.stats.multicomp as multi

# Combine the predictions with the original data for reference
data['pred'] = mixedlm_model_fit.fittedvalues

# Perform pairwise comparisons for each group
# Note: Modify the code according to your specific levels in 'treatment' and 'condition'
tukey_hsd = multi.pairwise_tukeyhsd(endog=data['pred'], groups=data['treatment'] + "_" + data['condition'])

print(tukey_hsd.summary())


import statsmodels.stats.multicomp as mc
interaction_groups =  data.treatment.astype(str) + " | " + data.condition.astype(str)
comp = mc.MultiComparison(data[f"{anova_target}"], interaction_groups)
df_post_hoc_result_without_gender = comp.tukeyhsd()
print(df_post_hoc_result_without_gender.summary())
print("===== Done! End =====")
embed(globals(), locals())