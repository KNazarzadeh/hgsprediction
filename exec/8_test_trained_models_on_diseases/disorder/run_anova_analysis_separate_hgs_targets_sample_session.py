import math
import sys
import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
from statsmodels.graphics.gofplots import qqplot
from itertools import product
from hgsprediction.load_results.load_disorder_matched_samples_results import load_disorder_matched_samples_results
from hgsprediction.save_results.save_disorder_anova_results import save_disorder_anova_results
from scipy import stats
from scipy.stats import zscore
import statsmodels.formula.api as smf
import researchpy as rp

import matplotlib.pyplot as plt
import seaborn as sns
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
##############################################################################
main_extracted_columns = ["gender", "handedness", "hgs_dominant", "hgs_dominant_side", "hgs_nondominant", "hgs_nondominant_side", "age", "bmi", "height", "waist_to_hip_ratio", "treatment", "disorder_episode", "hgs_target", "hgs", "hgs_predicted", "hgs_delta", "hgs_corrected_predicted", "hgs_corrected_delta"]

df_disorder = pd.DataFrame()
df_control = pd.DataFrame()

disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"
    
df_disorder_matched_female, df_mathced_controls_female = load_disorder_matched_samples_results(
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
    n_samples,
)

df_disorder_matched_male, df_mathced_controls_male = load_disorder_matched_samples_results(
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
    n_samples,
)

df_disorder_matched_female.loc[:, "treatment"] = f"{population}"
df_mathced_controls_female.loc[:, "treatment"] = "control"
df_disorder_matched_male.loc[:, "treatment"] = f"{population}"
df_mathced_controls_male.loc[:, "treatment"] = "control"

df_disorder_matched_female.loc[:, "hgs_target"] = target
df_mathced_controls_female.loc[:, "hgs_target"] = target
df_disorder_matched_male.loc[:, "hgs_target"] = target
df_mathced_controls_male.loc[:, "hgs_target"] = target

df_disorder_tmp = pd.concat([df_disorder_matched_female, df_disorder_matched_male], axis=0)
df_control_tmp = pd.concat([df_mathced_controls_female, df_mathced_controls_male], axis=0)

df_disorder_tmp.columns = [col.replace(f"{target}", "hgs") if f"{target}" in col else col for col in df_disorder_tmp.columns]    
df_control_tmp.columns = [col.replace(f"{target}", "hgs") if f"{target}" in col else col for col in df_control_tmp.columns]

# Replace values in the column
prefix_pre = f"1st_pre-{population}"
df_control_tmp.loc[:, f"{prefix_pre}_disorder_episode"] = df_control_tmp.loc[:, f"{prefix_pre}_disorder_episode"].replace({f"pre-{population}": "pre-control"})

prefix_post = f"1st_post-{population}"
df_control_tmp.loc[:, f"{prefix_post}_disorder_episode"] = df_control_tmp.loc[:, f"{prefix_post}_disorder_episode"].replace({f"post-{population}": "post-control"})


df_control_tmp.rename(columns=lambda x: x.replace("delta(true-predicted)", "delta") if "delta(true-predicted)" in x else x, inplace=True)

df_control_tmp = df_control_tmp.drop(columns=[f"1st_pre-{population}_age_range", f"1st_post-{population}_age_range"])

##############################################################################
for disorder_subgroup in [f"pre-{population}", f"post-{population}"]:
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}_"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}_"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}_"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}_"
        
    if disorder_subgroup == f"pre-{population}":
        df_control_extracted_pre = df_control_tmp[[col for col in df_control_tmp.columns if f"post-{population}" not in col]]        
        df_disorder_extracted_pre = df_disorder_tmp[[col for col in df_disorder_tmp.columns if f"post-{population}" not in col]]
        df_disorder_extracted_pre.loc[:, "disorder_episode"] = disorder_subgroup
        rename_columns = [col for col in df_disorder_extracted_pre.columns if prefix in col]
        # Remove the prefix from selected column names
        for col in rename_columns:
            new_col_name = col.replace(prefix, "")
            df_disorder_extracted_pre = df_disorder_extracted_pre.rename(columns={col: new_col_name})
        
        rename_columns = [col for col in df_control_extracted_pre.columns if prefix in col]
        # Remove the prefix from selected column names
        for col in rename_columns:
            new_col_name = col.replace(prefix, "")
            df_control_extracted_pre = df_control_extracted_pre.rename(columns={col: new_col_name})

        df_disorder_extracted_pre.rename(columns=lambda x: x.replace("delta(true-predicted)", "delta") if "delta(true-predicted)" in x else x, inplace=True)
        
    elif disorder_subgroup == f"post-{population}":
        df_control_extracted_post = df_control_tmp[[col for col in df_control_tmp.columns if f"pre-{population}" not in col]]
        df_disorder_extracted_post = df_disorder_tmp[[col for col in df_disorder_tmp.columns if f"pre-{population}" not in col]]
        df_disorder_extracted_post.loc[:, "disorder_episode"] = disorder_subgroup
        rename_columns = [col for col in df_disorder_extracted_post.columns if prefix in col]
        # Remove the prefix from selected column names
        for col in rename_columns:
            new_col_name = col.replace(prefix, "")
            df_disorder_extracted_post = df_disorder_extracted_post.rename(columns={col: new_col_name})
            
        rename_columns = [col for col in df_control_extracted_post.columns if prefix in col]
        # Remove the prefix from selected column names
        for col in rename_columns:
            new_col_name = col.replace(prefix, "")
            df_control_extracted_post = df_control_extracted_post.rename(columns={col: new_col_name})
            
        df_disorder_extracted_post.rename(columns=lambda x: x.replace("delta(true-predicted)", "delta") if "delta(true-predicted)" in x else x, inplace=True)

# Check if the indices are in the same order
if df_disorder_extracted_pre.index.equals(df_disorder_extracted_post.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")

# Check if the indices are in the same order
if df_control_extracted_pre.index.equals(df_control_extracted_post.index):
    print("The indices are in the same order.")
else:
    print("The indices are not in the same order.")

df_disorder_tmp2 = pd.concat([df_disorder_extracted_pre.loc[:, main_extracted_columns], df_disorder_extracted_post.loc[:, main_extracted_columns]], axis=0)
df_control_tmp2 = pd.concat([df_control_extracted_pre.loc[:, main_extracted_columns], df_control_extracted_post.loc[:, main_extracted_columns]], axis=0)

df_disorder = pd.concat([df_disorder, df_disorder_tmp2], axis=0)
df_control = pd.concat([df_control, df_control_tmp2], axis=0)

##############################################################################
# Perform the ANOVA
df = pd.concat([df_control, df_disorder], axis=0)
df.index.name = "SubjectID"
df["gender"].replace(0, "female", inplace=True)
df["gender"].replace(1, "male", inplace=True)

##############################################################################
for anova_target in ["hgs", "hgs_predicted", "hgs_delta", "hgs_corrected_predicted", "hgs_corrected_delta"]:
    data = df[["gender", "treatment", "disorder_episode", anova_target]]
    # Replace values based on conditions
    data.loc[data['disorder_episode'].str.contains('pre'), 'disorder_episode'] = 'pre'
    data.loc[data['disorder_episode'].str.contains('post'), 'disorder_episode'] = 'post'
    ############################## ASSUMPTION CHECK ##############################
    # Linear mixed effect models have the same assumptions as the traditional standard linear regression model. 

    ############################## NORMALITY CHECK ##############################
    # Q-Q plot:
    # Create a list of unique values for 'time' and 'stress'
    treatment_values = data['treatment'].unique()
    disorder_episode_values = data['disorder_episode'].unique()

    # Create a figure with subplots for each combination of 'time' and 'stress'
    fig, axes = plt.subplots(len(disorder_episode_values), len(treatment_values), figsize=(15, 10))

    # Iterate over each combination of 'time' and 'stress'
    for i, j in product(range(len(disorder_episode_values)), range(len(treatment_values))):
        # Subset the DataFrame for the current combination
        subset = data[(data['disorder_episode'] == disorder_episode_values[i]) & (data['treatment'] == treatment_values[j])]
        
        # Create a QQ plot for the 'score' column
        qqplot(subset[anova_target], line='s', ax=axes[i, j])
        
        # Set title for each subplot
        axes[i, j].set_title(f"disorder_episode: {disorder_episode_values[i]}, treatment: {treatment_values[j]}")
        
    # Adjust layout
    plt.tight_layout()
    plt.show()
    plt.savefig(f"QQplot_{population}_{anova_target}.png")
    ############################## NORMALITY CHECK ##############################
    ## KDE PLot gender separated
    data_tmp = df[["gender", "treatment", "disorder_episode", anova_target]]
    data_female = data_tmp[data_tmp["gender"]=="female"]
    data_male = data_tmp[data_tmp["gender"]=="male"]

    fig, ax = plt.subplots(1,2, figsize=(18,6))
    sns.kdeplot(data=data_female, x=anova_target, hue="disorder_episode", ax=ax[0], legend=False)
    sns.kdeplot(data=data_male, x=anova_target, hue="disorder_episode", ax=ax[1])
    sns.move_legend(ax[1], "upper left", bbox_to_anchor=(1, 1))

    # Add subtitles
    ax[0].set_title("Female")
    ax[1].set_title("Male")

    plt.show()
    plt.savefig(f"kde_gender_{population}_{anova_target}.png")
    plt.close()

    ## KDE PLot gender separated
    data_tmp = df[["gender", "treatment", "disorder_episode", anova_target]]
    fig, ax = plt.subplots(1,1, figsize=(18,6))
    sns.kdeplot(data=data_tmp, x=anova_target, hue="disorder_episode")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    # Add subtitles
    ax.set_title("Both gender together")

    plt.show()
    plt.savefig(f"kde_{population}_{anova_target}.png")
    plt.close()
    ############################## NORMALITY CHECK ##############################
    # Shapiro-Wilk test:
    # The normality assumption can be checked by computing Shapiro-Wilk test:
    def shapiro_test(x):
        a = 0.05
        test = stats.shapiro(x)
        if test.pvalue <= 0.05:
            return f'The distribution departed from normality significantly, W= {round(test.statistic,2)}, P value= {round(test.pvalue,2)}'
        else:
            return f"Shapiro Wilk Test result didn't show non-normality, W= {round(test.statistic,2)}, P value= {round(test.pvalue,2)}. There is no evidence to reject the null hypothesis of normality."
    treatment_values = data['treatment'].unique()
    disorder_episode_values = data['disorder_episode'].unique()
    for i, treat in enumerate(treatment_values):
        for j, epi in enumerate(disorder_episode_values):
            data_shapiro = data[(data["treatment"]==treat) & (data["disorder_episode"]==epi)][anova_target]
            print(f"For the treatment and episode: {treat} and {epi}:", shapiro_test(data_shapiro))
            print('\n')

    ############################## Homogneity of Variance CHECK ##############################
    # Perform Levene test for all samples together
    # Define a function to perform Levene's test for each group
    # def levene_test(group):
    #     return stats.levene(group[group['gender'] == 'male'][anova_target],
    #                   group[group['gender'] == 'female'][anova_target],
    #                   group[group['treatment'] == 'control'][anova_target],
    #                   group[group['treatment'] == population][anova_target])
    # # Group by 'time' and apply the Levene's test function
    # levene_results = data.groupby('disorder_episode').apply(levene_test)

    def levene_test(group):
        return stats.levene(
                    group[group['treatment'] == 'control'][anova_target],
                    group[group['treatment'] == population][anova_target])
    # Group by 'time' and apply the Levene's test function
    levene_results = data.groupby('disorder_episode').apply(levene_test)
    # Print the results
    print(levene_results)
    # If From the levene_results homogeneity test we can see p<0.05,
    # which means the variance level of the three groups are significantly different from each other, therefore we conclude, we can’t use ANOVA.
    # reject the null hypothesis that each group has the same variance. 

    print(rp.summary_cont(data.groupby(["treatment", "disorder_episode","gender"])[anova_target]))
##############################################################################
# # for anova_target in ["hgs_corrected_delta"]:
# formula = (
#     f'{anova_target} ~ '
#     'C(gender) + C(treatment) + C(disorder_episode) +'
#     'C(gender):C(treatment) + C(gender):C(disorder_episode) +'
#     'C(treatment):C(disorder_episode) +'
#     'C(gender):C(treatment):C(disorder_episode)'
# )

    # conduct 3-way ANOVA using mixedlm
    mixedlm_model_fit = smf.mixedlm(f"{anova_target} ~ treatment * gender * disorder_episode", data, groups=data.index).fit()
    # get fixed effects
    print(mixedlm_model_fit.summary())
    # get random effects
    mixedlm_model_fit.random_effects

    df_anova_result = pd.DataFrame(mixedlm_model_fit.summary().tables[1])

    # ##############################################################################
    # # 3-way analysis possible in ols, but random effects not accounted for!
    # my_model_fit = smf.ols(f"{anova_target} ~ treatment * gender * disorder_episode", data=data, groups=data.index).fit()
    # df_anova_result = sm.stats.anova_lm(my_model_fit, typ=2)
    # print(df_anova_result)

    # # model = ols(formula, data=df).fit()
    # # df_anova_result = sm.stats.anova_lm(model)
    # # print(df_anova_result)

    ##############################################################################
    # Perform post-hoc tests on significant interactions (Tukey's HSD)
    interaction_groups =  df.treatment.astype(str) + " | " + df.disorder_episode.astype(str)    
    comp = mc.MultiComparison(df[f"{anova_target}"], interaction_groups)
    df_post_hoc_result_without_gender = comp.tukeyhsd()
    print(df_post_hoc_result_without_gender.summary())

    interaction_groups =  df.treatment.astype(str) + " | " + df.gender.astype(str) + " | " + df.disorder_episode.astype(str)
    comp = mc.MultiComparison(df[f"{anova_target}"], interaction_groups)
    df_post_hoc_result_with_gender = comp.tukeyhsd()
    print(df_post_hoc_result_with_gender.summary())

    print(anova_target)
    print(target)
    print(population)
    # print("===== Done! =====")
    # embed(globals(), locals())
    save_disorder_anova_results(
            df,
            df_anova_result,
            df_post_hoc_result_without_gender,
            df_post_hoc_result_with_gender,
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
        )

print("===== Done! End =====")
embed(globals(), locals())

