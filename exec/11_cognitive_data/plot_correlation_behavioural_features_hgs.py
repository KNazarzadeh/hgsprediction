#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""
import os
import sys
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

from hgsprediction.load_results.healthy.load_corrected_prediction_results import load_corrected_prediction_results
from hgsprediction.load_results.healthy.load_prediction_correlation_results import load_prediction_correlation_results
####### Features Extraction #######
from hgsprediction.define_features import define_features

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection
#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
data_set = sys.argv[10]
gender = sys.argv[11]
correlation_type = sys.argv[12]
corr_target = sys.argv[13]

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    gender,
    session,
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)
##############################################################################
# Define main features and extra features:
features, extend_features = define_features(feature_type)

###############################################################################
if corr_target == "true_hgs":
    y = target
elif corr_target == "hgs_corrected_delta":
    y = f"{target}_corrected_delta(true-predicted)"
# print("===== Done! =====")
# embed(globals(), locals())    
###############################################################################
# Collect all p-values and corresponding cognitive features
df_significant = pd.DataFrame(columns=["feature_name", "corr-values", "p-values"])

for i, feat in enumerate(features):
    if correlation_type == "pearson":
        corr_value, p_value = pearsonr(df[feat], df[y])
    elif correlation_type == "spearman":
        corr_value, p_value = spearmanr(df[feat], df[y])
    df_significant.loc[i, "feature_name"]=feat
    df_significant.loc[i,"corr-values"]= corr_value
    df_significant.loc[i,"p-values"]= p_value
###############################################################################
cognitive_features = ["Fluid intelligence",
                    "Reaction time",
                    "Numeric memory",
                    "Trail making: part A",
                    "Trail making: part B",
                    "Pairs matching 1: error made",
                    "Pairs matching 2: error made",
                    "Pairs matching 1: time",
                    "Pairs matching 2: time",
                    "Prospective memory",
                    "Symbol-digit matching: corrected",
                    "Symbol-digit matching: attempted",
                    ]

depression_anxiety_features = [
                    "Neuroticism", 
                    "Anxiety symptom", 
                    "Depression symptom",
                    "CIDI-depression",
                    ]

life_satisfaction_features = [
                    "Happiness",
                    "Satisfaction: family relationship",
                    "Satisfaction: job/work",
                    "Satisfaction: health",
                    "Satisfaction:friendship",
                    "Satisfaction: financial situation",
                    ]

well_being_features = [
                    "Happiness: general",
                    "Happiness with own health",
                    "Belief that life is meaningful"
                    ]
###############################################################################
# print("===== Done! =====")
# embed(globals(), locals())
df_significant['feature_name'] = df_significant['feature_name'].replace(
                        {"fluid_intelligence":"Fluid intelligence",
                        "reaction_time":"Reaction time",
                        "numeric_memory_Max_digits":"Numeric memory",
                        "trail_making_duration_numeric":"Trail making: part A",
                        "trail_making_duration_alphanumeric":"Trail making: part B",
                        "pairs_matching_incorrected_number_3pairs":"Pairs matching 1: error made",
                        "pairs_matching_incorrected_number_6pairs":"Pairs matching 2: error made",
                        "pairs_matching_completed_time_3pairs":"Pairs matching 1: time",
                        "pairs_matching_completed_time_6pairs":"Pairs matching 2: time",
                        "prospective_memory":"Prospective memory",
                        "symbol_digit_matches_corrected":"Symbol-digit matching: corrected",
                        "symbol_digit_matches_attempted":"Symbol-digit matching: attempted",
                        "happiness":"Happiness",
                        "family_satisfaction":"Satisfaction: family relationship",
                        "job_satisfaction":"Satisfaction: job/work",
                        "health_satisfaction":"Satisfaction: health",
                        "friendship_satisfaction":"Satisfaction:friendship",
                        "financial_satisfaction":"Satisfaction: financial situation",
                        "neuroticism_score":"Neuroticism", 
                        "anxiety_score":"Anxiety symptom", 
                        "depression_score":"Depression symptom",
                        "CIDI_score":"CIDI-depression",
                        "general_happiness":"Happiness: general",
                        "health_happiness":"Happiness with own health",
                        "belief_life_meaningful":"Belief that life is meaningful"})
###############################################################################
idx = df_significant[df_significant['feature_name'].isin(cognitive_features)].index
df_significant.loc[idx, "cognitive_type"] = "Cognitive function"
idx = df_significant[df_significant['feature_name'].isin(depression_anxiety_features)].index
df_significant.loc[idx, "cognitive_type"] = "Depression/Anxiety"
idx = df_significant[df_significant['feature_name'].isin(life_satisfaction_features)].index
df_significant.loc[idx, "cognitive_type"] = "Life satisfaction"
idx = df_significant[df_significant['feature_name'].isin(well_being_features)].index
df_significant.loc[idx, "cognitive_type"] = "Subjective well-being"

###############################################################################
df_sorted = df_significant.sort_values(by='corr-values', ascending=False)
###############################################################################
custom_paltte = ["#eb0917", "#86AD21", "#5ACACA", "#B382D6"]
###############################################################################
# Specify the path of the new folder
folder_path = os.path.join("plots", correlation_type , f"{data_set}")
# Check if the directory already exists
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
##############################################################################
file_path = os.path.join(folder_path, f"{data_set}_{gender}_behavioural_correlation_with_{corr_target}.png")
###############################################################################
# Plot the significance values using Seaborn
plt.figure(figsize=(45,30))
plt.rcParams.update({"ytick.labelsize": 30,
                    "xtick.labelsize": 30})
ax = sns.barplot(x='corr-values', y='feature_name', data=df_sorted, hue="cognitive_type", palette=custom_paltte, width=0.5)
# Add bar labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=28, color='black', weight='bold')

plt.xlabel('Correlations', fontsize=30)
plt.ylabel('')

if data_set == 'mri_set':
    if corr_target == "true_hgs":
        plt.title(f'{gender.capitalize()} - MRI dataset (N={len(df)}) correlations with True HGS', weight="bold", fontsize=30)
    elif corr_target == "hgs_corrected_delta":
        plt.title(f'{gender.capitalize()} - MRI dataset (N={len(df)}) correlations with Delta HGS', weight="bold", fontsize=30)
elif data_set == 'holdout_test_set':
    if corr_target == "true_hgs":
        plt.title(f'{gender.capitalize()} - Holdout Test dataset (N={len(df)}) correlations with True HGS', weight="bold", fontsize=30)
    elif corr_target == "hgs_corrected_delta":
        plt.title(f'{gender.capitalize()} - Holdout Test dataset (N={len(df)}) correlations with Delta HGS', weight="bold", fontsize=30)

# Place legend outside the plot
plt.legend(fontsize='22', bbox_to_anchor=(1.005, 1), loc='upper left')

plt.tight_layout()

plt.show()
plt.savefig(file_path)
plt.close()

print("===== Done! =====")
embed(globals(), locals())