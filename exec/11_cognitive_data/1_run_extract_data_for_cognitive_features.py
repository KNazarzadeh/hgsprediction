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

from scipy.stats import pearsonr
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

###############################################################################
df_correlation_values, df_r2_values, df_mae_values = load_prediction_correlation_results(
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
    correlation_type,
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
    corr_value, p_value = pearsonr(df[feat], df[y])
    df_significant.loc[i, "feature_name"]=feat
    df_significant.loc[i,"corr-values"]= corr_value
    df_significant.loc[i,"p-values"]= p_value

print("===== Done! =====")
embed(globals(), locals())
df_sorted = df_significant.sort_values(by='corr-values', ascending=False)
###############################################################################
custom_paltte = ["#eb0917", "#86AD21", "#5ACACA", "#B382D6"]
# Plot the significance values using Seaborn
plt.figure(figsize=(20,30))
plt.rcParams.update({"font.weight": "bold", 
                    "axes.labelweight": "bold",
                    "ytick.labelsize": 25,
                    "xtick.labelsize": 25})
ax = sns.barplot(x='significance', y='feature_name', data=df_sorted, hue="cognitive_type", palette=custom_paltte, width=0.5)
# Add bar labels
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=3, fontsize=25, color='black')

plt.xlabel('-log(p-value)', weight="bold", fontsize=30)
plt.ylabel('')
plt.xticks(range(0, 25, 5))

plt.title(f'non-MRI Controls (N={len(df)})', weight="bold", fontsize=30)

# Place legend outside the plot
plt.legend(fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

plt.show()
plt.savefig(f"both_gender_cognitive_{target}.png")
plt.close()
