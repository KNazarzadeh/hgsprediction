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
correlation_type = sys.argv[11]
corr_target = sys.argv[12]
feature = sys.argv[13]

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
df_female = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)

df_male = load_corrected_prediction_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,
    data_set,
)
###############################################################################
if corr_target == "true_hgs":
    y = target
elif corr_target == "hgs_corrected_delta":
    y = f"{target}_corrected_delta(true-predicted)"
# print("===== Done! =====")
# embed(globals(), locals())    
###############################################################################
# Collect all p-values and corresponding cognitive features
if correlation_type == "pearson":
    corr_value_female, p_value_female = pearsonr(df_female[feature], df_female[y])
    corr_value_male, p_value_male = pearsonr(df_male[feature], df_male[y])

elif correlation_type == "spearman":
    corr_value_female, p_value_female = spearmanr(df_female[feature], df_female[y])
    corr_value_male, p_value_male = spearmanr(df_male[feature], df_male[y])

###############################################################################
# Specify the path of the new folder
folder_path = os.path.join("plots", correlation_type , f"{data_set}",f"{feature}")
# Check if the directory already exists
if(not os.path.isdir(folder_path)):
        os.makedirs(folder_path)
##############################################################################
file_path = os.path.join(folder_path, f"{data_set}_{feature}_correlation_with_{corr_target}.png")
###############################################################################
palette_male = sns.color_palette("dark")
palette_female = sns.color_palette("flare")
color_female = palette_female[2]
color_male = palette_male[9]
###############################################################################
# Plot the significance values using Seaborn
plt.rcParams.update({
    "ytick.labelsize": 25,
    "xtick.labelsize": 25,
})
#-----------------------------------------------------------#
fig, ax = plt.subplots(1,2, figsize=(24,12))

sns.regplot(x=feature, y=y, data=df_male, color=color_male, line_kws={"color": 'darkgrey', "linewidth": 7}, ax=ax[0])
sns.regplot(x=feature, y=y, data=df_female, color=color_female, line_kws={"color": 'darkgrey', "linewidth": 7}, ax=ax[1])
#-----------------------------------------------------------#
ax[0].set_xlabel(f"{feature.capitalize()}", fontsize=30)
ax[1].set_xlabel(f"{feature.capitalize()}", fontsize=30)
#-----------------------------------------------------------#
if corr_target == "true_hgs":
    ax[0].set_ylabel("True HGS", fontsize=30)
    ax[1].set_ylabel("True HGS", fontsize=30)
elif corr_target == "hgs_corrected_delta":
    ax[0].set_ylabel("$\Delta$ HGS", fontsize=30)
    ax[1].set_ylabel("$\Delta$ HGS", fontsize=30)
#-----------------------------------------------------------#    
# Set titles depending on the dataset and correlation target
if data_set == 'mri_set':
    if corr_target == "true_hgs":
        ax[0].set_title(f'Male - MRI dataset (N={len(df_male)}) correlations with True HGS', weight="bold", fontsize=18)
        ax[1].set_title(f'Female - MRI dataset (N={len(df_female)}) correlations with True HGS', weight="bold", fontsize=18)
        
    elif corr_target == "hgs_corrected_delta":
        ax[0].set_title(f'Male - MRI dataset (N={len(df_male)}) correlations with Delta HGS', weight="bold", fontsize=18)
        ax[1].set_title(f'Female - MRI dataset (N={len(df_female)}) correlations with Delta HGS', weight="bold", fontsize=18)

elif data_set == 'holdout_test_set':
    if corr_target == "true_hgs":
        ax[0].set_title(f'Male - Holdout test dataset (N={len(df_male)}) correlations with True HGS', weight="bold", fontsize=18)
        ax[1].set_title(f'Female - Holdout test dataset (N={len(df_female)}) correlations with True HGS', weight="bold", fontsize=18)
        
    elif corr_target == "hgs_corrected_delta":
        ax[0].set_title(f'Male - Holdout test dataset (N={len(df_male)}) correlations with Delta HGS', weight="bold", fontsize=18)
        ax[1].set_title(f'Female - Holdout test dataset (N={len(df_female)}) correlations with Delta HGS', weight="bold", fontsize=18)
#-----------------------------------------------------------#            
# Annotation for female data in the first subplot
r_text_male = f"r = {corr_value_male:.2f}"
p_text_male = r"$p = {0:.3f}$".format(p_value_male)  # p in math mode

r_text_female = f"r = {corr_value_female:.2f}"
p_text_female = r"$p = {0:.3f}$".format(p_value_female)  # p in math mode

# Annotations for male and female data in the fourth subplot
ax[0].annotate(f"{r_text_male}\n{p_text_male}" , xy=(0.025, 0.93), xycoords='axes fraction', fontsize=20,
                color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))

ax[1].annotate(f"{r_text_female}\n{p_text_female}", xy=(0.025, 0.93), xycoords='axes fraction', fontsize=20,
                color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
plt.tight_layout()

plt.show()
plt.savefig(file_path)
plt.close()

print("===== Done! =====")
embed(globals(), locals())