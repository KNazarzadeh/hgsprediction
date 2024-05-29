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
from hgsprediction.load_results.healthy.load_corrected_prediction_correlation_results import load_corrected_prediction_correlation_results

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
df_female_correlation_values, df_female_r2_values = load_corrected_prediction_correlation_results(
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
    correlation_type,
    data_set,
)

df_male_correlation_values, df_male_r2_values = load_corrected_prediction_correlation_results(
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
    correlation_type,
    data_set,
)

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
plot_folder = os.path.join(os.getcwd(), f"plots/with_vs_withou_bias_hgs/scatterplot/{target}/{n_repeats}_repeats_{n_folds}_folds/{correlation_type}")
if(not os.path.isdir(plot_folder)):
        os.makedirs(plot_folder)
plot_file = os.path.join(plot_folder, f"comparing_with_vs_withou_bias_hgs_{target}.png")
###############################################################################
# Create a custom color palette dictionary
# Define custom palettes
palette_male = sns.color_palette("Blues")
palette_female = sns.cubehelix_palette()
color_female = palette_female[1]
color_male = palette_male[2]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

sns.set_style("white")

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

sns.regplot(data=df_female, x=f"{target}", y=f"{target}_predicted", color=color_female, marker="$\circ$", scatter_kws={'s': 20, 'linewidths': 2}, line_kws={"color": color_female}, ax=ax[0][0])
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_predicted", color=color_male, marker="$\circ$", scatter_kws={'s': 20, 'linewidths': 2}, line_kws={"color": color_male}, ax=ax[0][0])


sns.regplot(data=df_female, x=f"{target}", y=f"{target}_corrected_predicted", color=color_female, marker="$\circ$", scatter_kws={'s': 20, 'linewidths': 2}, line_kws={"color": color_female}, ax=ax[0][1])
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_corrected_predicted", color=color_male, marker="$\circ$", scatter_kws={'s': 20, 'linewidths': 2}, line_kws={"color": color_male}, ax=ax[0][1])


sns.regplot(data=df_female, x=f"{target}", y=f"{target}_delta(true-predicted)", color=color_female, marker="$\circ$", scatter_kws={'s': 20, 'linewidths': 2}, line_kws={"color": color_female}, ax=ax[1][0])
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_delta(true-predicted)", color=color_male, marker="$\circ$", scatter_kws={'s': 20, 'linewidths': 2}, line_kws={"color": color_male}, ax=ax[1][0])


sns.regplot(data=df_female, x=f"{target}", y=f"{target}_corrected_delta(true-predicted)", color=color_female, marker="$\circ$", scatter_kws={'s': 20, 'linewidths': 2}, line_kws={"color": color_female}, ax=ax[1][1])
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_corrected_delta(true-predicted)", color=color_male, marker="$\circ$", scatter_kws={'s': 20, 'linewidths': 2}, line_kws={"color": color_male}, ax=ax[1][1])


ax[0][0].set_ylabel("Predicted HGS", fontsize=16)
ax[1][0].set_ylabel("Delta HGS", fontsize=16)
ax[0][1].set_ylabel("Predicted HGS", fontsize=16)
ax[1][1].set_ylabel("Delta HGS", fontsize=16)

ax[0][0].set_xlabel("") 
ax[0][1].set_xlabel("") 
ax[1][0].set_xlabel("Raw HGS", fontsize=16) 
ax[1][1].set_xlabel("Raw HGS", fontsize=16)     

                   
ax[0][0].set_title(f"Without bias-adjustment", fontsize=16)            
ax[0][1].set_title(f"With bias-adjustment", fontsize=16)

            

# r_text_female = f"r:{df_female_correlations.loc[fold, 'r_values_true_delta']:.3f}\nR2:{df_female_correlations.loc[fold, 'r2_values_true_delta']:.3f}"
# r_text_male = f"r:{df_male_correlations.loc[fold, 'r_values_true_delta']:.3f}\nR2:{df_male_correlations.loc[fold, 'r2_values_true_delta']:.3f}"
# ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color=color_female)
# ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color=color_male)

# r_text_female = f"r:{df_female_correlations.loc[fold, 'r_values_true_corrected_delta']:.3f}\nR2:{df_female_correlations.loc[fold, 'r2_values_true_corrected_delta']:.3f}"
# r_text_male = f"r:{df_male_correlations.loc[fold, 'r_values_true_corrected_delta']:.3f}\nR2:{df_male_correlations.loc[fold, 'r2_values_true_corrected_delta']:.3f}"
# ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color=color_female)
# ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color=color_male)

ax[0][0].set_box_aspect(1)
ax[0][1].set_box_aspect(1)
ax[1][0].set_box_aspect(1)
ax[1][1].set_box_aspect(1)

xmin00, xmax00 = ax[0][0].get_xlim()
ymin00, ymax00 = ax[0][0].get_ylim()

y_step_value = 20
ax[0][0].set_yticks(np.arange(math.floor(ymin00/10)*10, math.ceil(ymax00/10)*10+10, y_step_value))

ax[0][1].set_yticks(np.arange(math.floor(ymin00/10)*10, math.ceil(ymax00/10)*10+10, y_step_value))

xmin01, xmax01 = ax[0][1].get_xlim()
ymin01, ymax01 = ax[0][1].get_ylim()

xmin10, xmax10 = ax[1][0].get_xlim()
ymin10, ymax10 = ax[1][0].get_ylim()


xmin11, xmax11 = ax[1][1].get_xlim()
ymin11, ymax11 = ax[1][1].get_ylim()


ax[0][0].plot([xmin00, xmax00], [ymin00, ymax00], color='darkgrey', linestyle='--', linewidth=3)
ax[0][1].plot([xmin01, xmax01], [ymin01, ymax01], color='darkgrey', linestyle='--', linewidth=3)
ax[1][0].plot([xmin10, xmax10], [ymin10, ymax10], color='darkgrey', linestyle='--', linewidth=3)
ax[1][1].plot([xmin11, xmax11], [ymin11, ymax11], color='darkgrey', linestyle='--', linewidth=3)

plt.tight_layout()
plt.show()
plt.savefig(plot_file)
plt.close()

print("===== Done! =====")
embed(globals(), locals())