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
df_female_correlation_values, df_female_r2_values = load_prediction_correlation_results(
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

df_male_correlation_values, df_male_r2_values = load_prediction_correlation_results(
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
palette_female = sns.color_palette("Reds")
color_female = palette_female[5]
color_male = palette_male[5]

# palette_female = sns.cubehelix_palette()
# color_female = palette_female[1]
# color_male = palette_male[2]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

sns.set_style("white")

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# Plot female and male data for target vs. predicted in the first subplot
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_predicted", color=color_male, marker="o", scatter_kws={'s': 50, 'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[0][0])
sns.regplot(data=df_female, x=f"{target}", y=f"{target}_predicted", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[0][0])

# Plot female and male data for target vs. corrected predicted in the second subplot
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_corrected_predicted", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[0][1])
sns.regplot(data=df_female, x=f"{target}", y=f"{target}_corrected_predicted", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[0][1])

# Plot female and male data for target vs. delta (true-predicted) in the third subplot
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[1][0])
sns.regplot(data=df_female, x=f"{target}", y=f"{target}_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[1][0])

# Plot female and male data for target vs. corrected delta (true-predicted) in the fourth subplot
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_corrected_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[1][1])
sns.regplot(data=df_female, x=f"{target}", y=f"{target}_corrected_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[1][1])

#-----------------------------------------------------------#
ax[0][0].set_ylabel("Predicted HGS", fontsize=16)
ax[1][0].set_ylabel("Delta HGS", fontsize=16)
ax[0][1].set_ylabel("Adjusted HGS", fontsize=16)
ax[1][1].set_ylabel("Delta adjusted HGS", fontsize=16)

ax[0][0].set_xlabel("") 
ax[0][1].set_xlabel("") 
ax[1][0].set_xlabel("Raw HGS", fontsize=16) 
ax[1][1].set_xlabel("Raw HGS", fontsize=16)     

#-----------------------------------------------------------#                   
ax[0][0].set_title(f"Without bias-adjustment", fontsize=16, fontweight="bold")            
ax[0][1].set_title(f"With bias-adjustment", fontsize=16, fontweight="bold")

#-----------------------------------------------------------#            
# Annotation for female data in the first subplot
r_text_male_00 = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_predicted']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_predicted']:.2f}"
r_text_female_00 = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_predicted']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_predicted']:.2f}"
# Annotations for male and female data in the first subplot
ax[0, 0].annotate(r_text_male_00, xy=(0.03, 0.887), xycoords='axes fraction', fontsize=12, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax[0, 0].annotate(r_text_male_00, xy=(0.335, 0.887), xycoords='axes fraction', fontsize=12, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))

# Annotation for female data in the second subplot
r_text_male_01 = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_corrected_predicted']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_corrected_predicted']:.2f}"
r_text_female_01 = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_corrected_predicted']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_corrected_predicted']:.2f}"
# Annotations for male and female data in the second subplot
ax[0, 1].annotate(r_text_male_01, xy=(0.03, 0.887), xycoords='axes fraction', fontsize=12, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax[0, 1].annotate(r_text_female_01, xy=(0.335, 0.887), xycoords='axes fraction', fontsize=12, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))

# Annotation for female data in the third subplot
r_text_male_10 = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_delta']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_delta']:.2f}"
r_text_female_10 = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_delta']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_delta']:.2f}"
# Annotations for male and female data in the third subplot
ax[1, 0].annotate(r_text_male_10, xy=(0.03, 0.887), xycoords='axes fraction', fontsize=12, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax[1, 0].annotate(r_text_female_10, xy=(0.377, 0.887), xycoords='axes fraction', fontsize=12, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))

# Annotation for female data in the fourth subplot
r_text_male_11 = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_corrected_delta']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_corrected_delta']:.2f}"
r_text_female_11 = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_corrected_delta']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_corrected_delta']:.2f}"
# Annotations for male and female data in the fourth subplot
ax[1, 1].annotate(r_text_male_11, xy=(0.03, 0.887), xycoords='axes fraction', fontsize=12, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax[1, 1].annotate(r_text_female_11, xy=(0.377, 0.887), xycoords='axes fraction', fontsize=12, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
#-----------------------------------------------------------#
ax[0][0].set_box_aspect(1)
ax[0][1].set_box_aspect(1)
ax[1][0].set_box_aspect(1)
ax[1][1].set_box_aspect(1)
#-----------------------------------------------------------#
# Iterate over each subplot to change the font size for tick labels
for axis in ax.flatten():
    axis.tick_params(axis='both', labelsize=12, direction='out', length=5)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin00, xmax00 = ax[0][0].get_xlim()
ymin00, ymax00 = ax[0][0].get_ylim()
# Get x and y limits for the second subplot first row
xmin01, xmax01 = ax[0][1].get_xlim()
ymin01, ymax01 = ax[0][1].get_ylim()
# Find the common y-axis limits
ymin_0 = min(ymin00, ymin01)
ymax_0 = max(ymax00, ymax01)
# Set the y-ticks step value
ystep0_value = 20
# Calculate the range for y-ticks
yticks0_range = np.arange(math.floor(ymin_0 / 10) * 10, math.ceil(ymax_0 / 10) * 10 + 10, ystep0_value)
# Set the y-ticks for both subplots
ax[0][0].set_yticks(yticks0_range)
ax[0][1].set_yticks(yticks0_range)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot second row
xmin10, xmax10 = ax[1][0].get_xlim()
ymin10, ymax10 = ax[1][0].get_ylim()
# Get x and y limits for the second subplot second row
xmin11, xmax11 = ax[1][1].get_xlim()
ymin11, ymax11 = ax[1][1].get_ylim()
# Find the common y-axis limits
ymin_1 = min(ymin10, ymin11)
ymax_1 = max(ymax10, ymax11)
# Set the y-ticks step value
ystep1_value = 20
# Calculate the range for y-ticks
yticks1_range = np.arange(math.floor(ymin_1 / 10) * 10, math.ceil(ymax_1 / 10) * 10 + 10, ystep1_value)
# Set the y-ticks for both subplots
ax[1][0].set_yticks(yticks1_range)
ax[1][1].set_yticks(yticks1_range)

#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin00, xmax00 = ax[0][0].get_xlim()
ymin00, ymax00 = ax[0][0].get_ylim()
# Get x and y limits for the second subplot first row
xmin01, xmax01 = ax[0][1].get_xlim()
ymin01, ymax01 = ax[0][1].get_ylim()
# Get x and y limits for the first subplot second row
xmin10, xmax10 = ax[1][0].get_xlim()
ymin10, ymax10 = ax[1][0].get_ylim()
# Get x and y limits for the second subplot second row
xmin11, xmax11 = ax[1][1].get_xlim()
ymin11, ymax11 = ax[1][1].get_ylim()

#-----------------------------------------------------------#
# Plot a dark grey dashed line with width 3 from (xmin00, ymin00) to (xmax00, ymax00) on the first subplot
ax[0][0].plot([xmin00, xmax00], [ymin00, ymax00], color='darkgrey', linestyle='--', linewidth=3)
# Plot a dark grey dashed line with width 3 from (xmin01, ymin01) to (xmax01, ymax01) on the second subplot
ax[0][1].plot([xmin01, xmax01], [ymin01, ymax01], color='darkgrey', linestyle='--', linewidth=3)
# Plot a dark grey dashed line with width 3 from (xmin10, ymin10) to (xmax10, ymax10) on the third subplot
ax[1][0].plot([xmin10, xmax10], [0, 0], color='darkgrey', linestyle='--', linewidth=3)
# Plot a dark grey dashed line with width 3 from (xmin11, ymin11) to (xmax11, ymax11) on the fourth subplot
ax[1][1].plot([xmin11, xmax11], [0, 0], color='darkgrey', linestyle='--', linewidth=3)
# Ensure y-axis ticks (ymin and ymax) remain unchanged
ax[1][1].set_yticks(yticks1_range)

#-----------------------------------------------------------#
plt.tight_layout()
#-----------------------------------------------------------#
plt.show()
plt.savefig(plot_file)
plt.close()

print("===== Done! =====")
embed(globals(), locals())