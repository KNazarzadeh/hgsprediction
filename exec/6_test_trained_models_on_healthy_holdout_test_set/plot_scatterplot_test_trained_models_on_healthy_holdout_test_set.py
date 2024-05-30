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
plot_folder = os.path.join(os.getcwd(), f"plots/test_trained_models/scatterplot/{target}/{model_name}/{n_repeats}_repeats_{n_folds}_folds/{correlation_type}")
if(not os.path.isdir(plot_folder)):
        os.makedirs(plot_folder)
plot_file = os.path.join(plot_folder, f"test_trained_models_{target}.png")
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

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# Plot female and male data for target vs. predicted in the first subplot
sns.regplot(data=df_male, x=f"{target}", y=f"{target}_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50, 'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax)
sns.regplot(data=df_female, x=f"{target}", y=f"{target}_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax)

#-----------------------------------------------------------#
ax.set_ylabel("Delta HGS", fontsize=16)
ax.set_xlabel("Raw HGS", fontsize=16) 
#-----------------------------------------------------------#                   
ax.set_title(f"Test linear SVM on holdout test set", fontsize=16, fontweight="bold")            
#-----------------------------------------------------------#            
# Annotation for female data in the first subplot
r_text_male_00 = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_delta']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_delta']:.2f}"
r_text_female_00 = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_delta']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_delta']:.2f}"
# Annotations for male and female data in the first subplot
ax.annotate(r_text_male_00, xy=(0.03, 0.887), xycoords='axes fraction', fontsize=12, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax.annotate(r_text_female_00, xy=(0.377, 0.887), xycoords='axes fraction', fontsize=12, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
#-----------------------------------------------------------#
ax.set_box_aspect(1)
#-----------------------------------------------------------#
# Iterate over each subplot to change the font size for tick labels
ax.tick_params(axis='both', labelsize=12, direction='out', length=5)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin00, xmax00 = ax.get_xlim()
ymin00, ymax00 = ax.get_ylim()

ystep0_value = 20
# Calculate the range for y-ticks
yticks0_range = np.arange(math.floor(ymin00 / 10) * 10, math.ceil(ymax00 / 10) * 10 + 10, ystep0_value)
# Set the y-ticks for both subplots
ax.set_yticks(yticks0_range)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin00, xmax00 = ax.get_xlim()
ymin00, ymax00 = ax.get_ylim()
#-----------------------------------------------------------#
# Plot a dark grey dashed line with width 3 from (xmin00, ymin00) to (xmax00, ymax00) on the first subplot
ax.plot([xmin00, xmax00], [ymin00, ymax00], color='darkgrey', linestyle='--', linewidth=3)

#-----------------------------------------------------------#
plt.tight_layout()
#-----------------------------------------------------------#
plt.show()
plt.savefig(plot_file)
plt.close()

print("===== Done! =====")
embed(globals(), locals())