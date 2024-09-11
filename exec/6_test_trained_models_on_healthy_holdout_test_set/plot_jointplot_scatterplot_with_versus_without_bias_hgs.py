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
df_female_correlation_values, df_female_r2_values, df_female_mae_values = load_prediction_correlation_results(
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

df_male_correlation_values, df_male_r2_values, df_male_mae_values = load_prediction_correlation_results(
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

###############################################################################
df = pd.concat([df_female, df_male], axis=0)
#  Replace 0 with "Female" and 1 with "Male" in the gender column
df['gender'] = df['gender'].replace({0: 'Female', 1: 'Male'})

###############################################################################
plot_folder = os.path.join(os.getcwd(), f"plots/with_vs_withou_bias_hgs/jointplot_scatterplot/{target}/{model_name}/{n_repeats}_repeats_{n_folds}_folds/{correlation_type}")
if(not os.path.isdir(plot_folder)):
        os.makedirs(plot_folder)
###############################################################################
# Create a custom color palette dictionary
# Define custom palettes
palette_male = sns.color_palette("Blues")
palette_female = sns.color_palette("Reds")
color_female = palette_female[5]
color_male = palette_male[5]

custom_palette = {"Male": color_male, "Female": color_female}
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Define columns
x_col = f"{target}"
y_cols = [f"{target}_predicted", f"{target}_corrected_predicted", f"{target}_delta(true-predicted)", f"{target}_corrected_delta(true-predicted)"]  # Replace 'another_y_column' with the actual column name
hue_col = "gender"
gender_order = ["Male", "Female"]
# Sort DataFrame by gender column in the desired order
df = df.sort_values(by=hue_col, ascending=False)
###############################################################################
xmin = min(df[x_col])
xmax = max(df[x_col])

ymin_pred = 0
ymax_pred = 0
for y in [f"{target}_predicted", f"{target}_corrected_predicted"]:
    ymin_tmp = min(df[y])
    ymax_tmp = max(df[y])
    if ymin_tmp < ymin_pred:
        ymin_pred = ymin_tmp
    if ymax_tmp > ymax_pred:
        ymax_pred = ymax_tmp
# Set the y-ticks step value
ystep_value_pred = 20
# Calculate the range for y-ticks
yticks_range_pred = np.arange(math.floor(ymin_pred / 10) * 10, math.ceil(ymax_pred / 10) * 10 + 30, ystep_value_pred)
#-----------------------------------------------------------#  
ymin_delta = 0
ymax_delta = 0
for y in [f"{target}_delta(true-predicted)", f"{target}_corrected_delta(true-predicted)"]:
    ymin_tmp = min(df[y])
    ymax_tmp = max(df[y])
    if ymin_tmp < ymin_delta:
        ymin_delta = ymin_tmp
    if ymax_tmp > ymax_delta:
        ymax_delta = ymax_tmp
# Set the y-ticks step value
ystep_value_delta = 20
# Calculate the range for y-ticks
yticks_range_delta = np.arange(math.floor(ymin_delta / 10) * 10, math.ceil(ymax_delta / 10) * 10 + 20, ystep_value_delta)
###############################################################################
for y in y_cols:
    fig = plt.figure()
    sns.set_style("white")
    # Iterate over each subplot to change the font size for tick labels
    plt.tick_params(axis='both', direction='out')
    # Plot female and male data for target vs. predicted in the first subplot
    g = sns.jointplot(data=df, x=x_col, y=y, hue=hue_col, hue_order=gender_order, palette=custom_palette)
    # Remove the legend
    g.ax_joint.legend_.remove()
    # Modify scatter plot markers after creating the jointplot
    for artist in g.ax_joint.collections:
        artist.set_edgecolor('black')
        artist.set_sizes([60])
        artist.set_linewidths([1.2])  # Set the width of the marker borders
        # for _, gr in df.groupby(hue_col):
    for gender in gender_order:
        gr = df[df[hue_col] == gender]        
        line_color = custom_palette[gender]        
        sns.regplot(x=x_col, y=y, data=gr, truncate=False, scatter=False, line_kws={'color': line_color})
    
    # Add a dashed red line for y = x
    if y in [f"{target}_predicted", f"{target}_corrected_predicted"]:
        x_values = np.linspace(yticks_range_pred.min(), 159, 100)
        g.ax_joint.plot(x_values, x_values, color='darkgrey', linestyle='--', linewidth=2)

    # Add a dashed red line at y=0
    if y in [f"{target}_delta(true-predicted)", f"{target}_corrected_delta(true-predicted)"]:
        g.ax_joint.axhline(0, color='darkgrey', linestyle='--', linewidth=2)
    # print("===== Done! =====")
    # embed(globals(), locals())
    #-----------------------------------------------------------#
    if y == f"{target}_predicted":
        plt.ylabel("Predicted HGS", fontsize=14)      
        plt.yticks(yticks_range_pred, fontsize=14)
    if y == f"{target}_corrected_predicted":
        plt.ylabel("Corrected predicted HGS", fontsize=14)      
        plt.yticks(yticks_range_pred, fontsize=14)  
    if y == f"{target}_delta(true-predicted)":
        plt.ylabel("Delta HGS", fontsize=14) 
        plt.yticks(yticks_range_delta, fontsize=14)
        plt.ylim([-70, 70])
    if y == f"{target}_corrected_delta(true-predicted)":
        plt.ylabel("Delta corrected predicted HGS", fontsize=14) 
        plt.yticks(yticks_range_delta, fontsize=14)
        plt.ylim([-70, 70])
    #-----------------------------------------------------------#
    plt.xlabel("True HGS", fontsize=14)
    plt.xticks(fontsize=14)
    #-----------------------------------------------------------#            
    # Annotation for female data in the first subplot
    if y == f"{target}_predicted":
        r_text_male = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_predicted']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_predicted']:.2f}\nMAE(m) = {df_male_mae_values.loc['MAE_values', 'true_vs_predicted']:.2f}"
        r_text_female = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_predicted']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_predicted']:.2f}\nMAE(f) = {df_female_mae_values.loc['MAE_values', 'true_vs_predicted']:.2f}"
        # Annotations for male and female data in the first subplot
        plt.annotate(r_text_male, xy=(0.025, 0.91), xycoords='axes fraction', fontsize=10, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
        plt.annotate(r_text_female, xy=(0.334, 0.91), xycoords='axes fraction', fontsize=10, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
    # Annotation for female data in the second subplot
    if y == f"{target}_corrected_predicted":    
        r_text_male = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_corrected_predicted']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_corrected_predicted']:.2f}\nMAE(m) = {df_male_mae_values.loc['MAE_values', 'true_vs_corrected_predicted']:.2f}"
        r_text_female = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_corrected_predicted']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_corrected_predicted']:.2f}\nMAE(f) = {df_female_mae_values.loc['MAE_values', 'true_vs_corrected_predicted']:.2f}"
        # Annotations for male and female data in the second subplot
        plt.annotate(r_text_male, xy=(0.025, 0.91), xycoords='axes fraction', fontsize=10, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
        plt.annotate(r_text_female, xy=(0.3135, 0.91), xycoords='axes fraction', fontsize=10, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
    # Annotation for female data in the third subplot
    if y == f"{target}_delta(true-predicted)":
        r_text_male = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_delta']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_delta']:.2f}\nMAE(m) = {df_male_mae_values.loc['MAE_values', 'true_vs_delta']:.2f}"
        r_text_female = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_delta']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_delta']:.2f}\nMAE(f) = {df_female_mae_values.loc['MAE_values', 'true_vs_delta']:.2f}"
        # Annotations for male and female data in the third subplot
        plt.annotate(r_text_male, xy=(0.025, 0.91), xycoords='axes fraction', fontsize=10, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
        plt.annotate(r_text_female, xy=(0.335, 0.91), xycoords='axes fraction', fontsize=10, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
    # Annotation for female data in the fourth subplot
    if y == f"{target}_corrected_delta(true-predicted)":
        r_text_male = f"r(m) = {df_male_correlation_values.loc['r_values', 'true_vs_corrected_delta']:.2f}\nR²(m) = {df_male_r2_values.loc['r2_values', 'true_vs_corrected_delta']:.2f}\nMAE(m) = {df_male_mae_values.loc['MAE_values', 'true_vs_corrected_delta']:.2f}"
        r_text_female = f"r(f) = {df_female_correlation_values.loc['r_values', 'true_vs_corrected_delta']:.2f}\nR²(f) = {df_female_r2_values.loc['r2_values', 'true_vs_corrected_delta']:.2f}\nMAE(f) = {df_female_mae_values.loc['MAE_values', 'true_vs_corrected_delta']:.2f}"
        # Annotations for male and female data in the fourth subplot
        plt.annotate(r_text_male, xy=(0.025, 0.91), xycoords='axes fraction', fontsize=10, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
        plt.annotate(r_text_female, xy=(0.335, 0.91), xycoords='axes fraction', fontsize=10, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
    #-----------------------------------------------------------#
    if y in [f"{target}_predicted", f"{target}_delta(true-predicted)"]:               
        plt.suptitle(f"Without bias-adjustment", fontsize=16, fontweight="bold")
        # Get x and y limits for the first subplot first row
        # ymin, ymax = plt.ylim()
    #-----------------------------------------------------------#
        # Plot a dark grey dashed line with width 3 from (xmin, ymin) to (xmax, ymax) on the first subplot
        # plt.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=2)
        
    if y in [f"{target}_corrected_predicted", f"{target}_corrected_delta(true-predicted)"]:
        plt.suptitle(f"With bias-adjustment", fontsize=16, fontweight="bold")
        # # Plot a dark grey dashed line with width 3 from (xmin, ymin) to (xmax, ymax) on the first subplot
        # plt.plot([xmin, xmax], [0, 0], color='darkgrey', linestyle='--', linewidth=2)
    #-----------------------------------------------------------#
    plt.tight_layout()
    #-----------------------------------------------------------#
    plt.show()
    plot_file = os.path.join(plot_folder, f"comparing_with_vs_withou_bias_hgs_{y}_{target}.png")
    plt.savefig(plot_file)
    plt.close()

print("===== Done! =====")
embed(globals(), locals())
