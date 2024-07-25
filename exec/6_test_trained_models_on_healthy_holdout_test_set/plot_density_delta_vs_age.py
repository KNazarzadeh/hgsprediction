import os
import sys
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from scipy.stats import pearsonr

from hgsprediction.load_results.healthy.load_corrected_prediction_results import load_corrected_prediction_results

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

###############################################################################
plot_folder = os.path.join(os.getcwd(), f"plots/with_vs_withou_bias_hgs/density_plots/scatterplot/{target}/{model_name}/{n_repeats}_repeats_{n_folds}_folds/{correlation_type}")
if(not os.path.isdir(plot_folder)):
        os.makedirs(plot_folder)
plot_file = os.path.join(plot_folder, f"comparing_delta_vs_age_two_sessions_with_vs_withou_bias_hgs_{target}.png")
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

fig, ax = plt.subplots(1, 2, figsize=(10, 10))

# Plot female and male data for target vs. delta (true-predicted) in the third subplot
sns.regplot(data=df_male, x=f"age", y=f"{target}_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[0])
sns.regplot(data=df_female, x=f"age", y=f"{target}_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[0])

# Plot female and male data for target vs. corrected delta (true-predicted) in the fourth subplot
sns.regplot(data=df_male, x=f"age", y=f"{target}_corrected_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[1])
sns.regplot(data=df_female, x=f"age", y=f"{target}_corrected_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[1])

#-----------------------------------------------------------#
ax[0].set_ylabel("Delta HGS", fontsize=16)
ax[1].set_ylabel("Delta adjusted HGS", fontsize=16)

ax[0].set_xlabel("Age", fontsize=16) 
ax[1].set_xlabel("Age", fontsize=16) 

#-----------------------------------------------------------#                   
ax[0].set_title(f"Without bias-adjustment", fontsize=16, fontweight="bold")            
ax[1].set_title(f"With bias-adjustment", fontsize=16, fontweight="bold")

#-----------------------------------------------------------#
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)
#-----------------------------------------------------------#
# Iterate over each subplot to change the font size for tick labels
for axis in ax.flatten():
    axis.tick_params(axis='both', labelsize=12, direction='out', length=5)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin00, xmax00 = ax[0].get_xlim()
ymin00, ymax00 = ax[0].get_ylim()
# Get x and y limits for the second subplot first row
xmin01, xmax01 = ax[1].get_xlim()
ymin01, ymax01 = ax[1].get_ylim()
# Find the common y-axis limits
ymin_0 = min(ymin00, ymin01)
ymax_0 = max(ymax00, ymax01)
# Set the y-ticks step value
ystep0_value = 20
# Calculate the range for y-ticks
yticks0_range = np.arange(math.floor(ymin_0 / 10) * 10, math.ceil(ymax_0 / 10) * 10 + 10, ystep0_value)
# Set the y-ticks for both subplots
ax[0].set_yticks(yticks0_range)
ax[1].set_yticks(yticks0_range)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin00, xmax00 = ax[0].get_xlim()
ymin00, ymax00 = ax[0].get_ylim()
# Get x and y limits for the second subplot first row
xmin01, xmax01 = ax[1].get_xlim()
ymin01, ymax01 = ax[1].get_ylim()
#-----------------------------------------------------------#
# Plot a dark grey dashed line with width 3 from (xmin00, ymin00) to (xmax00, ymax00) on the first subplot
ax[0].plot([xmin00, xmax00], [ymin00, ymax00], color='darkgrey', linestyle='--', linewidth=3)
# Plot a dark grey dashed line with width 3 from (xmin01, ymin01) to (xmax01, ymax01) on the second subplot
ax[1].plot([xmin01, xmax01], [ymin01, ymax01], color='darkgrey', linestyle='--', linewidth=3)
#-----------------------------------------------------------#
plt.tight_layout()
#-----------------------------------------------------------#
plt.show()
plt.savefig(plot_file)
plt.close()

print("===== Done! =====")
embed(globals(), locals())

x_female=df_female["age"]
y_female=df_female[f"{target}_delta(true-predicted)"]

female_corr_delta = pearsonr(y_female, x_female)
#-----------------------------------------------------------#
x_female=df_female["age"]
y_female=df_female[f"{target}_corrected_delta(true-predicted)"]

female_corr_adjusted_delta = pearsonr(y_female, x_female)
#-----------------------------------------------------------#
x_male=df_male["age"]
y_male=df_male[f"{target}_delta(true-predicted)"]

male_corr_delta = pearsonr(y_male, x_male)

x_male=df_male["age"]
y_male=df_male[f"{target}_corrected_delta(true-predicted)"]

male_corr_adjusted_delta = pearsonr(y_male, x_male)