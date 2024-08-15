
import sys
import os
import pandas as pd
import numpy as np
from hgsprediction.load_results.healthy.load_zscore_results import load_zscore_results
from hgsprediction.save_results.healthy.save_prediction_correlation_results import save_prediction_correlation_results
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import math
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
confound_status = sys.argv[6]
n_repeats = sys.argv[7]
n_folds = sys.argv[8]
data_set = sys.argv[9]
correlation_type = sys.argv[10]
###############################################################################
if correlation_type == "pearson":
    correlation_func = pearsonr
elif correlation_type == "spearman":
    correlation_func = spearmanr
###############################################################################
df_female = load_zscore_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    "2",
    confound_status,
    n_repeats,
    n_folds,
)
print(df_female)

df_male = load_zscore_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    "2",
    confound_status,
    n_repeats,
    n_folds,
)
print(df_male)
###############################################################################        
df_test_folder = "/data/project/stroke_ukb/knazarzadeh/ukb_original_data/new_version_ukb_bids_2024/ukb_bids"
df_test_file = os.path.join(df_test_folder, "acceleration_average.csv")

df_test = pd.read_csv(df_test_file, sep=',')
df_test = df_test.rename(columns={"eid":"SubjectID"})
df_test = df_test.set_index("SubjectID")

###############################################################################

df_test_female = df_test[df_test['31-0.0']==0]
df_test_male = df_test[df_test['31-0.0']==1]    
###############################################################################
# Select the columns from df_test
columns_to_merge_female = df_test_female["90012-0.0"]

# Merge the selected columns to df based on the indexes
df_merged_female = df_female.merge(columns_to_merge_female, left_index=True, right_index=True)

df_merged_female = df_merged_female.dropna(subset="90012-0.0")

###############################################################################
# Select the columns from df_test
columns_to_merge_male = df_test_male["90012-0.0"]

# Merge the selected columns to df based on the indexes
df_merged_male = df_male.merge(columns_to_merge_male, left_index=True, right_index=True)

df_merged_male = df_merged_male.dropna(subset="90012-0.0") 

###############################################################################
# Create a custom color palette dictionary
# Define custom palettes
palette_male = sns.color_palette("Blues")
palette_female = sns.color_palette("Reds")
color_female = palette_female[5]
color_male = palette_male[5]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
sns.set_style("white")

fig, ax = plt.subplots(1, 3, figsize=(22, 10))

# Plot female and male data for target vs. corrected delta (true-predicted) in the fourth subplot
sns.regplot(data=df_merged_male, x=f"90012-0.0", y=f"{target}", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[0])
sns.regplot(data=df_merged_female, x=f"90012-0.0", y=f"{target}", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[0])

# Plot female and male data for target vs. corrected delta (true-predicted) in the fourth subplot
sns.regplot(data=df_merged_male, x=f"90012-0.0", y=f"{target}_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[1])
sns.regplot(data=df_merged_female, x=f"90012-0.0", y=f"{target}_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[1])

# Plot female and male data for target vs. corrected delta (true-predicted) in the fourth subplot
sns.regplot(data=df_merged_male, x=f"90012-0.0", y=f"{target}_corrected_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax[2])
sns.regplot(data=df_merged_female, x=f"90012-0.0", y=f"{target}_corrected_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax[2])

#-----------------------------------------------------------#
ax[0].set_ylabel("True HGS", fontsize=16)
ax[0].set_xlabel("FieldID:90012 (Overall acceleration average)", fontsize=16) 
ax[1].set_ylabel("Delta HGS", fontsize=16)
ax[1].set_xlabel("FieldID:90012 (Overall acceleration average)", fontsize=16) 
ax[2].set_ylabel("Delta corrected HGS", fontsize=16)
ax[2].set_xlabel("FieldID:90012 (Overall acceleration average)", fontsize=16) 

#-----------------------------------------------------------#                   
ax[0].set_title(f"True HGS", fontsize=16, fontweight="bold")
ax[1].set_title(f"Without bias-adjustment", fontsize=16, fontweight="bold")
ax[2].set_title(f"With bias-adjustment", fontsize=16, fontweight="bold")

#-----------------------------------------------------------#
ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)
ax[2].set_box_aspect(1)
#-----------------------------------------------------------#
# Iterate over each subplot to change the font size for tick labels
for axis in ax.flatten():
    axis.tick_params(axis='both', labelsize=12, direction='out', length=5)
#-----------------------------------------------------------#
# Get x and y limits for the first subplot first row
xmin0, xmax0 = ax[0].get_xlim()
ymin0, ymax0 = ax[0].get_ylim()
# Get x and y limits for the second subplot first row
xmin1, xmax1 = ax[1].get_xlim()
ymin1, ymax1 = ax[1].get_ylim()
# Get x and y limits for the first subplot second row
xmin2, xmax2 = ax[2].get_xlim()
ymin2, ymax2 = ax[2].get_ylim()
#-----------------------------------------------------------#
# Plot a dark grey dashed line with width 3 from (xmin00, ymin00) to (xmax00, ymax00) on the first subplot
ax[0].plot([xmin0, xmax0], [ymin0, ymax0], color='darkgrey', linestyle='--', linewidth=3)
# Plot a dark grey dashed line with width 3 from (xmin01, ymin01) to (xmax01, ymax01) on the second subplot
ax[1].plot([xmin1, xmax1], [ymin1, ymax1], color='darkgrey', linestyle='--', linewidth=3)
# Plot a dark grey dashed line with width 3 from (xmin10, ymin10) to (xmax10, ymax10) on the third subplot
ax[2].plot([xmin2, xmax2], [ymin2, ymax2], color='darkgrey', linestyle='--', linewidth=3)
# Plot a dark grey dashed line with width 3 from (xmin11, ymin11) to (xmax11, ymax11) on the fourth subplot
#-----------------------------------------------------------#
x_female=df_merged_female["90012-0.0"]
y_female=df_merged_female[f"{target}_corrected_delta(true-predicted)"]

female_corr_adjusted_delta = pearsonr(y_female, x_female)

x_male=df_merged_male["90012-0.0"]
y_male=df_merged_male[f"{target}_corrected_delta(true-predicted)"]

male_corr_adjusted_delta = pearsonr(y_male, x_male)
#-----------------------------------------------------------#
x_female=df_merged_female["90012-0.0"]
y_female=df_merged_female[f"{target}"]

female_corr_true = pearsonr(y_female, x_female)

x_male=df_merged_male["90012-0.0"]
y_male=df_merged_male[f"{target}"]

male_corr_true = pearsonr(y_male, x_male)

#-----------------------------------------------------------#
x_female=df_merged_female["90012-0.0"]
y_female=df_merged_female[f"{target}_delta(true-predicted)"]

female_corr_delta = pearsonr(y_female, x_female)

x_male=df_merged_male["90012-0.0"]
y_male=df_merged_male[f"{target}_delta(true-predicted)"]

male_corr_delta = pearsonr(y_male, x_male)

#-----------------------------------------------------------#
r_text_male_0 = f"r(m) = {male_corr_true[0]:.2f}\np-value(m) = {male_corr_true[1]:.2f}"
r_text_female_0 = f"r(f) = {female_corr_true[0]:.2f}\np-value(f) = {female_corr_true[1]:.2f}"

ax[0].annotate(r_text_male_0, xy=(0.025, 0.88), xycoords='axes fraction', fontsize=10, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax[0].annotate(r_text_female_0, xy=(0.025, 0.94), xycoords='axes fraction', fontsize=10, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
#-----------------------------------------------------------#
r_text_male_1 = f"r(m) = {male_corr_delta[0]:.2f}\np-value(m) = {male_corr_delta[1]:.2f}"
r_text_female_1 = f"r(f) = {female_corr_delta[0]:.2f}\np-value(f) = {female_corr_delta[1]:.2f}"

ax[1].annotate(r_text_male_1, xy=(0.025, 0.88), xycoords='axes fraction', fontsize=10, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax[1].annotate(r_text_female_1, xy=(0.025, 0.94), xycoords='axes fraction', fontsize=10, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
#-----------------------------------------------------------#
r_text_male_2 = f"r(m) = {male_corr_adjusted_delta[0]:.2f}\np-value(m) = {male_corr_adjusted_delta[1]:.2f}"
r_text_female_2 = f"r(f) = {female_corr_adjusted_delta[0]:.2f}\np-value(f) = {female_corr_adjusted_delta[1]:.2f}"

ax[2].annotate(r_text_male_2, xy=(0.025, 0.88), xycoords='axes fraction', fontsize=10, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax[2].annotate(r_text_female_2, xy=(0.025, 0.94), xycoords='axes fraction', fontsize=10, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))

plt.tight_layout()
#-----------------------------------------------------------#
plt.show()
plt.savefig("test_health_score_90012.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())

