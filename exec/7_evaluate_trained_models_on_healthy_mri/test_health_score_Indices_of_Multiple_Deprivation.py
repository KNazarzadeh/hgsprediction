
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
df_test_file = os.path.join(df_test_folder, "health_score.csv")

df_test = pd.read_csv(df_test_file, sep=',')
df_test = df_test.rename(columns={"eid":"SubjectID"})
df_test = df_test.set_index("SubjectID")
###############################################################################

df_test_female = df_test[df_test['31-0.0']==0]
df_test_male = df_test[df_test['31-0.0']==1]    
###############################################################################
# Select the columns from df_test
columns_to_merge_female = df_test_female["26413-0.0"]

# Merge the selected columns to df based on the indexes
df_merged_female = df_female.merge(columns_to_merge_female, left_index=True, right_index=True)

df_merged_female = df_merged_female.dropna(subset="26413-0.0")


# Select the columns from df_test
columns_to_merge_male = df_test_male["26413-0.0"]

# Merge the selected columns to df based on the indexes
df_merged_male = df_male.merge(columns_to_merge_male, left_index=True, right_index=True)

df_merged_male = df_merged_male.dropna(subset="26413-0.0")

############################################################################### 

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

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot female and male data for target vs. corrected delta (true-predicted) in the fourth subplot
sns.regplot(data=df_merged_male, x=f"26413-0.0", y=f"{target}_corrected_delta(true-predicted)", color=color_male, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_male}, ax=ax)
sns.regplot(data=df_merged_female, x=f"26413-0.0", y=f"{target}_corrected_delta(true-predicted)", color=color_female, marker="o", scatter_kws={'s': 50,'edgecolor': 'black'}, line_kws={"color": color_female}, ax=ax)

#-----------------------------------------------------------#
ax.set_ylabel("Delta corrected HGS", fontsize=16)

ax.set_xlabel("FieldID:26413 (Health score-England)", fontsize=16) 

#-----------------------------------------------------------#                   
ax.set_title(f"With bias-adjustment", fontsize=16, fontweight="bold")

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
x_female=df_merged_female["26413-0.0"]
y_female=df_merged_female[f"{target}_corrected_delta(true-predicted)"]

female_corr_adjusted_delta = pearsonr(y_female, x_female)
#-----------------------------------------------------------#

x_male=df_merged_male["26413-0.0"]
y_male=df_merged_male[f"{target}_corrected_delta(true-predicted)"]

male_corr_adjusted_delta = pearsonr(y_male, x_male)


r_text_male_01 = f"r(m) = {male_corr_adjusted_delta[0]:.2f}\np-value(m) = {male_corr_adjusted_delta[1]:.2f}"
r_text_female_01 = f"r(f) = {female_corr_adjusted_delta[0]:.2f}\np-value(f) = {female_corr_adjusted_delta[1]:.2f}"

ax.annotate(r_text_male_01, xy=(0.025, 0.91), xycoords='axes fraction', fontsize=10, color=color_male, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))
ax.annotate(r_text_female_01, xy=(0.025, 0.96), xycoords='axes fraction', fontsize=10, color=color_female, bbox=dict(boxstyle='square,pad=0.3', edgecolor='lightgrey', facecolor='none'))

plt.tight_layout()
#-----------------------------------------------------------#
plt.show()
plt.savefig("test_health_score_26413.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())

