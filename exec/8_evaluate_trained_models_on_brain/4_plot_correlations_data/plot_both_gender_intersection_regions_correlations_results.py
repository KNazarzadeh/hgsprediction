import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec

from hgsprediction.load_results.brain_correlate.load_brain_correlation_results import load_hgs_correlation_with_brain_regions_results
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
data_set =sys.argv[9]
brain_data_type = sys.argv[10]
tiv_status = sys.argv[11]
schaefer = sys.argv[12]
stats_correlation_type = sys.argv[13]
corr_target = sys.argv[14]
n_top_regions = sys.argv[15]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
session = '2_and_3'
df_female_corr = load_hgs_correlation_with_brain_regions_results(
    brain_data_type,
    schaefer,
    session,
    "female",
    corr_target,
)

df_male_corr = load_hgs_correlation_with_brain_regions_results(
    brain_data_type,
    schaefer,
    session,
    "male",
    corr_target,
)
###############################################################################
df_female_corr.loc[:, "correlations_values"] = df_female_corr.loc[:, "correlations"]
df_male_corr.loc[:, "correlations_values"] = df_male_corr.loc[:, "correlations"]
###############################################################################
# Check if the columns are in the same order
columns_in_same_order = (df_female_corr['regions'].tolist() == df_male_corr['regions'].tolist())
# Print the result
print("Columns are in the same order:", columns_in_same_order)

###############################################################################
# Create a new DataFrame with a column containing values from df_original
df_both_gender_corr = pd.DataFrame({'regions': df_female_corr.loc[:, 'regions']})
# Calculate average values of col1 from df1 and df2
average_values = (df_female_corr.loc[:, 'correlations'] + df_male_corr.loc[:, 'correlations']) / 2
# Create a new DataFrame with regions from df_female_corr and average_values
df_both_gender_corr.loc[:, "average_correlations"] =  average_values
df_both_gender_corr
###############################################################################
df_female_survived = df_female_corr[df_female_corr["significant"]==True]
df_male_survived = df_male_corr[df_male_corr["significant"]==True]
print("len(df_female_survived):", len(df_female_survived))
print("len(df_male_survived):", len(df_male_survived))

###############################################################################
female_corr_threashold = df_female_survived[abs(df_female_survived["correlations"])>.1]
male_corr_threashold = df_male_survived[abs(df_male_survived["correlations"])>.1]
print("len(female_corr_threashold)>.1:",len(female_corr_threashold))
print("len(male_corr_threashold)>.1:",len(male_corr_threashold))
###############################################################################
matching_regions = np.intersect1d(female_corr_threashold['regions'].unique(), male_corr_threashold['regions'].unique())
print("len(matching_regions):", len(matching_regions))
###############################################################################
filtered_rows_both_gender = df_both_gender_corr[df_both_gender_corr["regions"].isin(matching_regions)]
filtered_rows_female = df_female_corr[df_female_corr["regions"].isin(female_corr_threashold["regions"])]
filtered_rows_male = df_male_corr[df_male_corr["regions"].isin(male_corr_threashold["regions"])]

###############################################################################
# Set values in column 'correlations' to 0 for rows where 'regions' is not in matching_regions
df_both_gender_corr.loc[~df_both_gender_corr.index.isin(filtered_rows_both_gender.index), 'average_correlations'] = 0
df_female_corr.loc[~df_female_corr['regions'].isin(filtered_rows_female['regions']), 'correlations_values'] = 0
df_female = df_female_corr[["regions", "correlations_values"]]
df_male_corr.loc[~df_male_corr['regions'].isin(filtered_rows_male['regions']), 'correlations_values'] = 0
df_male = df_male_corr[["regions", "correlations_values"]]
###############################################################################
print("===== Done! =====")
embed(globals(), locals())
df_both_gender_corr_intersected = df_both_gender_corr[df_both_gender_corr['average_correlations']!=0]
df_female_corr_not_intersected = df_female[df_female['correlations_values']!=0]
df_male_corr_not_intersected = df_male[df_male['correlations_values']!=0]

print("len(df_both_gender_corr_intersected):", len(df_both_gender_corr_intersected))
print(df_both_gender_corr_intersected)
print("len(df_female_corr_not_intersected):", len(df_female_corr_not_intersected))
print(df_female_corr_not_intersected)
print("len(df_male_corr_not_intersected):", len(df_male_corr_not_intersected))
print(df_male_corr_not_intersected)
###############################################################################
# Sorting the 'average_correlations' column from maximum to minimum
# based on Absolute value of correlations
df_sorted_both_gender = df_both_gender_corr_intersected.sort_values(by='average_correlations', key=abs, ascending=False)
df_sorted_female = df_female_corr_not_intersected.sort_values(by='correlations_values', key=abs, ascending=False)
df_sorted_male = df_male_corr_not_intersected.sort_values(by='correlations_values', key=abs, ascending=False)

###############################################################################
# Select the top 30 regions
top_corr_both_gender = df_sorted_both_gender.head(int(n_top_regions))
top_corr_female = df_sorted_female.head(int(n_top_regions))
top_corr_male = df_sorted_male.head(int(n_top_regions))

###############################################################################
# Apply the custom colormap
cmap_custom = plt.cm.RdBu_r
# Normalize the 'correlation' column to the range [-0.32, 0.32]
norm = Normalize(vmin=-0.32, vmax=0.32)
###############################################################################
# Normalize the 'Correlation' values to the range [-0.32, 0.32] for coloring
# Ensure that all values are within the range [-0.32, 0.32] before applying normalization
top_corr_both_gender['Normalized_Correlation'] = top_corr_both_gender['average_correlations'].clip(lower=-0.32, upper=0.32)
normalized_values_both_gender = norm(top_corr_both_gender['Normalized_Correlation'])

top_corr_female['Normalized_Correlation'] = top_corr_female['correlations_values'].clip(lower=-0.32, upper=0.32)
normalized_values_female = norm(top_corr_female['Normalized_Correlation'])

top_corr_male['Normalized_Correlation'] = top_corr_male['correlations_values'].clip(lower=-0.32, upper=0.32)
normalized_values_male = norm(top_corr_male['Normalized_Correlation'])

# Plot the barplot
# Create a color map for the bar colors based on normalized values
colors_both_gender = cmap_custom(normalized_values_both_gender)
colors_female = cmap_custom(normalized_values_female)
colors_male = cmap_custom(normalized_values_male)

###############################################################################
# Plot the barplot
fig, ax = plt.subplots(3,1, figsize=(50, 18))
sns.set_style("white")
sns.barplot(x='regions', y='average_correlations', data=top_corr_both_gender, palette=colors_both_gender, ax=ax[0])
sns.barplot(x='regions', y='correlations_values', data=top_corr_male, palette=colors_male, ax=ax[1])
sns.barplot(x='regions', y='correlations_values', data=top_corr_female, palette=colors_female, ax=ax[2])

# Adjust the bar width manually
bar_width = 0.3  # Decrease bar width
for bar in ax[0].patches:
    bar.set_width(bar_width)
for bar in ax[1].patches:
    bar.set_width(bar_width)
for bar in ax[1].patches:
    bar.set_width(bar_width)

# Adjust x-tick positions and labels
x_positions_0 = np.arange(len(top_corr_both_gender))
ax[0].set_xticks(x_positions_0)
ax[0].set_xticklabels(top_corr_both_gender['regions'], rotation=45, ha='right', fontsize=18)

# Adjust x-tick positions and labels
x_positions_1 = np.arange(len(top_corr_male))
ax[1].set_xticks(x_positions_1)
ax[1].set_xticklabels(top_corr_male['regions'], rotation=45, ha='right', fontsize=18)

# Adjust x-tick positions and labels
x_positions_2 = np.arange(len(top_corr_female))
ax[2].set_xticks(x_positions_2)
ax[2].set_xticklabels(top_corr_female['regions'], rotation=45, ha='right', fontsize=18)

# Set y-axis limits (you can adjust these values based on your data)
ymin = -0.32  # Example minimum value
ymax = 0  # Example maximum value
ystep = 0.08
ax[0].set_ylim(ymin, ymax)
ax[1].set_ylim(ymin, ymax)
ax[2].set_ylim(ymin, ymax)

# Set specific y-axis ticks
ax[0].set_yticks(np.arange(ymin, ymax+0.08, ystep))
ax[1].set_yticks(np.arange(ymin, ymax+0.08, ystep))
ax[2].set_yticks(np.arange(ymin, ymax+0.08, ystep))

# Set y-tick labels font size
ax[0].yaxis.set_tick_params(labelsize=18)
ax[1].yaxis.set_tick_params(labelsize=18)
ax[2].yaxis.set_tick_params(labelsize=18)

# Add labels and title
ax[0].set_xlabel('')
ax[1].set_xlabel('')
ax[2].set_xlabel('')

ax[0].set_ylabel('correlations', fontsize=20)
ax[1].set_ylabel('correlations', fontsize=20)
ax[2].set_ylabel('correlations', fontsize=20)

plt.title(f'Top {n_top_regions} Regions with Highest Absolute Correlations')

plt.tight_layout()
# Show the plot
plt.show()
plt.savefig(f"all_genders_data_regions_{corr_target}_{n_top_regions}_top_regions.png")

print("===== Done! =====")
embed(globals(), locals())
