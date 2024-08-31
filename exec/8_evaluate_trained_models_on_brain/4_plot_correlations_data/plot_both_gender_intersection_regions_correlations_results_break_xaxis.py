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
# Subset the data for top 30 and bottom 30 regions
n_top_regions = int(n_top_regions)
top_both = df_sorted_both_gender.head(n_top_regions)
bottom_both = df_sorted_both_gender.tail(n_top_regions)

top_male = df_sorted_male.head(n_top_regions)
bottom_male = df_sorted_male.tail(n_top_regions)

top_female = df_sorted_female.head(n_top_regions)
bottom_female = df_sorted_female.tail(n_top_regions)
###############################################################################
# Apply the custom colormap
cmap_custom = plt.cm.RdBu_r
# Normalize the 'correlation' column to the range [-0.32, 0.32]
norm = Normalize(vmin=-0.32, vmax=0.32)
###############################################################################
# Normalize the 'Correlation' values to the range [-0.32, 0.32] for coloring
# Ensure that all values are within the range [-0.32, 0.32] before applying normalization
# Create color maps for each subset
colors_top_30_both = cmap_custom(norm(top_both['average_correlations'].clip(lower=-0.32, upper=0.32)))
colors_bottom_30_both = cmap_custom(norm(bottom_both['average_correlations'].clip(lower=-0.32, upper=0.32)))

colors_top_30_male = cmap_custom(norm(top_male['correlations_values'].clip(lower=-0.32, upper=0.32)))
colors_bottom_30_male = cmap_custom(norm(bottom_male['correlations_values'].clip(lower=-0.32, upper=0.32)))

colors_top_30_female = cmap_custom(norm(top_female['correlations_values'].clip(lower=-0.32, upper=0.32)))
colors_bottom_30_female = cmap_custom(norm(bottom_female['correlations_values'].clip(lower=-0.32, upper=0.32)))

print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# Plot the barplot
fig, ax = plt.subplots(3,1, figsize=(60, 30))
sns.set_style("white")
# Plotting the top 30 and bottom 30 for both genders
sns.barplot(x='regions', y='average_correlations', data=top_both, palette=colors_top_30_both, ax=ax[0])
sns.barplot(x='regions', y='average_correlations', data=bottom_both, palette=colors_bottom_30_both, ax=ax[0])

# Plotting the top 30 and bottom 30 for male
sns.barplot(x='regions', y='correlations_values', data=top_male, palette=colors_top_30_male, ax=ax[1])
sns.barplot(x='regions', y='correlations_values', data=bottom_male, palette=colors_bottom_30_male, ax=ax[1])

# Plotting the top 30 and bottom 30 for female
sns.barplot(x='regions', y='correlations_values', data=top_female, palette=colors_top_30_female, ax=ax[2])
sns.barplot(x='regions', y='correlations_values', data=bottom_female, palette=colors_bottom_30_female, ax=ax[2])

# Adjust bar width and recenter for all plots
bar_width = 0.3
for i in range(3):
    for j in range(2):
        for bar in ax[i, j].patches:
            bar.set_width(bar_width)
            bar.set_x(bar.get_x() + (1 - bar_width) / 2)

# Adjust x-tick positions and labels for each subplot
for i in range(3):
    ax[i, 0].set_xticks(np.arange(len(top_both)) + bar_width / 2)
    ax[i, 0].set_xticklabels(top_both['regions'], rotation=45, ha='right', fontsize=24)
    ax[i, 1].set_xticks(np.arange(len(bottom_both)) + bar_width / 2)
    ax[i, 1].set_xticklabels(bottom_both['regions'], rotation=45, ha='right', fontsize=24)

# Set y-axis limits to be from -0.32 to 0.32
ymin, ymax, ystep = -0.32, 0.32, 0.08

for i in range(3):
    ax[i, 0].set_ylim(ymin, ymax)
    ax[i, 1].set_ylim(ymin, ymax)
    ax[i, 0].set_yticks(np.arange(ymin, ymax + ystep, ystep))
    ax[i, 1].set_yticks(np.arange(ymin, ymax + ystep, ystep))
    ax[i, 0].yaxis.set_tick_params(labelsize=24)
    ax[i, 1].yaxis.set_tick_params(labelsize=24)

# Set y-axis labels for all rows
for i in range(3):
    ax[i, 0].set_ylabel('correlations', fontsize=28)
    ax[i, 1].set_ylabel('correlations', fontsize=28)

# Remove x-axis labels
for i in range(3):
    ax[i, 0].set_xlabel('')
    ax[i, 1].set_xlabel('')

# Add title
plt.suptitle(f'Top 30 and Bottom 30 Regions with Highest and Lowest Absolute Correlations', fontsize=32)

plt.tight_layout()
# Show the plot
plt.show()
plt.savefig(f"xxxxx.png")

print("===== Done! =====")
embed(globals(), locals())
