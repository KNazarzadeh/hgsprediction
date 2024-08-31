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
filtered_rows = df_both_gender_corr[df_both_gender_corr["regions"].isin(matching_regions)]
###############################################################################
# Set values in column 'correlations' to 0 for rows where 'regions' is not in matching_regions
df_both_gender_corr.loc[~df_both_gender_corr.index.isin(filtered_rows.index), 'average_correlations'] = 0
###############################################################################
df_both_gender_corr_intersected = df_both_gender_corr[df_both_gender_corr['average_correlations']!=0]

print("len(df_both_gender_corr_intersected):", len(df_both_gender_corr_intersected))
print(df_both_gender_corr_intersected)
###############################################################################
# Sorting the 'average_correlations' column from maximum to minimum
# based on Absolute value of correlations
df_sorted = df_both_gender_corr_intersected.sort_values(by='average_correlations', key=abs, ascending=False)
###############################################################################
n_top_regions = int(n_top_regions)
# Select the top 30 regions
top_corr = df_sorted.head(n_top_regions)
# top_corr = top_corr.reindex()
###############################################################################
# Select the bottom 30 regions
bottom_corr = df_sorted.tail(n_top_regions)
###############################################################################
# Combine top 30 and bottom 30 regions
combined_corr = pd.concat([top_corr, bottom_corr])
###############################################################################
# Apply the custom colormap
cmap_custom = plt.cm.RdBu_r

# Normalize the 'Age' column to the range [-0.32, 0.32]
norm = Normalize(vmin=-0.32, vmax=0.32)

# Normalize the 'Correlation' values to the range [-0.32, 0.32] for coloring
# Ensure that all values are within the range [-0.32, 0.32] before applying normalization
combined_corr['Normalized_Correlation'] = combined_corr['average_correlations'].clip(lower=-0.32, upper=0.32)
normalized_values = norm(combined_corr['Normalized_Correlation'])

# Plot the barplot
# Create a color map for the bar colors based on normalized values
colors = cmap_custom(normalized_values)

###############################################################################
print("===== Done! =====")
embed(globals(), locals())
# Define x-limits for breaks
xlim1 = [0, n_top_regions]
xlim2 = [n_top_regions + 5, n_top_regions * 2 + 5]  # Adding some space between breaks


# Plot the barplot
fig = plt.figure(figsize=(120, 40))
bax = brokenaxes(xlims=(xlim1, xlim2), hspace=.05)

# Ensure colors are correctly sliced
top_colors = colors[:n_top_regions]
bottom_colors = colors[n_top_regions:]

# Plot top 30 and bottom 30 regions
bax.bar(top_corr.index, top_corr['average_correlations'], color=colors[:n_top_regions], label='Top 30')
bax.bar(bottom_corr.index + n_top_regions + 5, bottom_corr['average_correlations'], color=colors[n_top_regions:], label='Bottom 30')

# Add broken axis markers and labels
bax.axvline(x=n_top_regions, color='k', linestyle='--', alpha=0.7)
bax.axvline(x=n_top_regions + 5, color='k', linestyle='--', alpha=0.7)

# Adjust x-tick positions and labels
combined_x_positions = np.concatenate([top_corr.index, bottom_corr.index + n_top_regions + 5])
bax.set_xticks(combined_x_positions)
bax.set_xticklabels(combined_corr['regions'], rotation=45, ha='right', fontsize=18)

# Determine y limits from the two segments
ylim_min = bax.get_ylim()[0][0]  # Limits for the first segment
ylim_max = bax.get_ylim()[1][1]  # Limits for the second segment

# Choose a y position that's within both segments
y_text_position = ylim_min + (ylim_max - ylim_min) * 0.5  # Midpoint of the visible y-range

bax.text(x=n_top_regions / 2, y=y_text_position, s='Break', horizontalalignment='center', verticalalignment='center', fontsize=14, color='k')


# Set y-axis limits and ticks
bax.set_ylim(ylim_min, ylim_max)
bax.set_yticks(np.arange(ylim_min, ylim_max + 0.08, 0.08))
# bax.yaxis.set_tick_params(labelsize=18)

# Add labels and title
bax.set_xlabel('')
bax.set_ylabel('correlations', fontsize=20)
bax.set_title(f'Top {n_top_regions} and Bottom {n_top_regions} Regions with Highest and Lowest Absolute Correlations')

plt.tight_layout()
plt.show()
plt.savefig(f"xxxxx.png")


###############################################################################
# sns.set_style("white")
# ax = sns.barplot(x='regions', y='average_correlations', data=top_corr, palette=colors)

# # Adjust the bar width manually
# bar_width = 0.3  # Decrease bar width
# for bar in ax.patches:
#     bar.set_width(bar_width)

# # Adjust x-tick positions and labels
# x_positions = np.arange(len(top_corr))
# ax.set_xticks(x_positions)
# ax.set_xticklabels(top_corr['regions'], rotation=45, ha='right', fontsize=18)

# # Set y-axis limits (you can adjust these values based on your data)
# ymin = -0.32  # Example minimum value
# ymax = 0  # Example maximum value
# ystep = 0.08
# ax.set_ylim(ymin, ymax)

# # Set specific y-axis ticks
# ax.set_yticks(np.arange(ymin, ymax+0.08, ystep))
# # Set y-tick labels font size
# ax.yaxis.set_tick_params(labelsize=18)

# # Add labels and title
# plt.xlabel('')
# plt.ylabel('correlations', fontsize=20)
# plt.title(f'Top {n_top_regions} Regions with Highest Absolute Correlations')

# plt.tight_layout()
# # Show the plot
# plt.show()
# plt.savefig(f"both_gender_intersection_regions_{corr_target}_{n_top_regions}_top_regions.png")

print("===== Done! =====")
embed(globals(), locals())
