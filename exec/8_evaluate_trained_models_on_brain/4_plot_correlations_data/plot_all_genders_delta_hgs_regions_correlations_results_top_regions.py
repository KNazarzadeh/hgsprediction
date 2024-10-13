import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from matplotlib.gridspec import GridSpec
from matplotlib import colors
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
# print("===== Done! =====")
# embed(globals(), locals())
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
# Select the top 30 regions
top_corr_both_gender = df_sorted_both_gender.head(int(n_top_regions))
top_corr_female = df_sorted_female.head(int(n_top_regions))
top_corr_male = df_sorted_male.head(int(n_top_regions))

###############################################################################
if corr_target == "hgs_corrected_delta":
    vmin = -0.32
    vmax = 0
    threshold = vmax
    # Apply the custom colormap
    # cmap_custom = plt.cm.YlGnBu_r
    cmap_custom = plt.cm.Spectral_r
    ##########################################################
    # Define the levels (bins) you want within the colormap
    levels = MaxNLocator(nbins=10).tick_values(vmin, vmax)  # Specify your bin edges
    # Create a BoundaryNorm object
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=cmap_custom.N)
    ##########################################################
    vmin_female = -0.32
    vmax_female = 0.16
    threshold_female = 0
    # Apply the custom colormap
    cmap_custom_female = plt.cm.Spectral_r
###############################################################################
# Normalize the 'Correlation' values to the range [-0.32, 0.32] for coloring
# Ensure that all values are within the range [-0.32, 0.32] before applying normalization
top_corr_both_gender['Normalized_Correlation'] = top_corr_both_gender['average_correlations'].clip(lower=vmin, upper=vmax)
normalized_values_both_gender = norm(top_corr_both_gender['Normalized_Correlation'])

top_corr_male['Normalized_Correlation'] = top_corr_male['correlations_values'].clip(lower=vmin, upper=vmax)
normalized_values_male = norm(top_corr_male['Normalized_Correlation'])

top_corr_female['Normalized_Correlation'] = top_corr_female['correlations_values'].clip(lower=vmin, upper=vmax)
normalized_values_female = norm(top_corr_female['Normalized_Correlation'])

# Create a color map for the bar colors based on normalized values
colors_both_gender = cmap_custom(normalized_values_both_gender)
colors_female = cmap_custom(normalized_values_female)
colors_male = cmap_custom(normalized_values_male)

###############################################################################
# Plot the barplot
fig, ax = plt.subplots(3,1, figsize=(40, 32))
sns.set_style("white")
###############################################################################
# custom_ticks = levels
# Normalize the color values based on vmin and vmax
# norm = colors.BoundaryNorm(boundaries=custom_ticks, ncolors=cmap_custom.N)
ax[0].scatter(
    x=top_corr_both_gender['regions'], 
    y=top_corr_both_gender['average_correlations'], 
    c=top_corr_both_gender['average_correlations'],  # Use correlation values for coloring
    cmap=cmap_custom,                        # Apply the custom colormap
    norm=norm,                               # Normalize between vmin and vmax
    zorder=5, 
    s=300
)
# Draw vertical lines from each scatter point down to the x-axis
ax[0].vlines(x=top_corr_both_gender['regions'], 
          ymin=0,                          # From the x-axis (y=0)
          ymax=top_corr_both_gender['average_correlations'], 
          color='darkgrey', 
          linewidth=4)
###############################################################################
# custom_ticks = [-0.3, -0.25, -0.2, -0.15, -0.1]
# # Normalize the color values based on vmin and vmax
# norm = colors.BoundaryNorm(boundaries=custom_ticks, ncolors=cmap_custom.N)
# Scatter plot with colors based on the colormap and normalized values
ax[1].scatter(
    x=top_corr_male['regions'], 
    y=top_corr_male['correlations_values'], 
    c=top_corr_male['correlations_values'],  # Use correlation values for coloring
    cmap=cmap_custom,                        # Apply the custom colormap
    norm=norm,                               # Normalize between vmin and vmax
    zorder=5, 
    s=300
)
# Draw vertical lines from each scatter point down to the x-axis
ax[1].vlines(x=top_corr_male['regions'], 
          ymin=0,                          # From the x-axis (y=0)
          ymax=top_corr_male['correlations_values'], 
          color='darkgrey', 
          linewidth=4)
###############################################################################

# custom_ticks = [-0.3, -0.2, -0.08, 0.04, 0.16]
# Normalize the color values based on vmin and vmax
# norm = colors.BoundaryNorm(boundaries=custom_ticks, ncolors=cmap_custom.N)
ax[2].scatter(
    x=top_corr_female['regions'], 
    y=top_corr_female['correlations_values'], 
    c=top_corr_female['correlations_values'],  # Use correlation values for coloring
    cmap=cmap_custom,                        # Apply the custom colormap
    norm=norm,                               # Normalize between vmin and vmax
    zorder=5, 
    s=300
)
# Draw vertical lines from each scatter point down to the x-axis
ax[2].vlines(x=top_corr_female['regions'], 
          ymin=0,                          # From the x-axis (y=0)
          ymax=top_corr_female['correlations_values'], 
          color='darkgrey', 
          linewidth=4)

###############################################################################
# Adjust the bar width manually
bar_width = 0.4  # Decrease bar width
for bar in ax[0].patches:
    bar.set_width(bar_width)
    bar.set_x(bar.get_x() + (1 - bar_width) / 2)  # Centering the bar
for bar in ax[1].patches:
    bar.set_width(bar_width)
    bar.set_x(bar.get_x() + (1 - bar_width) / 2)  # Centering the bar    
for bar in ax[2].patches:
    bar.set_width(bar_width)
    bar.set_x(bar.get_x() + (1 - bar_width) / 2)  # Centering the bar    
###############################################################################
# Adjust x-tick positions and labels for all subplots
ax[0].set_xticks(np.arange(len(top_corr_both_gender)))
ax[0].set_xticklabels(top_corr_both_gender['regions'], rotation=45, ha='right', fontsize=24)
# Step 3: Adjust the x-axis limits to start closer to the y-axis
# Set the x-axis limits to start from 0.5 and end at len(regions) - 0.5
ax[0].set_xlim(-0.5, len(top_corr_both_gender) - 0.5)
# Move x-ticks to the top and set the label on top
# Move ticks to the top
# ax[0].xaxis.set_ticks_position('top')  
# Show ticks only at the top 
# ax[0].tick_params(axis='x', which='both', labeltop=False, labelbottom=True)    

ax[1].set_xticks(np.arange(len(top_corr_male)))
ax[1].set_xticklabels(top_corr_male['regions'], rotation=45, ha='right', fontsize=24)
ax[0].set_xlim(-0.5, len(top_corr_male) - 0.5)

ax[2].set_xticks(np.arange(len(top_corr_female)))
ax[2].set_xticklabels(top_corr_female['regions'], rotation=45, ha='right', fontsize=24)
ax[2].set_xlim(-0.5, len(top_corr_female) - 0.5)

###############################################################################
if corr_target == "hgs_corrected_delta":
    # Set y-axis limits (you can adjust these values based on your data)
    ymin = -0.3  # Example minimum value
    ymax = -0.1  # Example maximum value
    ystep = 0.05
    # Remove the top and right spines (borders)
    # Move x-ticks to the top and set the label on top
    # ax[0].xaxis.set_ticks_position('top')  # Move ticks to the top
    # ax[0].tick_params(axis='x', which='both', labeltop=True, labelbottom=False)  # Show ticks only at the top        

############################################################################### 
for i in range(3): 
    ax[i].set_yticks(np.arange(ymin, ymax+0.05, ystep))
    ax[i].set_ylim(ymin-0.01, ymax)
    # Set y-tick labels font size
    ax[i].yaxis.set_tick_params(labelsize=24)
    # Add labels and title
    ax[i].set_xlabel('')
    ax[i].set_ylabel('correlations', fontsize=28)
    
    # ax[i].spines['bottom'].set_visible(False)
    # ax[i].spines['right'].set_visible(False)
    # Change spines colors to light grey
    ax[i].spines['top'].set_color('lightgrey')
    ax[i].spines['left'].set_color('lightgrey')
    ax[i].spines['right'].set_color('lightgrey')
    ax[i].spines['bottom'].set_color('lightgrey')
###############################################################################
fig.suptitle(f'Top {n_top_regions} Regions with Highest Absolute Correlations', fontsize=24, y=1)
###############################################################################
plt.tight_layout()
# Show the plot
plt.show()
plt.savefig(f"all_genders_{corr_target}_{n_top_regions}_top_regions.png")

print("===== Done! =====")
embed(globals(), locals())
