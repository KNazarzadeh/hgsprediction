import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize

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
df_sorted = df_both_gender_corr_intersected.sort_values(by='average_correlations', ascending=False)

###############################################################################
# Apply the custom colormap
cmap_custom = plt.cm.RdBu_r

# Normalize the 'Age' column to the range [-0.32, 0.32]
norm = Normalize(vmin=-0.32, vmax=0.32)

# Normalize the Age values to the range [-0.32, 0.32] for coloring
normalized_values = norm(df_sorted['average_correlations'])# Normalize the Age values to the range [-0.32, 0.32] for coloring

# Plot the barplot
# Create a color map for the bar colors based on normalized values
colors = cmap_custom(normalized_values)

# Plot the barplot
plt.figure(figsize=(8, 6))
bars = plt.bar(df_sorted['regions'], df_sorted['average_correlations'], color=colors)

# Show the plot
plt.show()
plt.savefig("both_gender_intersection_regions.png")


print("===== Done! =====")
embed(globals(), locals())
