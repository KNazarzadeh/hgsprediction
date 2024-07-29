
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
session = sys.argv[13]
stats_correlation_type = sys.argv[14]
corr_target = sys.argv[15]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

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
print("===== Done! =====")
embed(globals(), locals())
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

###############################################################################
df_female_survived = df_female_corr[df_female_corr["significant"]==True]
df_male_survived = df_male_corr[df_male_corr["significant"]==True]
print(len(df_female_survived))
print(len(df_male_survived))
###############################################################################
female_corr_threashold = df_female_survived[abs(df_female_survived["correlations"])>.1]
male_corr_threashold = df_male_survived[abs(df_male_survived["correlations"])>.1]
print(len(female_corr_threashold))
print(len(male_corr_threashold))
###############################################################################
matching_regions = np.intersect1d(female_corr_threashold['regions'].unique(), male_corr_threashold['regions'].unique())
print(len(matching_regions))
###############################################################################
filtered_rows = df_both_gender_corr[df_both_gender_corr["regions"].isin(matching_regions)]
print(filtered_rows)
###############################################################################
#  Set values in column 'correlations' to 0 for rows where 'regions' is not in matching_regions
df_both_gender_corr.loc[~df_both_gender_corr.index.isin(filtered_rows.index), 'average_correlations'] = 0
###############################################################################
# Sort the DataFrame from high to low based on 'average_correlations'
# Assuming df_both_gender_corr is your DataFrame
# Filter out rows where 'average_correlations' is zero
df_non_zero = df_both_gender_corr[df_both_gender_corr['average_correlations'] != 0]
###############################################################################
sorted_df = df_non_zero.reindex(df_non_zero['average_correlations'].abs().sort_values(ascending=False).index)
print(sorted_df)
###############################################################################
# Select the top 10 regions
top_10_regions = sorted_df.head(10)
# Extract the 'regions' column from the top 10
top_10_regions_list = top_10_regions['regions'].tolist()
print(top_10_regions)
###############################################################################
plot_folder = os.path.join(os.getcwd(), f"plots/box_violinplot/{target}/{n_repeats}_repeats_{n_folds}_folds/{score_type}")
if(not os.path.isdir(plot_folder)):
        os.makedirs(plot_folder)
plot_file = os.path.join(plot_folder, f"comparing_SVM_RF_models_multi_samplesize_by_gender_{target}.png")
###############################################################################
# Plotting
def plot_bar_with_scatter(data, x, y, corr_target, gender, n_regions_survived, color):
    if len(brain_df.columns) == 150:
        plt.figure(figsize=(40, 10))
    else:
        plt.figure(figsize=(250, 10))
    sns.barplot(data=data, x=x, y=y, color='darkgrey', errorbar=None, width=0.3)
    sns.scatterplot(data=data, x=x, y=y, color=color, zorder=5, s=100)
    plt.xlabel(x.capitalize(), fontsize=20, fontweight='bold')
    plt.ylabel(f'Correlations {corr_target} HGS vs brain regions', fontsize=20, fontweight='bold')
    plt.title(f"{gender.capitalize()} - {corr_target.capitalize()} {target.replace('hgs_', '')} HGS vs brain regions - survived regions({n_regions_survived}/{len(brain_df.columns)})-Schaefer{schaefer}", fontsize=20, fontweight='bold')
    plt.xticks(rotation=90, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(-0.5, len(data[x]) - 0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"corr_gmv_without_TIV_schaefer{schaefer}_{stats_correlation_type}_{model_name}_{corr_target}_{target}_{gender}.png")  # Save the plot as a PNG file
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################