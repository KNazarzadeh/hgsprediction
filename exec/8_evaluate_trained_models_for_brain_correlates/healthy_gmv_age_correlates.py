import os
import pandas as pd
import numpy as np
import sys
from hgsprediction.load_results import healthy
from hgsprediction.load_results import load_trained_models
from hgsprediction.define_features import define_features
from hgsprediction.load_imaging_data import load_imaging_data
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import datatable as dt
from hgsprediction.extract_data import stroke_extract_data
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation
from hgsprediction.save_results.healthy import save_spearman_correlation_results, \
                                               save_hgs_predicted_results
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation_on_brain_correlations
from hgsprediction.save_results import save_data_overlap_hgs_predicted_brain_correlations_results,\
                                       save_spearman_correlation_on_brain_correlations_results
from sklearn.metrics import r2_score 
import statsmodels.stats.multitest as sm
from statsmodels.stats import multitest

from scipy.stats import linregress
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests
# from hgsprediction.plots import create_regplot
from nilearn import datasets
import nibabel as nib

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
model_name = sys.argv[3]
feature_type = sys.argv[4]
target = sys.argv[5]
session = sys.argv[6]
brain_correlation_type = sys.argv[7]

###############################################################################
jay_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "brain_imaging_data",
    f"{brain_correlation_type.upper()}",
)

schaefer_file = os.path.join(
    jay_path,
    f"{brain_correlation_type.upper()}_Schaefer100_Mean.jay")
feature_dt_schaefer = dt.fread(schaefer_file)
feature_df_schaefer = feature_dt_schaefer.to_pandas()
feature_df_schaefer.set_index('SubjectID', inplace=True)

tian_file = os.path.join(
    jay_path,
    f"4_gmd_tianS1_all_subjects.jay")
feature_dt_tian = dt.fread(tian_file)
feature_df_tian = feature_dt_tian.to_pandas()
feature_df_tian.set_index('SubjectID', inplace=True)

df_brain_correlation = pd.concat([feature_df_schaefer, feature_df_tian], axis=1)

if brain_correlation_type == "gmv":
    suit_file = os.path.join(
        jay_path,
        f"{brain_correlation_type.upper()}_SUIT_Mean.jay")
    feature_dt_suit = dt.fread(suit_file)
    feature_df_suit = feature_dt_suit.to_pandas()
    feature_df_suit.set_index('SubjectID', inplace=True)
    df_brain_correlation = pd.concat([df_brain_correlation, feature_df_suit], axis=1)

    
df_brain_correlation = df_brain_correlation.dropna()
df_brain_correlation.index = df_brain_correlation.index.str.replace("sub-", "")
df_brain_correlation.index = df_brain_correlation.index.map(int)

##############################################################################

# load data
df = healthy.load_hgs_predicted_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "both_gender",
    session,
)
# print("===== Done! =====")
# embed(globals(), locals())

df = df[df['1st_scan_hgs_dominant_side']=="right"]

df_intersected = df[df.index.isin(df_brain_correlation.index)]
# Find the intersection of indexes
df_brain_correlation_overlap = df_brain_correlation[df_brain_correlation.index.isin(df.index)]
df_brain_correlation_overlap = df_brain_correlation_overlap.reindex(df_intersected.index)

intersection_index_female = df_intersected[df_intersected["gender"]==0].index
intersection_index_male = df_intersected[df_intersected["gender"]==1].index

df_intersected_female = df_intersected[df_intersected.index.isin(intersection_index_female)]
df_intersected_male = df_intersected[df_intersected.index.isin(intersection_index_male)]

df_brain_correlation_overlap_female = df_brain_correlation_overlap[df_brain_correlation_overlap.index.isin(intersection_index_female)]
df_brain_correlation_overlap_male = df_brain_correlation_overlap[df_brain_correlation_overlap.index.isin(intersection_index_male)]

print("===== Done! =====")
embed(globals(), locals())
##############################################################################
##############################################################################
fig, (ax0, ax1) = plt.subplots(1,2, figsize=(8,8))
# Set aspect ratio to be equal
ax0.set_box_aspect(1)
ax1.set_box_aspect(1)
sns.regplot(data=df_intersected_female, x="1st_scan_age", y="1st_scan_hgs_right_true", color="red", ax=ax0, line_kws={"color": "darkgrey"})
sns.regplot(data=df_intersected_male, x="1st_scan_age", y="1st_scan_hgs_right_true", color="#069AF3", ax=ax1, line_kws={"color": "darkgrey"})
ax0.set_title('Females', fontsize=14, fontweight='bold')
ax1.set_title('Males', fontsize=14, fontweight='bold')

plt.show()
plt.savefig("hgs_age.png")

##############################################################################
##############################################################################
correlation_values = pd.DataFrame(columns=["regions", "correlations", "p_values"])
correlation_values_female = pd.DataFrame(columns=["regions", "correlations_female", "p_values_female"])
correlation_values_male = pd.DataFrame(columns=["regions", "correlations_male", "p_values_male"])

for i, region in enumerate(df_brain_correlation.columns):
    # Compute correlations and p-values for all, female, and male datasets
    corr, p_value = pearsonr(df_brain_correlation_overlap.iloc[:, i].values.ravel(), df_intersected['1st_scan_age'].values.ravel())
    correlation_values.loc[i, "regions"] = region
    correlation_values.loc[i, "correlations"] = corr
    correlation_values.loc[i, "p_values"] = p_value

    corr_female, p_value_female = pearsonr(df_brain_correlation_overlap_female.iloc[:, i].values.ravel(), df_intersected_female['1st_scan_age'].values.ravel())
    correlation_values_female.loc[i, "regions"] = region
    correlation_values_female.loc[i, "correlations_female"] = corr_female
    correlation_values_female.loc[i, "p_values_female"] = p_value_female
    
    corr_male, p_value_male = pearsonr(df_brain_correlation_overlap_male.iloc[:, i].values.ravel(), df_intersected_male['1st_scan_age'].values.ravel())
    correlation_values_male.loc[i, "regions"] = region
    correlation_values_male.loc[i, "correlations_male"] = corr_male
    correlation_values_male.loc[i, "p_values_male"] = p_value_male


# Perform FDR correction on p-values
reject, p_corrected, _, _ = multitest.multipletests(correlation_values['p_values'], method='fdr_bh')
reject_female, p_corrected_female, _, _ = multitest.multipletests(correlation_values_female['p_values_female'], method='fdr_bh')
reject_male, p_corrected_male, _, _ = multitest.multipletests(correlation_values_male['p_values_male'], method='fdr_bh')

# Add corrected p-values and significance indicator columns to dataframes
correlation_values['pcorrected'] = p_corrected
correlation_values['significant'] = reject

correlation_values_female['pcorrected_female'] = p_corrected_female
correlation_values_female['significant_female'] = reject_female

correlation_values_male['pcorrected_male'] = p_corrected_male
correlation_values_male['significant_male'] = reject_male

correlation_values_sorted = correlation_values[correlation_values['significant']].sort_values(by='pcorrected', ascending=True)
correlation_values_sorted_female = correlation_values_female[correlation_values_female['significant_female']].sort_values(by='pcorrected_female', ascending=True)
correlation_values_sorted_male= correlation_values_male[correlation_values_male['significant_male']].sort_values(by='pcorrected_male', ascending=True)

##############################################################################
##############################################################################

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
# Plot histogram for significant correlations in females
sns.histplot(data=correlation_values_female[correlation_values_female['significant_female']==True], x="correlations_female", ax=axes[0], color='red')
axes[0].set_title('Significant Correlations in Females')

# Plot histogram for significant correlations in males
sns.histplot(data=correlation_values_male[correlation_values_male['significant_male']==True], x="correlations_male", ax=axes[1], color="#069AF3")
axes[1].set_title('Significant Correlations in Males')
# Show the plot
plt.tight_layout()
plt.show()

# Save the plot
fig.savefig("hist_age_r_value_gmv.png")


##############################################################################
##############################################################################

plt.figure(figsize=(40, 10))

plt.bar(correlation_values_sorted['regions'], correlation_values_sorted['pcorrected'], color='darkgrey', width=0.2)  # Plot the bars
plt.scatter(correlation_values_sorted['regions'], correlation_values_sorted['pcorrected'], color='orange', zorder=5)  # Overlay scatter points
plt.xlabel('Regions', fontsize=20, fontweight='bold')
plt.ylabel('p-values', fontsize=20, fontweight='bold')
plt.title('Age vs GMV regions (150)')
# Get the current x-axis tick positions and labels
current_tick_positions, current_tick_labels = plt.xticks()
# Convert tick positions to integers
current_tick_positions = [int(pos) for pos in current_tick_positions]
# Add 5 units to each x-coordinate
new_tick_positions = [pos + 10 for pos in current_tick_positions]
plt.xticks(new_tick_positions, current_tick_labels, rotation=90)  # Set x-axis tick labels with modified positions

plt.yticks(fontsize=20, fontweight='bold')  # Set y-axis tick labels with fontsize and fontweight
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better visualization
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.savefig("gmv_age_both_genders.png")  # Save the plot as a PNG file
plt.show()  # Display the plot

##############################################################################
##############################################################################
plt.figure(figsize=(40, 10))
plt.bar(correlation_values_sorted_female['regions'], correlation_values_sorted_female['pcorrected_female'], color='darkgrey', width=0.2)  # Plot the bars
plt.scatter(correlation_values_sorted_female['regions'], correlation_values_sorted_female['pcorrected_female'], color='orange', zorder=5)  # Overlay scatter points
plt.xlabel('Regions', fontsize=20, fontweight='bold')
plt.ylabel('p-values', fontsize=20, fontweight='bold')
plt.title('Female - Age vs GMV regions (150)')
# Get the current x-axis tick positions and labels
current_tick_positions, current_tick_labels = plt.xticks()
# Convert tick positions to integers
current_tick_positions = [int(pos) for pos in current_tick_positions]
# Add 5 units to each x-coordinate
new_tick_positions = [pos + 10 for pos in current_tick_positions]
plt.xticks(new_tick_positions, current_tick_labels, rotation=90)  # Set x-axis tick labels with modified positionsplt.yticks(fontsize=20, fontweight='bold')  # Set y-axis tick labels with fontsize and fontweight
plt.yticks(fontsize=20, fontweight='bold')  # Set y-axis tick labels with fontsize and fontweight

plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better visualization
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.savefig("gmv_age_female.png")  # Save the plot as a PNG file
plt.show()  # Display the plot

##############################################################################
##############################################################################
plt.figure(figsize=(40, 10))

plt.bar(correlation_values_sorted_male['regions'], correlation_values_sorted_male['pcorrected_male'], color='darkgrey', width=0.2)  # Plot the bars
plt.scatter(correlation_values_sorted_male['regions'], correlation_values_sorted_male['pcorrected_male'], color='orange', zorder=5)  # Overlay scatter points
plt.xlabel('Regions', fontsize=20, fontweight='bold')
plt.ylabel('p-values', fontsize=20, fontweight='bold')
plt.title('Male - Age vs GMV regions (150)')
# Get the current x-axis tick positions and labels
current_tick_positions, current_tick_labels = plt.xticks()
# Convert tick positions to integers
current_tick_positions = [int(pos) for pos in current_tick_positions]
# Add 5 units to each x-coordinate
new_tick_positions = [pos + 10 for pos in current_tick_positions]
plt.xticks(new_tick_positions, current_tick_labels, rotation=90)  # Set x-axis tick labels with modified positionsplt.yticks(fontsize=20, fontweight='bold')  # Set y-axis tick labels with fontsize and fontweight
plt.yticks(fontsize=20, fontweight='bold')  # Set y-axis tick labels with fontsize and fontweight

plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines for better visualization
plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.savefig("gmv_age_male.png")  # Save the plot as a PNG file
plt.show()  # Display the plot


print("===== Done! =====")
embed(globals(), locals())
