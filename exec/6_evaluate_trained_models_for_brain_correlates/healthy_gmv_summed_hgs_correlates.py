import os
import pandas as pd
import numpy as np
import sys
from hgsprediction.load_results import healthy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import datatable as dt
from hgsprediction.predict_hgs import calculate_spearman_hgs_correlation_on_brain_correlations                    
from hgsprediction.predict_hgs import calculate_t_valuesGMV_HGS

from sklearn.metrics import r2_score 
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests

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
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
brain_correlation_type = sys.argv[10]

###############################################################################
# Fetch the Schaefer 2018 atlas with 100 regions and Yeo networks set to 17
atlas = datasets.fetch_atlas_schaefer_2018(
    n_rois=100, yeo_networks=17, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)

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

# feature_dt = dt.fread(fname.as_posix())

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
    confound_status,
    n_repeats,
    n_folds,    
)
print("===== Done! =====")
embed(globals(), locals())

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

# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
##############################################################################
##############################################################################
n_regions = df_brain_correlation.shape[1]
x_axis = df_brain_correlation.columns

correlation_values_true, correlation_values_true_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap, df_intersected, f"1st_scan_{target}_true", x_axis)
correlation_values_true_female, correlation_values_true_female_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_female, df_intersected_female, f"1st_scan_{target}_true", x_axis)
correlation_values_true_male, correlation_values_true_male_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_male, df_intersected_male, f"1st_scan_{target}_true", x_axis)

correlation_values_predicted, correlation_values_predicted_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap, df_intersected, f"1st_scan_{target}_predicted", x_axis)
correlation_values_predicted_female, correlation_values_predicted_female_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_female, df_intersected_female, f"1st_scan_{target}_predicted", x_axis)
correlation_values_predicted_male, correlation_values_predicted_male_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_male, df_intersected_male, f"1st_scan_{target}_predicted", x_axis)

correlation_values_delta, correlation_values_delta_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap, df_intersected, f"1st_scan_{target}_(true-predicted)", x_axis)
correlation_values_delta_female, correlation_values_delta_female_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_female, df_intersected_female, f"1st_scan_{target}_(true-predicted)", x_axis)
correlation_values_delta_male, correlation_values_delta_male_significants = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_male, df_intersected_male, f"1st_scan_{target}_(true-predicted)", x_axis)



merged_true_delta_female = pd.merge(correlation_values_true_female, correlation_values_delta_female, how='inner', left_on='regions', right_on='regions', suffixes=('_true', '_delta'))
merged_true_predicted_female = pd.merge(correlation_values_true_female, correlation_values_predicted_female, how='inner', left_on='regions', right_on='regions', suffixes=('_true', '_predicted'))
merged_true_delta_male = pd.merge(correlation_values_true_male, correlation_values_delta_male, how='inner', left_on='regions', right_on='regions', suffixes=('_true', '_delta'))
merged_true_predicted_male = pd.merge(correlation_values_true_male, correlation_values_predicted_male, how='inner', left_on='regions', right_on='regions', suffixes=('_true', '_predicted'))

# Calculate correlation coefficient
correlation_true_delta_female = np.corrcoef(merged_true_delta_female["correlations_true"].astype(float), merged_true_delta_female["correlations_delta"].astype(float))[0, 1]
correlation_true_predicted_female = np.corrcoef(merged_true_predicted_female["correlations_true"].astype(float), merged_true_predicted_female["correlations_predicted"].astype(float))[0, 1]
correlation_true_delta_male = np.corrcoef(merged_true_delta_male["correlations_true"].astype(float), merged_true_delta_male["correlations_delta"].astype(float))[0, 1]
correlation_true_predicted_male = np.corrcoef(merged_true_predicted_male["correlations_true"].astype(float), merged_true_predicted_male["correlations_predicted"].astype(float))[0, 1]

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot on each subplot
sns.regplot(x=merged_true_delta_female["correlations_true"].astype(float), y=merged_true_delta_female["correlations_delta"].astype(float), ax=axs[0, 0], color="red")
axs[0, 0].set_title('Female True vs. Delta')
axs[0, 0].set_xlabel('Correlation GMV vs True HGS', fontsize=14, fontweight='bold')
axs[0, 0].set_ylabel('Correlation GMV vs (True-Predicted) HGS', fontsize=14, fontweight='bold')
axs[0, 0].annotate(f'r = {correlation_true_delta_female:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
# Plot regression line
xmin, xmax = axs[0, 0].get_xlim()
ymin, ymax = axs[0, 0].get_ylim()
axs[0, 0].plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

sns.regplot(x=merged_true_predicted_female["correlations_true"].astype(float), y=merged_true_predicted_female["correlations_predicted"].astype(float), ax=axs[0, 1], color="red")
axs[0, 1].set_title('Female True vs. Predicted')
axs[0, 1].set_xlabel('Correlation GMV vs True HGS', fontsize=14, fontweight='bold')
axs[0, 1].set_ylabel('Correlation GMV vs Predicted HGS', fontsize=14, fontweight='bold')
axs[0, 1].annotate(f'r = {correlation_true_predicted_female:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
# Plot regression line
xmin, xmax = axs[0, 1].get_xlim()
ymin, ymax = axs[0, 1].get_ylim()
axs[0, 1].plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

sns.regplot(x=merged_true_delta_male["correlations_true"].astype(float), y=merged_true_delta_male["correlations_delta"].astype(float), ax=axs[1, 0], color="#069AF3")
axs[1, 0].set_title('Male True vs. Delta')
axs[1, 0].set_xlabel('Correlation GMV vs True HGS', fontsize=14, fontweight='bold')
axs[1, 0].set_ylabel('Correlation GMV vs (True-Predicted) HGS', fontsize=14, fontweight='bold')
axs[1, 0].annotate(f'r = {correlation_true_delta_male:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
# Plot regression linexmin, xmax = axs[1, 0].get_xlim()
ymin, ymax = axs[1, 0].get_ylim()
xmin, xmax = axs[1, 0].get_xlim()
axs[1, 0].plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

sns.regplot(x=merged_true_predicted_male["correlations_true"].astype(float), y=merged_true_predicted_male["correlations_predicted"].astype(float), ax=axs[1, 1], color="#069AF3")
axs[1, 1].set_title('Male True vs. Predicted')
axs[1, 1].set_xlabel('Correlation GMV vs True HGS', fontsize=14, fontweight='bold')
axs[1, 1].set_ylabel('Correlation GMV vs Predicted HGS', fontsize=14, fontweight='bold')
axs[1, 1].annotate(f'r = {correlation_true_predicted_male:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
# Plot regression line
xmin, xmax = axs[1, 1].get_xlim()
ymin, ymax = axs[1, 1].get_ylim()
axs[1, 1].plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

# Show the plot
plt.tight_layout()
plt.savefig(f"hgs_gmv_{target}.png")
plt.show()

##############################################################################
##############################################################################
# Set the significance level
significance_level = 0.01
t_values_true, t_values_true_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap, df_intersected, f"1st_scan_{target}_true", x_axis, significance_level)
t_values_true_female, t_values_true_female_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap_female, df_intersected_female, f"1st_scan_{target}_true", x_axis, significance_level)
t_values_true_male, t_values_true_male_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap_male, df_intersected_male, f"1st_scan_{target}_true", x_axis, significance_level)

t_values_predicted, t_values_predicted_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap, df_intersected, f"1st_scan_{target}_predicted", x_axis, significance_level)
t_values_predicted_female, t_values_predicted_female_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap_female, df_intersected_female, f"1st_scan_{target}_predicted", x_axis, significance_level)
t_values_predicted_male, t_values_predicted_male_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap_male, df_intersected_male, f"1st_scan_{target}_predicted", x_axis, significance_level)

t_values_delta, t_values_delta_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap, df_intersected, f"1st_scan_{target}_(true-predicted)", x_axis, significance_level)
t_values_delta_female, t_values_delta_female_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap_female, df_intersected_female, f"1st_scan_{target}_(true-predicted)", x_axis, significance_level)
t_values_delta_male, t_values_delta_male_significants = calculate_t_valuesGMV_HGS(df_brain_correlation_overlap_male, df_intersected_male, f"1st_scan_{target}_(true-predicted)", x_axis, significance_level)


##############################################################################
##############################################################################

# Plotting
sorted_t_values = t_values_true_significants.sort_values(by='t_values', ascending=False)
sorted_t_values_female = t_values_true_female_significants.sort_values(by='t_values', ascending=False)
sorted_t_values_male = t_values_true_male_significants.sort_values(by='t_values', ascending=False)
##############################################################################
##############################################################################
plt.figure(figsize=(40, 10))

plt.bar(sorted_t_values['regions'], sorted_t_values['t_values'], color='darkgrey', width=0.2)  # Plot the bars
plt.scatter(sorted_t_values['regions'], sorted_t_values['t_values'], color='orange', zorder=5)  # Overlay scatter points
plt.xlabel('Regions', fontsize=20, fontweight='bold')
plt.ylabel('T-values', fontsize=20, fontweight='bold')
plt.title('Right dominanct HGS vs brain regions (150)')
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
plt.savefig(f"gmv_{target}.png")  # Save the plot as a PNG file
plt.show()  # Display the plot

##############################################################################
##############################################################################
plt.figure(figsize=(40, 10))
plt.bar(sorted_t_values_female['regions'], sorted_t_values_female['t_values'], color='darkgrey', width=0.2)  # Plot the bars
plt.scatter(sorted_t_values_female['regions'], sorted_t_values_female['t_values'], color='orange', zorder=5)  # Overlay scatter points
plt.xlabel('Regions', fontsize=20, fontweight='bold')
plt.ylabel('T-values', fontsize=20, fontweight='bold')
plt.title('Female - Right dominanct HGS vs brain regions (150)')
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
plt.savefig(f"gmv_{target}_female.png")  # Save the plot as a PNG file
plt.show()  # Display the plot

##############################################################################
##############################################################################
plt.figure(figsize=(40, 10))

plt.bar(sorted_t_values_male['regions'], sorted_t_values_male['t_values'], color='darkgrey', width=0.2)  # Plot the bars
plt.scatter(sorted_t_values_male['regions'], sorted_t_values_male['t_values'], color='orange', zorder=5)  # Overlay scatter points
plt.xlabel('Regions', fontsize=20, fontweight='bold')
plt.ylabel('T-values', fontsize=20, fontweight='bold')
plt.title('Male - Right dominanct HGS vs brain regions (150)')
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
plt.savefig(f"gmv_{target}_male.png")  # Save the plot as a PNG file
plt.show()  # Display the plot

print("===== Done! =====")
embed(globals(), locals())
##############################################################################
"""
This script provides functions for neuroimaging data processing and analysis.
"""

def vec2image(
brain_maps_2_nii, 
atlas_filename,
output_filename
):
    """Converts ROI-wise brain maps to a 3D NIfTI image.

    Arguments:
    -----------
    brain_maps_2_nii: array of size N_ROI x N_measure
        2D array containing brain maps for multiple graph measures.
    atlas_filename: str or path
        Path to a parcellated brain atlas with N_ROI regions.
    output_filename: str or path
        Filename of the output file.

    Returns:
    --------
    brain_surf_map: str or path
        Filename of the converted 3D brain surface NIfTI file.
    """
    N_ROI, N_measure = brain_maps_2_nii.shape

    # Read the brain atlas image
    atlas_img = nib.load(atlas_filename)
    affine_mat = atlas_img.affine
    atlas_img = atlas_img.get_fdata()
    N_x, N_y, N_z = np.shape(atlas_img)
    
    unique_labels = np.unique(atlas_img.flatten())
    unique_labels = unique_labels[unique_labels != 0]
    
    brain_maps_img = np.zeros((N_x, N_y, N_z, N_measure))

    for n_roi in range(N_ROI):

        ind = np.where(atlas_img==unique_labels[n_roi])
        ix = ind[0]
        iy = ind[1]
        iz = ind[2]

        for n_meas in range(N_measure):
            brain_maps_img[ix, iy, iz, n_meas] = brain_maps_2_nii[n_roi, n_meas]

    brain_map_nii = nib.Nifti1Image(brain_maps_img, affine=affine_mat)
    brain_map_nii.to_filename(output_filename)

    return atlas_filename


output_filename = "/data/project/stroke_ukb/knazarzadeh/GIT_repositories/hgsprediction/exec/6_evaluate_trained_models_for_brain_correlates/shaefer100.nii"
atlas_filename = "/home/knazarzadeh/nilearn_data/schaefer_2018/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
# corr_female_100 = correlation_values_female[correlation_values_female['regions'].isin(feature_df_schaefer.columns)]
# brain_maps_2_nii = corr_female_100['correlations_female'].to_numpy()
# # Reshape the array from shape (150,) to (150, 1)
# brain_maps_2_nii = brain_maps_2_nii.reshape(-1, 1)


t_values_female_100 = t_values[t_values['regions'].isin(feature_df_schaefer.columns)]
brain_maps_2_nii = t_values_female_100['t_values_female'].to_numpy()
# Reshape the array from shape (150,) to (150, 1)
brain_maps_2_nii = brain_maps_2_nii.reshape(-1, 1)


atlas_file = vec2image(brain_maps_2_nii, atlas_filename, output_filename)