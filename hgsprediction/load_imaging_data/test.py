

import numpy as np
import pandas as pd
import nibabel as nib
import os
import datalad.api as dl
from scipy.stats import pearsonr
from nilearn.datasets import fetch_atlas_schaefer_2018
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

# Fetch the Schaefer 400 atlas
atlas_schaefer = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)

# Load the local correlation map for one subject
nifti_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "data_ukk",
    "tmp",
    "ukb_fc_metrix",
    "LCOR",
)

file_path = os.path.join(
        nifti_path,
        'LCOR_sub_2337058.nii')

# Load the local correlation map for one subject
img = nib.load(file_path)

# Get the local correlation map data array
lcor_map_data = img.get_fdata()

# Get the atlas data array
atlas_data = atlas_schaefer.maps

# Calculate the mean LC value for each atlas label
# label_values = []
# for label in range(0, 399):
#     # Find voxels for this label
#     label_voxels = np.where(atlas_data == label)
#     # Get the LC values for the voxels in this label
#     lc_values = lcor_map_data[label_voxels]
#     # Check for NaN values
#     if np.isnan(lc_values).any():
#         # Skip this label if it contains NaN values
#         continue
#     # Calculate the mean LC value for this label
#     label_mean = np.mean(lc_values)
#     # Append the label mean to the list of label values
#     label_values.append(label_mean)

# # Convert the label values to a numpy array
# label_values = np.array(label_values)

# # Save the label values as a 400x1 vector
# np.savetxt('subject_label_values.txt', label_values)

anthro_data = pd.read_csv('anthro_data.csv')

# Calculate the difference between actual and predicted HGS
# actual_hgs = anthro_data['actual_hgs']
# predicted_hgs = anthro_data['predicted_hgs']
# hgs_diff = actual_hgs - predicted_hgs

# Extract LCOR values for each region
n_regions = 400
lcor_values = np.zeros((n_regions, len(hgs_diff)))
for region in range(n_regions):
    region_voxels = np.where(lcor_data == region + 1)
    region_lcor = lcor_data[region_voxels]
    lcor_values[region, :] = region_lcor

# Calculate correlations between LCOR and HGS difference
correlations = np.zeros(n_regions)
p_values = np.zeros(n_regions)
for region in range(n_regions):
    corr, p_value = pearsonr(lcor_values[region, :], hgs_diff)
    correlations[region] = corr
    p_values[region] = p_value

print("===== Done! =====")
embed(globals(), locals())
# # Define a list to store the features for each subject
# features = []
# # Loop through each subject and extract the LCOR features
# for subj in range(1, 2):
#     # Load the LCOR map from a NIfTI file for the current subject
#     lcor_img = nib.load(f'subject_{subj}_lcor_map.nii.gz')
#     lcor_data = lcor_img.get_fdata()
# # Extract the mean, standard deviation, and median features for the current subject
#     mean_lcor = np.mean(lcor_data)
#     std_lcor = np.std(lcor_data)
#     median_lcor = np.median(lcor_data)

#     # Append the features to the list for the current subject
#     features.append([mean_lcor, std_lcor, median_lcor])

# # Convert the list of features to a NumPy array
# features = np.array(features)

print("===== Done! =====")
embed(globals(), locals())