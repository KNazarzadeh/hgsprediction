#!/usr/bin/env Disorderspredwp3
"""
Predict the motor based on specific features on populations data.
Motor is Handgrip strength (1 phase).

# Author:  Kimia Nazarzadeh <k.nazarzadeh@fz-juelich.de>

# License: AGPL

"""
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from nilearn import image, plotting, datasets
import nibabel as nib

from hgsprediction.load_results.load_brain_correlation_results import load_brain_hgs_correlation_results_for_plot
#--------------------------------------------------------------------------#
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
# Parse, add and return the arguments by function parse_args.
###############################################################################
filename = sys.argv[0]
brain_data_type = sys.argv[1]
schaefer = sys.argv[2]
stats_correlation_type = sys.argv[3]
corr_target = sys.argv[4]
# print("===== Done! =====")
# embed(globals(), locals())
    
df_female_corr,df_male_corr = load_brain_hgs_correlation_results_for_plot(
brain_data_type,
schaefer,
corr_target, 
)
print("===== Done! =====")
embed(globals(), locals())
# Schaefer
# Fetch the Schaefer atlas
# Specify the directory where you want to download the atlas
download_path = '/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/brain_imaging_data/'
# Download the atlas to the specified directory
cort_img = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=1, data_dir=download_path, resume=True, verbose=1)
nii_gmv_path = '/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/brain_imaging_data/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii'
cort_img = nib.load(nii_gmv_path)
# Tian
nii_tian_file_path = '/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/brain_imaging_data/Tian_Subcortex_S1_3T.nii'
sub_img = nib.load(nii_tian_file_path)
# SUIT
nii_suit_file_path = '/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/brain_imaging_data/SUIT_MNISpace_1mm.nii'
cer_img = nib.load(nii_suit_file_path)
# Schaefer
# Fetch the Schaefer atlas
# Specify the directory where you want to download the atlas
for gender in ["female", "male"]:
    if gender == "female":
        df_tmp = df_female_corr.copy()
    elif gender == "male":
        df_tmp = df_male_corr.copy()
    
    for cnfd_idx, cnfd in enumerate(list(df_tmp.columns)[:2]):
        print(cnfd_idx, cnfd)
        # Get correlation data per atlas
        # cortical
        cort_corr = df_tmp.iloc[:100, cnfd_idx-1].copy()
        cort_corr_newidx = cort_corr.copy()
        cort_corr_newidx.reset_index()
        # subcortical Tian
        sub_corr = df_tmp.iloc[100:116, cnfd_idx-1].copy()
        sub_corr_newidx = sub_corr.copy()
        sub_corr_newidx.reset_index()
        # cerebellar SUID
        cer_corr = df_tmp.iloc[116:, cnfd_idx-1].copy()
        cer_corr_newidx = cer_corr.copy()
        cer_corr_newidx.reset_index()
        
        # Get atlas data as array
        # Then load the images
        # Assuming cort_img is the path to your NIfTI file
        cort_data = cort_img.get_fdata().copy()
        sub_data = sub_img.get_fdata().copy()
        cer_data = cer_img.get_fdata().copy()  # TODO try to copy either cer_img before getting data or copy it by using new_img_like before getting data and see if then after get_data is no memmap anymore
    
        # replace ROI number with correlation value (ROI idx and label order same)
        for i in range(1, 101):
            cort_data[cort_data == i] = cort_corr_newidx.iloc[i-1]
        for j in range(1, 17):
            sub_data[sub_data == j] = sub_corr_newidx.iloc[j-1]    
        for k in range(1, 35):
            cer_data[cer_data == k] = cer_corr_newidx.iloc[k-1]
        
        # Make nifti from replaced data array
        cort_corr_nii = image.new_img_like(cort_img, cort_data)
        sub_corr_nii = image.new_img_like(sub_img, sub_data)
        cer_corr_nii = image.new_img_like(cer_img, cer_data)

        # figure saving names
        folder_path = f"/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/brain_imaging_data/brain_plots/schaefer{schaefer}/{gender}/{corr_target}"
            
        cort_corr_fig_fname =  os.path.join(folder_path, f"schaefer100_correlation-GMV-{cnfd}_2.pdf")
        sub_corr_fig_fname =  os.path.join(folder_path, f"tian_correlation-GMV-{cnfd}_2.pdf")
        cer_corr_fig_fname =  os.path.join(folder_path, f"suit_correlation-GMV{cnfd}_2.pdf")

        # plot in MNI standard tempalte
        display = plotting.plot_stat_map(
            cort_corr_nii, colorbar=True,
            title=f'Correlation cortical GMV with {cnfd}')
        plt.savefig(cort_corr_fig_fname, bbox_inches='tight')

        display = plotting.plot_stat_map(
            sub_corr_nii, colorbar=True,
            title=f'Correlation subcortical GMV with {cnfd}')
        plt.savefig(sub_corr_fig_fname, bbox_inches='tight')
        
        display = plotting.plot_stat_map(
            cer_corr_nii, colorbar=True,
            title=f'Correlation cerebellar GMV with {cnfd}')
        plt.savefig(cer_corr_fig_fname, bbox_inches='tight')

        display.add_overlay(cer_corr_nii) # , cmap=plotting.cm.purple_green)
        display.add_overlay(sub_corr_nii, cmap=plotting.cm.purple_green)
        plotting.show()

        # Save correlation nifti
        folder_path = f"/data/project/stroke_ukb/knazarzadeh/project_hgsprediction/brain_imaging_data/brain_plots/schaefer{schaefer}/{gender}/{corr_target}"
        cort_corr_atlas_fname = os.path.join(folder_path, f"schaefer100_correlation-GMV-{cnfd}.nii")
        sub_corr_atlas_fname = os.path.join(folder_path, f"tian_correlation-GMV-{cnfd}.nii")
        cer_corr_atlas_fname = os.path.join(folder_path, f"suit_correlation-GMV{cnfd}.nii")

        nib.save(cort_corr_nii, cort_corr_atlas_fname)
        nib.save(sub_corr_nii, sub_corr_atlas_fname)
        nib.save(cer_corr_nii, cer_corr_atlas_fname)

print("===== Done! =====")
embed(globals(), locals())