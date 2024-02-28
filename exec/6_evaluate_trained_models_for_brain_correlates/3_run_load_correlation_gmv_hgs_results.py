import os
import pandas as pd
import numpy as np
import sys
from hgsprediction.load_results import healthy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import datatable as dt
from hgsprediction.predict_hgs import calculate_brain_hgs                    
from sklearn.metrics import r2_score 
from scipy.stats import linregress
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression

from hgsprediction.load_results.load_brain_correlates_results import load_brain_overlap_data_results, load_brain_hgs_correlation_results
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
brain_data_type = sys.argv[10]
schaefer = sys.argv[11]
stats_correlation_type = sys.argv[12]
###############################################################################
true_corr_female = load_brain_hgs_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,
    brain_data_type,
    schaefer,
    "true_hgs",
)


true_corr_male = load_brain_hgs_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,
    brain_data_type,
    schaefer,
    "true_hgs",
)
###############################################################################
predicted_corr_female = load_brain_hgs_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,
    brain_data_type,
    schaefer,
    "predicted_hgs",
)


predicted_corr_male = load_brain_hgs_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,
    brain_data_type,
    schaefer,
    "predicted_hgs",
)
###############################################################################
delta_corr_female = load_brain_hgs_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,
    brain_data_type,
    schaefer,
    "delta_hgs",
)


delta_corr_male = load_brain_hgs_correlation_results(
    population,
    mri_status,
    model_name,
    feature_type,
    target,
    "male",
    session,
    confound_status,
    n_repeats,
    n_folds,
    brain_data_type,
    schaefer,
    "delta_hgs",
)


##############################################################################
true_corr_significant_female = true_corr_female[true_corr_female['significant']==True]
predicted_corr_significant_female = predicted_corr_female[predicted_corr_female['significant']==True]
delta_corr_significant_female = delta_corr_female[delta_corr_female['significant']==True]

true_corr_significant_male = true_corr_male[true_corr_male['significant']==True]
predicted_corr_significant_male = predicted_corr_male[predicted_corr_male['significant']==True]
delta_corr_significant_male = delta_corr_male[delta_corr_male['significant']==True]

##############################################################################
# Plotting True HGS vs GMV
sorted_p_values_true_female = true_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_true_male = true_corr_significant_male.sort_values(by='correlations', ascending=False)

##############################################################################
##############################################################################
# Plotting Predicted HGS vs GMV
sorted_p_values_predicted_female = predicted_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_predicted_male = predicted_corr_significant_male.sort_values(by='correlations', ascending=False)

##############################################################################
##############################################################################
# Plotting Delta HGS vs GMV
sorted_p_values_delta_female = delta_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_delta_male = delta_corr_significant_male.sort_values(by='correlations', ascending=False)

##############################################################################
##############################################################################

print("===== Done! =====")
embed(globals(), locals())