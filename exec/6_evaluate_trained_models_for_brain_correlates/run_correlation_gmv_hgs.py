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
brain_data_type = sys.argv[10]
schaefer = sys.argv[11]

###############################################################################
folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "project_hgsprediction",  
            "results_hgsprediction",
            "brain_correlation_results",
            f"{brain_data_type.upper()}_subcorticals_cerebellum",
            "brain_ready_data",            
            f"Schaefer{schaefer}",
        )
# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"brain_{brain_data_type}_Schaefer{schaefer}_data.csv")

brain_df = pd.read_csv(file_path, sep=',', index_col=0)

print("===== Done! =====")
embed(globals(), locals())
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

merged_df = pd.merge(brain_df, df, left_index=True, right_index=True, how='inner')

merged_df_female = merged_df[merged_df['gender']==0]
merged_df_male = merged_df[merged_df['gender']==1]


folder_path = os.path.join(
            "/data",
            "project",
            "stroke_ukb",
            "knazarzadeh",
            "project_hgsprediction",  
            "results_hgsprediction",
            "brain_correlation_results",
            f"{brain_data_type.upper()}_subcorticals_cerebellum",
            f"data_overlap_with_{mri_status}_{population}",            
            )
# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"both_gender_overlap_data_with_{mri_status}_{population}.csv")

merged_df.to_csv(file_path, sep=',', index=True)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"female_overlap_data_with_{mri_status}_{population}.csv")

merged_df_female.to_csv(file_path, sep=',', index=True)

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"male_overlap_data_with_{mri_status}_{population}.csv")

merged_df_male.to_csv(file_path, sep=',', index=True)

print("===== Done! =====")
embed(globals(), locals())
##############################################################################
n_regions = brain_df.shape[1]
x_axis = brain_df.columns    

# Correlation with True HGS
true_corr_female, true_corr_significant_female, true_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_true", x_axis)
true_corr_male, true_corr_significant_male, true_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_true", x_axis)
true_corr, true_corr_significant, true_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_true", x_axis)

# Correlation with predicted HGS
predicted_corr_female, predicted_corr_significant_female, predicted_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_predicted", x_axis)
predicted_corr_male, predicted_corr_significant_male, predicted_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_predicted", x_axis)
predicted_corr, predicted_corr_significant, predicted_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_predicted", x_axis)

# Correlation with Delta HGS
delta_corr_female, delta_corr_significant_female, delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_delta(true-predicted)", x_axis)
delta_corr_male, delta_corr_significant_male, delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_delta(true-predicted)", x_axis)
delta_corr, delta_corr_significant, delta_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_delta(true-predicted)", x_axis)

##############################################################################
# Plotting
def plot_bar_with_scatter(data, x, y, corr_target, gender, n_regions_survived, color):
    plt.figure(figsize=(40, 10))
    sns.barplot(data=data, x=x, y=y, color='darkgrey', errorbar=None, width=0.3)
    sns.scatterplot(data=data, x=x, y=y, color=color, zorder=5, s=100)
    plt.xlabel(x.capitalize(), fontsize=20, fontweight='bold')
    plt.ylabel(f'Correlations {corr_target} HGS vs brain regions', fontsize=20, fontweight='bold')
    plt.title(f"{gender.capitalize()} - {corr_target.capitalize()} {target.replace('hgs_', '')} HGS vs brain regions - survived regions({n_regions_survived}/150)", fontsize=20, fontweight='bold')
    plt.xticks(rotation=90, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(-0.5, len(data[x]) - 0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"corr_gmv_{corr_target}_{target}_{gender}.png")  # Save the plot as a PNG file

##############################################################################
##############################################################################
# Plotting True HGS vs GMV
sorted_p_values_true_female = true_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_true_male = true_corr_significant_male.sort_values(by='correlations', ascending=False)
sorted_p_values_true = true_corr_significant.sort_values(by='correlations', ascending=False)
# print("===== Done! =====")
# embed(globals(), locals())
# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_true_female, 'regions', 'correlations', "true", 'female', true_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_true_male, 'regions', 'correlations', "true", 'male', true_n_regions_survived_male, color="#069AF3")
# Both gender Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_true, 'regions', 'correlations', "true", 'both gender', true_n_regions_survived, color="orange")

##############################################################################
##############################################################################
# Plotting Predicted HGS vs GMV
sorted_p_values_predicted_female = predicted_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_predicted_male = predicted_corr_significant_male.sort_values(by='correlations', ascending=False)
sorted_p_values_predicted = predicted_corr_significant.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_predicted_female, 'regions', 'correlations', "predicted", 'female', predicted_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_predicted_male, 'regions', 'correlations', "predicted", 'male', predicted_n_regions_survived_male, color="#069AF3")
# Both gender Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_predicted, 'regions', 'correlations', "predicted", 'both gender', predicted_n_regions_survived, color="orange")

##############################################################################
##############################################################################
# Plotting Delta HGS vs GMV
sorted_p_values_delta_female = delta_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_delta_male = delta_corr_significant_male.sort_values(by='correlations', ascending=False)
sorted_p_values_delta = delta_corr_significant.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_female, 'regions', 'correlations', "delta(true-predicted)", 'female', delta_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_male, 'regions', 'correlations', "delta(true-predicted)", 'male', delta_n_regions_survived_male, color="#069AF3")
# Both gender Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta, 'regions', 'correlations', "delta(true-predicted)", 'both gender', delta_n_regions_survived, color="orange")

print("===== Done! =====")
embed(globals(), locals())
