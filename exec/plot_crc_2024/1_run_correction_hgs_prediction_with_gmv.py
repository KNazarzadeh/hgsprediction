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
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from hgsprediction.load_data import load_healthy_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.predict_hgs import calculate_brain_hgs                    
from hgsprediction.prediction_corrector_model import prediction_corrector_model
from hgsprediction.load_results import load_corrected_prediction_results
from hgsprediction.load_results import load_corrected_prediction_correlation_results
from hgsprediction.save_results.save_brain_correlation_results import save_brain_correlation_overlap_data_with_mri, save_brain_hgs_correlation_results, save_brain_hgs_correlation_results_for_plot


from nilearn import image, plotting, datasets
import nibabel as nib

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
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
brain_data_type = sys.argv[10]
schaefer = sys.argv[11]
stats_correlation_type = sys.argv[12]
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################

df_female = load_corrected_prediction_results(
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
)

df_male = load_corrected_prediction_results(
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
)

###############################################################################

df = pd.concat([df_female, df_male], axis=0)
# print("===== Done! =====")
# embed(globals(), locals())
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
            f"schaefer{schaefer}",            
            "brain_ready_data",
        )
# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"brain_{brain_data_type}_Schaefer{schaefer}_data.csv")

brain_df = pd.read_csv(file_path, sep=',', index_col=0)

brain_df.index = 'sub-' + brain_df.index.astype(str)

# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
tiv_path = os.path.join(
    "/data",
    "project",
    "stroke_ukb",
    "knazarzadeh",
    "project_hgsprediction",
    "brain_imaging_data",
    f"TIV",
)

df_tiv = pd.read_csv(f"{tiv_path}/cat_rois_Schaefer2018_600Parcels_17Networks_order.csv", sep=',', index_col=0)

tiv = df_tiv[df_tiv['Session']=='ses-2']['TIV']

merged_gmv_tiv = pd.merge(brain_df, tiv , left_index=True, right_index=True, how='inner')

brain_regions = brain_df.columns
# Initialize a DataFrame to store residuals
residuals_df = pd.DataFrame(index=merged_gmv_tiv.index, columns=brain_regions)
# Loop through each region
for region in brain_regions:
    # Extract TIV values
    X = merged_gmv_tiv.loc[:, 'TIV'].values.reshape(-1, 1)
    # Extract the region's values
    y = merged_gmv_tiv.loc[:, region].values.reshape(-1, 1)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict using the model
    y_pred = model.predict(X)
    
    # Calculate residuals
    residuals = y - y_pred
    # Store residuals in the DataFrame
    residuals_df.loc[:, region] = residuals

residuals_df.index = residuals_df.index.str.replace("sub-", "")
residuals_df.index = residuals_df.index.map(int)
##############################################################################
merged_df = pd.merge(residuals_df, df, left_index=True, right_index=True, how='inner')

merged_df_female = merged_df[merged_df['gender']==0]
merged_df_male = merged_df[merged_df['gender']==1]

save_brain_correlation_overlap_data_with_mri(
    merged_df,
    merged_df_female,
    merged_df_male,
    brain_data_type,
    schaefer,
)


##############################################################################
n_regions = brain_df.shape[1]
x_axis = brain_df.columns    

# Correlation with True HGS
true_corr_female, true_corr_significant_female, true_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}", x_axis, stats_correlation_type)
true_corr_male, true_corr_significant_male, true_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}", x_axis, stats_correlation_type)
true_corr, true_corr_significant, true_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}", x_axis, stats_correlation_type)

corr_target = "true_hgs"
save_brain_hgs_correlation_results(
    true_corr_female,
    true_corr_male,
    true_corr,
    brain_data_type,
    schaefer,
    corr_target,
)
# Correlation with predicted HGS
predicted_corr_female, predicted_corr_significant_female, predicted_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_predicted", x_axis, stats_correlation_type)
predicted_corr_male, predicted_corr_significant_male, predicted_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_predicted", x_axis, stats_correlation_type)
predicted_corr, predicted_corr_significant, predicted_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_predicted", x_axis, stats_correlation_type)

corr_target = "predicted_hgs"
save_brain_hgs_correlation_results(
    predicted_corr_female,
    predicted_corr_male,
    predicted_corr,
    brain_data_type,
    schaefer,
    corr_target,
)
# Correlation with corrected predicted HGS
corrected_predicted_corr_female, corrected_predicted_corr_significant_female, corrected_predicted_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_corrected_predicted", x_axis, stats_correlation_type)
corrected_predicted_corr_male, corrected_predicted_corr_significant_male, corrected_predicted_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_corrected_predicted", x_axis, stats_correlation_type)
corrected_predicted_corr, corrected_predicted_corr_significant, corrected_predicted_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_corrected_predicted", x_axis, stats_correlation_type)

corr_target = "corrected_predicted_hgs"
save_brain_hgs_correlation_results(
    corrected_predicted_corr_female,
    corrected_predicted_corr_male,
    corrected_predicted_corr,
    brain_data_type,
    schaefer,
    corr_target,
)
# Correlation with Delta HGS
delta_corr_female, delta_corr_significant_female,delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
delta_corr_male, delta_corr_significant_male, delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
delta_corr, delta_corr_significant, delta_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)

corr_target = "delta_hgs"
save_brain_hgs_correlation_results(
    delta_corr_female,
    delta_corr_male,
    delta_corr,
    brain_data_type,
    schaefer,
    corr_target,
)
# Correlation with corrected Delta HGS
corrected_delta_corr_female, corrected_delta_corr_significant_female, corrected_delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_corrected_delta(true-predicted)", x_axis, stats_correlation_type)
corrected_delta_corr_male, corrected_delta_corr_significant_male, corrected_delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_corrected_delta(true-predicted)", x_axis, stats_correlation_type)
corrected_delta_corr, corrected_delta_corr_significant, corrected_delta_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_corrected_delta(true-predicted)", x_axis, stats_correlation_type)

corr_target = "corrected_delta_hgs"
save_brain_hgs_correlation_results(
    corrected_delta_corr_female,
    corrected_delta_corr_male,
    corrected_delta_corr,
    brain_data_type,
    schaefer,
    corr_target,
)

##############################################################################
# Plotting
def plot_bar_with_scatter(data, x, y, corr_target, gender, n_regions_survived, color):
    if len(brain_df.columns) == 150:
        plt.figure(figsize=(50, 10))
    else:
        plt.figure(figsize=(250, 10))
    sns.barplot(data=data, x=x, y=y, color='darkgrey', errorbar=None, width=0.2)
    sns.scatterplot(data=data, x=x, y=y, color=color, zorder=5, s=100)
    plt.xlabel(x.capitalize(), fontsize=35, fontweight='bold')
    plt.ylabel(f'Correlations {corr_target} HGS vs brain regions', fontsize=20, fontweight='bold')
    plt.title(f"{gender.capitalize()} - {corr_target.capitalize()} {target.replace('hgs_', '')} HGS vs brain regions - survived regions({n_regions_survived}/{len(brain_df.columns)})-Schaefer{schaefer}", fontsize=20, fontweight='bold')
    plt.xticks(rotation=90, fontsize=20, fontweight='bold')
    plt.yticks(fontsize=25, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(-0.5, len(data[x]) - 0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"beheshti_corr_gmv_without_TIV_schaefer{schaefer}_{stats_correlation_type}_{model_name}_{corr_target}_{target}_{gender}.png")  # Save the plot as a PNG file
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
##############################################################################
# Plotting True HGS vs GMV
sorted_p_values_true_female = true_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_true_male = true_corr_significant_male.sort_values(by='correlations', ascending=False)
sorted_p_values_true = true_corr_significant.sort_values(by='correlations', ascending=False)

# print("===== Done! =====")
# embed(globals(), locals())
# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_true_female, 'regions', 'correlations', "true", 'female', true_n_regions_survived_female, color="#f45f74")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_true_male, 'regions', 'correlations', "true", 'male', true_n_regions_survived_male, color="#00b0be")
plot_bar_with_scatter(sorted_p_values_true, 'regions', 'correlations', "true", 'both_gender', true_n_regions_survived, color="orange")

##############################################################################
##############################################################################
# Plotting Predicted HGS vs GMV
sorted_p_values_corrected_predicted_female = corrected_predicted_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_corrected_predicted_male = corrected_predicted_corr_significant_male.sort_values(by='correlations', ascending=False)
sorted_p_values_corrected_predicted = corrected_predicted_corr_significant.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_corrected_predicted_female, 'regions', 'correlations', "corrected_predicted", 'female', corrected_predicted_n_regions_survived_female, color="#f45f74")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_corrected_predicted_male, 'regions', 'correlations', "corrected_predicted", 'male', corrected_predicted_n_regions_survived_male, color="#00b0be")
plot_bar_with_scatter(sorted_p_values_corrected_predicted, 'regions', 'correlations', "corrected_predicted", 'both_gender', corrected_predicted_n_regions_survived, color="orange")

##############################################################################
##############################################################################
sorted_p_values_predicted_female = predicted_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_predicted_male = predicted_corr_significant_male.sort_values(by='correlations', ascending=False)
sorted_p_values_predicted = predicted_corr_significant.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_predicted_female, 'regions', 'correlations', "predicted", 'female', predicted_n_regions_survived_female, color="#f45f74")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_predicted_male, 'regions', 'correlations', "predicted", 'male', predicted_n_regions_survived_male, color="#00b0be")

plot_bar_with_scatter(sorted_p_values_predicted, 'regions', 'correlations', "predicted", 'both_gender', predicted_n_regions_survived, color="orange")

##############################################################################
##############################################################################
# Plotting Delta HGS vs GMV
sorted_p_values_corrected_delta_female = corrected_delta_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_corrected_delta_male = corrected_delta_corr_significant_male.sort_values(by='correlations', ascending=False)
sorted_p_values_corrected_delta = corrected_delta_corr_significant.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_corrected_delta_female, 'regions', 'correlations', "corrected_delta", 'female', corrected_delta_n_regions_survived_female, color="#f45f74")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_corrected_delta_male, 'regions', 'correlations', "corrected_delta", 'male', corrected_delta_n_regions_survived_male, color="#00b0be")
plot_bar_with_scatter(sorted_p_values_corrected_delta, 'regions', 'correlations', "corrected_delta", 'both_gender', corrected_delta_n_regions_survived, color="orange")

##############################################################################
##############################################################################
sorted_p_values_delta_female = delta_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_delta_male = delta_corr_significant_male.sort_values(by='correlations', ascending=False)
sorted_p_values_delta = delta_corr_significant.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_female, 'regions', 'correlations', "delta", 'female', delta_n_regions_survived_female, color="#f45f74")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_male, 'regions', 'correlations', "delta", 'male', delta_n_regions_survived_male, color="#00b0be")
plot_bar_with_scatter(sorted_p_values_delta, 'regions', 'correlations', "delta", 'both_gender', delta_n_regions_survived, color="orange")

##############################################################################
##############################################################################
# Assuming predicted_corr_significant_female and true_corr_significant_female are pandas Series objects
# predicted_corr_series = predicted_corr_significant_female["p_values"]
# true_corr_series = true_corr_significant_female["p_values"]

# # Iterate over each element in the Series and compare their absolute values
# predicted_corr_stronger = 0
# true_corr_stronger = 0
# for pred_corr, true_corr in zip(predicted_corr_series, true_corr_series):
#     if abs(pred_corr) < abs(true_corr):
#         predicted_corr_stronger += 1
#     elif abs(pred_corr) > abs(true_corr):
#         true_corr_stronger += 1

# # Check which one is stronger based on the count of comparisons
# if predicted_corr_stronger > true_corr_stronger:
#     print("predicted_corr_significant_female is stronger")
# elif predicted_corr_stronger < true_corr_stronger:
#     print("true_corr_significant_female is stronger")
# else:
#     print("Both correlations have equal strength")
    
    
# predicted_corr_series = predicted_corr_significant_male["p_values"]
# true_corr_series = true_corr_significant_male["p_values"]

# ##############################################################################
# # Iterate over each element in the Series and compare their absolute values
# predicted_corr_stronger = 0
# true_corr_stronger = 0
# for pred_corr, true_corr in zip(predicted_corr_series, true_corr_series):
#     if abs(pred_corr) < abs(true_corr):
#         predicted_corr_stronger += 1
#     elif abs(pred_corr) > abs(true_corr):
#         true_corr_stronger += 1

# # Check which one is stronger based on the count of comparisons
# if predicted_corr_stronger > true_corr_stronger:
#     print("predicted_corr_significant_male is stronger")
# elif predicted_corr_stronger < true_corr_stronger:
#     print("true_corr_significant_male is stronger")
# else:
#     print("Both correlations have equal strength")

##############################################################################
##############################################################################

# %%
# Brain map with correlation values per confound


df_female_corr = true_corr_female.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_female_corr.columns))
df_female_corr.columns = column_numbers

df_male_corr = true_corr_male.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_male_corr.columns))
df_male_corr.columns = column_numbers

df_corr = true_corr.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_corr.columns))
df_corr.columns = column_numbers


save_brain_hgs_correlation_results_for_plot(
df_female_corr,
df_male_corr,
df_corr,
brain_data_type,
schaefer,
"true_hgs", 
)
    
df_female_corr = corrected_predicted_corr_female.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_female_corr.columns))
df_female_corr.columns = column_numbers

df_male_corr = corrected_predicted_corr_male.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_male_corr.columns))
df_male_corr.columns = column_numbers

df_corr = corrected_predicted_corr.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_corr.columns))
df_corr.columns = column_numbers

save_brain_hgs_correlation_results_for_plot(
df_female_corr,
df_male_corr,
df_corr,
brain_data_type,
schaefer,
"corrected_predicted_hgs", 
)

df_female_corr = predicted_corr_female.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_female_corr.columns))
df_female_corr.columns = column_numbers

df_male_corr = predicted_corr_male.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_male_corr.columns))
df_male_corr.columns = column_numbers


df_corr = predicted_corr.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_corr.columns))
df_corr.columns = column_numbers

save_brain_hgs_correlation_results_for_plot(
df_female_corr,
df_male_corr,
df_corr,
brain_data_type,
schaefer,
"predicted_hgs", 
)

df_female_corr = corrected_delta_corr_female.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_female_corr.columns))
df_female_corr.columns = column_numbers

df_male_corr = corrected_delta_corr_male.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_male_corr.columns))
df_male_corr.columns = column_numbers


df_corr = corrected_delta_corr.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_corr.columns))
df_corr.columns = column_numbers

save_brain_hgs_correlation_results_for_plot(
df_female_corr,
df_male_corr,
df_corr,
brain_data_type,
schaefer,
"corrected_delta_hgs", 
)
    
df_female_corr = delta_corr_female.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_female_corr.columns))
df_female_corr.columns = column_numbers

df_male_corr = delta_corr_male.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_male_corr.columns))
df_male_corr.columns = column_numbers


df_corr = delta_corr.drop(columns=["regions", "p_values", "pcorrected", "significant"])
column_numbers = range(len(df_corr.columns))
df_corr.columns = column_numbers

save_brain_hgs_correlation_results_for_plot(
df_female_corr,
df_male_corr,
df_corr,
brain_data_type,
schaefer,
"delta_hgs", 
)
print("===== Done! =====")
embed(globals(), locals())
