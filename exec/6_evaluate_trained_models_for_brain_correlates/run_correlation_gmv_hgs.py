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
from sklearn.linear_model import LinearRegression

from hgsprediction.save_results.brain_save_correlates_results import save_brain_hgs_correlation_results, save_brain_overlap_data_results
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
# print("===== Done! =====")
# embed(globals(), locals())
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

merged_df = pd.merge(residuals_df, df, left_index=True, right_index=True, how='inner')

merged_df_female = merged_df[merged_df['gender']==0]
merged_df_male = merged_df[merged_df['gender']==1]


print("===== Done! =====")
embed(globals(), locals())
save_brain_overlap_data_results(
    merged_df,
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
    brain_data_type,
    schaefer,
)
save_brain_overlap_data_results(
    merged_df_female,
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
)
save_brain_overlap_data_results(
    merged_df_male,
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
)
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
n_regions = brain_df.shape[1]
x_axis = brain_df.columns    

# Correlation with True HGS
true_corr_female, true_corr_significant_female, true_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_true", x_axis, stats_correlation_type)
true_corr_male, true_corr_significant_male, true_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_true", x_axis, stats_correlation_type)
true_corr, true_corr_significant, true_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_true", x_axis, stats_correlation_type)
save_brain_hgs_correlation_results(
    true_corr_female,
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
save_brain_hgs_correlation_results(
    true_corr_male,
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
# Correlation with predicted HGS
predicted_corr_female, predicted_corr_significant_female, predicted_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_predicted", x_axis, stats_correlation_type)
predicted_corr_male, predicted_corr_significant_male, predicted_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_predicted", x_axis, stats_correlation_type)
predicted_corr, predicted_corr_significant, predicted_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_predicted", x_axis, stats_correlation_type)
save_brain_hgs_correlation_results(
    predicted_corr_female,
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
save_brain_hgs_correlation_results(
    predicted_corr_male,
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
# Correlation with Delta HGS
delta_corr_female, delta_corr_significant_female, delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
delta_corr_male, delta_corr_significant_male, delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
delta_corr, delta_corr_significant, delta_n_regions_survived = calculate_brain_hgs(merged_df, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
save_brain_hgs_correlation_results(
    delta_corr_female,
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
save_brain_hgs_correlation_results(
    delta_corr_male,
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
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################
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
if not sorted_p_values_delta_female.empty:
    plot_bar_with_scatter(sorted_p_values_delta_female, 'regions', 'correlations', "delta(true-predicted)", 'female', delta_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_male, 'regions', 'correlations', "delta(true-predicted)", 'male', delta_n_regions_survived_male, color="#069AF3")
# Both gender Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta, 'regions', 'correlations', "delta(true-predicted)", 'both gender', delta_n_regions_survived, color="orange")



print("Females - Delta - survival regions:")
print(sorted_p_values_delta_female.iloc[0:30, :])
print("Males - Delta - survival regions:")
print(sorted_p_values_delta_male.iloc[0:30, :])
print("===== Done! =====")
embed(globals(), locals())

##############################################################################
##############################################################################

custom_palette = {1: '#069AF3', 0: 'red'}

fig = plt.figure(figsize=(12,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

g = sns.jointplot(data=df_combined, x=f'1st_scan_{target}_actual', y=f'1st_scan_{target}_(actual-predicted)', hue="gender", palette=custom_palette,  marker="$\circ$", s=120)
# Lists to store correlation coefficients and p-values
# correlation_coefficients = []
# p_values = []

for gender_type, gr in df_combined.groupby('gender'):
    
    slope, intercept, r_value, p_value, std_err = linregress(gr[f'1st_scan_{target}_actual'], gr[f'1st_scan_{target}_(actual-predicted)'])
    # correlation_coefficients.append(r_value)
    # p_values.append(p_value)
    # Perform FDR correction
    reject, corrected_p_values, _, _ = sm.multipletests(p_value, alpha=0.05, method='fdr_bh')  
# Plot regression lines and label significant correlations
# for i, (gender_type, gr) in enumerate(df_combined.groupby('gender')):
    # Plot regression line
    sns.regplot(x=gr[f'1st_scan_{target}_actual'], y=gr[f'1st_scan_{target}_(actual-predicted)'], ax=g.ax_joint, scatter=False, color=custom_palette[gender_type])
    # Label significant correlations
    if reject:
        g.ax_joint.text(gr[f'1st_scan_{target}_actual'].mean(), gr[f'1st_scan_{target}_(actual-predicted)'].mean(), 'Significant', ha='center', va='center', color='black', fontsize=10)
    slope, intercept, r_value, p_value, std_err = linregress(gr[f'1st_scan_{target}_actual'], gr[f'1st_scan_{target}_(actual-predicted)'])
    if gr['gender'].any() == 0:
        female_corr = pearsonr(gr[f'1st_scan_{target}_(actual-predicted)'], gr[f'1st_scan_{target}_actual'])[0]
        female_R2 = r2_score(gr[f'1st_scan_{target}_actual'], gr[f'1st_scan_{target}_(actual-predicted)'])
        print(female_corr)
        print("female_r2=", female_R2)
    elif gr['gender'].any() == 1:
        male_corr = pearsonr(gr[f'1st_scan_{target}_(actual-predicted)'], gr[f'1st_scan_{target}_actual'])[0]
        male_R2 = r2_score(gr[f'1st_scan_{target}_actual'], gr[f'1st_scan_{target}_(actual-predicted)'])
        print(male_corr)
        print("male_r2=", male_R2)
        
    sns.regplot(x=gr[f'1st_scan_{target}_actual'], y=gr[f'1st_scan_{target}_(actual-predicted)'], ax=g.ax_joint, scatter=False, color=custom_palette[gender_type])
   
# remove the legend from ax_joint
g.ax_joint.legend_.remove()

g.fig.suptitle(f"{population} {mri_status}: {target}", fontsize=10, fontweight="bold")
g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

g.ax_joint.set_xlabel("Correlation of True HGS and GMV", fontsize=12, fontweight="bold")
g.ax_joint.set_ylabel("Correlation of Delta HGS and GMV correlation", fontsize=12, fontweight="bold")

xmin, xmax = g.ax_joint.get_xlim()
ymin, ymax = g.ax_joint.get_ylim()
# g.ax_joint.set_xticks(np.arange(0, round(xmax), 30))

 # Plot regression line
g.ax_joint.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')
plt.tight_layout()

plt.show()
plt.savefig(f"correlate_mri_delta_true_{target}_FDR.png")
plt.close()