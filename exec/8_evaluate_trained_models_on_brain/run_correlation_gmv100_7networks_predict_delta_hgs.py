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
from hgsprediction.LinearSVRHeuristicC_zscore import LinearSVRHeuristicC_zscore as svrhc
####### Julearn #######
from julearn import run_cross_validation
####### sklearn libraries #######
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score


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

# Convert index labels to strings and then append 'sub-'
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
# print("===== Done! =====")
# embed(globals(), locals())
##############################################################################

X = brain_regions.to_list()
y = "hgs_L+R_delta(true-predicted)"

# if model_name == "linear_svm":
#     model = svrhc(dual=False, loss='squared_epsilon_insensitive')
# elif model_name == "random_forest":
model = "rf"
###############################################################################
# When cv=None, it define as follows:
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=47)
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# run run_cross_validation
if confound_status == '0':
    scores_trained_female, model_trained_female = run_cross_validation(
        X=X, y=y, data=merged_df_female, cv=cv, seed=47,
        preprocess_X='zscore', problem_type='regression',
        model=model,
        return_estimator='all', scoring='r2'
    )
    scores_trained_male, model_trained_male = run_cross_validation(
        X=X, y=y, data=merged_df_male, cv=cv, seed=47,
        preprocess_X='zscore', problem_type='regression',
        model=model,
        return_estimator='all', scoring='r2'
    )

###############################################################################
df_test_score_female = scores_trained_female.set_index(
    ['repeat', 'fold'])['test_score'].unstack()
df_test_score_female.index.name = 'Repeats'
df_test_score_female.columns.name = 'K-fold splits'

print(df_test_score_female)

df_test_score_male = scores_trained_male.set_index(
    ['repeat', 'fold'])['test_score'].unstack()
df_test_score_male.index.name = 'Repeats'
df_test_score_male.columns.name = 'K-fold splits'

print(df_test_score_male)
###############################################################################














##############################################################################
n_regions = residuals_df.shape[1]
x_axis = residuals_df.columns    

delta_corr_female, delta_corr_significant_female, delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
delta_corr_male, delta_corr_significant_male, delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)


positive_delta_corr_female, positive_delta_corr_significant_female, positive_delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female[merged_df_female['hgs_L+R_delta(true-predicted)']>0], f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
positive_delta_corr_male, positive_delta_corr_significant_male, positive_delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male[merged_df_male['hgs_L+R_delta(true-predicted)']>0], f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)


negative_delta_corr_female, negative_delta_corr_significant_female, negative_delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female[merged_df_female['hgs_L+R_delta(true-predicted)']<0], f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
negative_delta_corr_male, negative_delta_corr_significant_male, negative_delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male[merged_df_male['hgs_L+R_delta(true-predicted)']<0], f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)

##############################################################################
##############################################################################
def plot_bar_with_scatter(data, x, y, corr_target, gender, n_regions_survived, color):
    if len(residuals_df.columns) == 7:
        plt.figure(figsize=(12, 10))
    else:
        plt.figure(figsize=(250, 10))
    sns.barplot(data=data, x=x, y=y, color='darkgrey', errorbar=None, width=0.3)
    sns.scatterplot(data=data, x=x, y=y, color=color, zorder=5, s=100)
    plt.xlabel(x.capitalize(), fontsize=20, fontweight='bold')
    plt.ylabel(f'Correlations {corr_target} HGS vs brain regions', fontsize=20, fontweight='bold')
    plt.title(f"{gender.capitalize()} - {corr_target.capitalize()} {target.replace('hgs_', '')} HGS vs brain regions - survived regions({n_regions_survived}/{len(residuals_df.columns)})-Schaefer{schaefer}", fontsize=20, fontweight='bold')
    plt.xticks(rotation=90, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(-0.5, len(data[x]) - 0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"corr_gmv_average_networks_without_TIV_schaefer{schaefer}_{stats_correlation_type}_{model_name}_{corr_target}_{target}_{gender}.png")  # Save the plot as a PNG file

##############################################################################
##############################################################################    
    # Plotting Delta HGS vs GMV
sorted_p_values_delta_female = delta_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_delta_male = delta_corr_significant_male.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
if not sorted_p_values_delta_female.empty:
    plot_bar_with_scatter(sorted_p_values_delta_female, 'regions', 'correlations', "delta(true-predicted)", 'female', delta_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_male, 'regions', 'correlations', "delta(true-predicted)", 'male', delta_n_regions_survived_male, color="#069AF3")
# Both gender Correlations GMV vs True HGS
print("===== Done! =====")
embed(globals(), locals())