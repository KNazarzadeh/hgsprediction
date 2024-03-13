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
from hgsprediction.load_data import healthy_load_data
from hgsprediction.define_features import define_features
from hgsprediction.predict_hgs import predict_hgs
from hgsprediction.extract_data import healthy_extract_data
from hgsprediction.predict_hgs import calculate_brain_hgs                    
from hgsprediction.prediction_corrector_model import prediction_corrector_model
from hgsprediction.load_results import load_corrected_prediction_results
from hgsprediction.load_results import load_corrected_prediction_correlation_results
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

##############################################################################
n_regions = brain_df.shape[1]
x_axis = brain_df.columns    

# Correlation with True HGS
true_corr_female, true_corr_significant_female, true_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}", x_axis, stats_correlation_type)
true_corr_male, true_corr_significant_male, true_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}", x_axis, stats_correlation_type)

# Correlation with predicted HGS
predicted_corr_female, predicted_corr_significant_female, predicted_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_predicted", x_axis, stats_correlation_type)
predicted_corr_male, predicted_corr_significant_male, predicted_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_predicted", x_axis, stats_correlation_type)

# Correlation with corrected predicted HGS
corrected_predicted_corr_female, corrected_predicted_corr_significant_female, corrected_predicted_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_corrected_predicted", x_axis, stats_correlation_type)
corrected_predicted_corr_male, corrected_predicted_corr_significant_male, corrected_predicted_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_corrected_predicted", x_axis, stats_correlation_type)

# Correlation with Delta HGS
delta_corr_female, delta_corr_significant_female,delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)
delta_corr_male, delta_corr_significant_male, delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_delta(true-predicted)", x_axis, stats_correlation_type)

# Correlation with corrected Delta HGS
corrected_delta_corr_female, corrected_delta_corr_significant_female, corrected_delta_n_regions_survived_female = calculate_brain_hgs(merged_df_female, f"{target}_corrected_delta(true-predicted)", x_axis, stats_correlation_type)
corrected_delta_corr_male, corrected_delta_corr_significant_male, corrected_delta_n_regions_survived_male = calculate_brain_hgs(merged_df_male, f"{target}_corrected_delta(true-predicted)", x_axis, stats_correlation_type)


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
    plt.ylabel(f'Correlations {corr_target} HGS vs brain regions', fontsize=10, fontweight='bold')
    plt.title(f"{gender.capitalize()} - {corr_target.capitalize()} {target.replace('hgs_', '')} HGS vs brain regions - survived regions({n_regions_survived}/{len(brain_df.columns)})-Schaefer{schaefer}", fontsize=20, fontweight='bold')
    plt.xticks(rotation=90, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
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
# print("===== Done! =====")
# embed(globals(), locals())
# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_true_female, 'regions', 'correlations', "true", 'female', true_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_true_male, 'regions', 'correlations', "true", 'male', true_n_regions_survived_male, color="#069AF3")
##############################################################################
##############################################################################
# Plotting Predicted HGS vs GMV
sorted_p_values_predicted_female = corrected_predicted_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_predicted_male = corrected_predicted_corr_significant_male.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_predicted_female, 'regions', 'correlations', "corrected_predicted", 'female', corrected_predicted_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_predicted_male, 'regions', 'correlations', "corrected_predicted", 'male', corrected_predicted_n_regions_survived_male, color="#069AF3")

##############################################################################
##############################################################################
sorted_p_values_delta_female = predicted_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_delta_male = predicted_corr_significant_male.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_female, 'regions', 'correlations', "predicted", 'female', predicted_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_male, 'regions', 'correlations', "predicted", 'male', predicted_n_regions_survived_male, color="#069AF3")


##############################################################################
##############################################################################
# Plotting Delta HGS vs GMV
sorted_p_values_delta_female = corrected_delta_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_delta_male = corrected_delta_corr_significant_male.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_female, 'regions', 'correlations', "corrected_delta", 'female', corrected_delta_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_male, 'regions', 'correlations', "corrected_delta", 'male', corrected_delta_n_regions_survived_male, color="#069AF3")

##############################################################################
##############################################################################
sorted_p_values_delta_female = delta_corr_significant_female.sort_values(by='correlations', ascending=False)
sorted_p_values_delta_male = delta_corr_significant_male.sort_values(by='correlations', ascending=False)

# Females Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_female, 'regions', 'correlations', "delta", 'female', delta_n_regions_survived_female, color="red")
# Males Correlations GMV vs True HGS
plot_bar_with_scatter(sorted_p_values_delta_male, 'regions', 'correlations', "delta", 'male', delta_n_regions_survived_male, color="#069AF3")

##############################################################################
##############################################################################
# Assuming predicted_corr_significant_female and true_corr_significant_female are pandas Series objects
predicted_corr_series = predicted_corr_significant_female["correlations"]
true_corr_series = true_corr_significant_female["correlations"]

# Iterate over each element in the Series and compare their absolute values
predicted_corr_stronger = 0
true_corr_stronger = 0
for pred_corr, true_corr in zip(predicted_corr_series, true_corr_series):
    if abs(pred_corr) > abs(true_corr):
        predicted_corr_stronger += 1
    elif abs(pred_corr) < abs(true_corr):
        true_corr_stronger += 1

# Check which one is stronger based on the count of comparisons
if predicted_corr_stronger > true_corr_stronger:
    print("predicted_corr_significant_female is stronger")
elif predicted_corr_stronger < true_corr_stronger:
    print("true_corr_significant_female is stronger")
else:
    print("Both correlations have equal strength")
    
    
predicted_corr_series = predicted_corr_significant_male["correlations"]
true_corr_series = true_corr_significant_male["correlations"]

# Iterate over each element in the Series and compare their absolute values
predicted_corr_stronger = 0
true_corr_stronger = 0
for pred_corr, true_corr in zip(predicted_corr_series, true_corr_series):
    if abs(pred_corr) > abs(true_corr):
        predicted_corr_stronger += 1
    elif abs(pred_corr) < abs(true_corr):
        true_corr_stronger += 1

# Check which one is stronger based on the count of comparisons
if predicted_corr_stronger > true_corr_stronger:
    print("predicted_corr_significant_male is stronger")
elif predicted_corr_stronger < true_corr_stronger:
    print("true_corr_significant_male is stronger")
else:
    print("Both correlations have equal strength")

print("===== Done! =====")
embed(globals(), locals())
##############################################################################
##############################################################################

print("Females - Delta - First 30 survival regions:")
print(sorted_p_values_delta_female.iloc[0:30, :])
print("Males - Delta - First 30 survival regions:")
print(sorted_p_values_delta_male.iloc[0:30, :])

print("Females - Delta - Last 30 survival regions:")
print(sorted_p_values_delta_female.tail(30))
print("Males - Delta - Last 30 survival regions:")
print(sorted_p_values_delta_male.tail(30))
print("===== Done! =====")
embed(globals(), locals())

##############################################################################

fig = plt.figure(figsize=(8,8))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})
sns.regplot(x=true_corr_female["correlations"].astype(float), y=corrected_delta_corr_female["correlations"].astype(float), color='red',  marker="$\circ$", scatter_kws={'s': 50})
sns.regplot(x=true_corr_male["correlations"].astype(float), y=corrected_delta_corr_male["correlations"].astype(float), color='#069AF3',  marker="$\circ$", scatter_kws={'s': 50})

plt.xlabel("Correlation GMV and true HGS", fontsize=12, fontweight="bold")
plt.ylabel("Correlation GMV and corrected delta(true-predicted) HGS", fontsize=12, fontweight="bold")
female_corr = pearsonr(true_corr_female["correlations"].astype(float), corrected_delta_corr_female["correlations"].astype(float))[0]
male_corr = pearsonr(true_corr_male["correlations"].astype(float), corrected_delta_corr_male["correlations"].astype(float))[0]
r2_female = r2_score(true_corr_female["correlations"].astype(float), corrected_delta_corr_female["correlations"].astype(float))
r2_male = r2_score(true_corr_male["correlations"].astype(float), corrected_delta_corr_male["correlations"].astype(float))

r_text_female = f"r:{female_corr:.3f}\nR2:{r2_female:.3f}"
r_text_male = f"r:{male_corr:.3f}\nR2:{r2_male:.3f}"
plt.annotate(r_text_female, xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12, fontweight="bold", color='red')
plt.annotate(r_text_male, xy=(0.5, 0.8), xycoords='axes fraction', fontsize=12, fontweight="bold", color='#069AF3')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()

 # Plot regression line
plt.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

plt.show()
plt.savefig(f"corrected_delta_true_hgs_jointplot_circles_{population} {mri_status}: {target}_{model_name}_gmv_new.png")
plt.close()

###############################################################################
# Delta vs True HGS
# Raw delta HGS vs True HGS
# Corrected delta HGS vs True HGS
df_female_correlations, df_female_p_values, df_female_r2_values = load_corrected_prediction_correlation_results(population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,)

df_male_correlations, df_male_p_values, df_male_r2_values =  load_corrected_prediction_correlation_results(population,
    mri_status,
    model_name,
    feature_type,
    target,
    "female",
    session,
    confound_status,
    n_repeats,
    n_folds,)

fig, axes = plt.subplots(2, 1, figsize=(25, 25))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 40,
                     "xtick.labelsize": 40,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i in range(2):
    ax = axes[i]
    
    if i == 0:
        sns.regplot(data=df_female, x=f"{target}", y=f"{target}_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male, x=f"{target}", y=f"{target}_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Delta HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("")                        
        
        r_text_female = f"r:{df_female_correlations.loc[0, 'r_values_true_delta']:.3f}\nR2:{df_female_r2_values.loc[0, 'r2_values_true_delta']:.3f}"
        r_text_male = f"r:{df_male_correlations.loc[0, 'r_values_true_delta']:.3f}\nR2:{df_male_r2_values.loc[0, 'r2_values_true_delta']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    elif i == 1:
        sns.regplot(data=df_female, x=f"{target}", y=f"{target}_corrected_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male, x=f"{target}", y=f"{target}_corrected_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Corrected delta HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

        r_text_female = f"r:{df_female_correlations.loc[0, 'r_values_true_corrected_delta']:.3f}\nR2:{df_female_r2_values.loc[0, 'r2_values_true_corrected_delta']:.3f}"
        r_text_male = f"r:{df_male_correlations.loc[0, 'r_values_true_corrected_delta']:.3f}\nR2:{df_male_r2_values.loc[0, 'r2_values_true_corrected_delta']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=12)
    
fig.suptitle(f"MRI (N={len(pd.concat([df_female, df_male]))})", fontsize=40, fontweight="bold")
plt.tight_layout()
plt.show()
plt.savefig(f"true_delta_corrected_delta10_10_new.png")
plt.close()

###############################################################################
# Delta vs True HGS
# Raw delta HGS vs True HGS
# Corrected delta HGS vs True HGS
fig, axes = plt.subplots(2, 1, figsize=(25, 25))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 40,
                     "xtick.labelsize": 40,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i in range(2):
    ax = axes[i]
    
    if i == 0:
        sns.regplot(data=merged_df_female, x=f"{target}", y=f"{target}_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=merged_df_male, x=f"{target}", y=f"{target}_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Delta HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("")                        
        
        female_corr = pearsonr(merged_df_female[f"{target}"], merged_df_female[f"{target}_delta(true-predicted)"])[0]
        male_corr = pearsonr(merged_df_male[f"{target}"], merged_df_male[f"{target}_delta(true-predicted)"])[0]
        r2_female = r2_score(merged_df_female[f"{target}"], merged_df_female[f"{target}_delta(true-predicted)"])
        r2_male = r2_score(merged_df_male[f"{target}"], merged_df_male[f"{target}_delta(true-predicted)"])

        r_text_female = f"r:{female_corr:.3f}\nR2:{r2_female:.3f}"
        r_text_male = f"r:{male_corr:.3f}\nR2:{r2_male:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    elif i == 1:
        sns.regplot(data=merged_df_female, x=f"{target}", y=f"corrected_delta_hgs", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=merged_df_male, x=f"{target}", y=f"corrected_delta_hgs", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Corrected delta HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

        female_corr = pearsonr(merged_df_female[f"{target}"], merged_df_female[f"corrected_delta_hgs"])[0]
        male_corr = pearsonr(merged_df_male[f"{target}"], merged_df_male[f"corrected_delta_hgs"])[0]
        r2_female = r2_score(merged_df_female[f"{target}"], merged_df_female[f"corrected_delta_hgs"])
        r2_male = r2_score(merged_df_male[f"{target}"], merged_df_male[f"corrected_delta_hgs"])

        r_text_female = f"r:{female_corr:.3f}\nR2:{r2_female:.3f}"
        r_text_male = f"r:{male_corr:.3f}\nR2:{r2_male:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=12)

fig.suptitle(f"MRI (N={len(pd.concat([merged_df_female, merged_df_male]))}) overlap with GMV", fontsize=40, fontweight="bold")
plt.tight_layout()
plt.show()
plt.savefig(f"true_corrected_delta_10_10_new.png")
plt.close()

###############################################################################