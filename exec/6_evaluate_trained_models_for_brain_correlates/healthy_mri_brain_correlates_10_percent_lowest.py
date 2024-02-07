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
from scipy.stats import linregress

# from hgsprediction.plots import create_regplot

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
    f"{brain_correlation_type.upper()}_Schaefer400x7_Mean.jay")
feature_dt_schaefer = dt.fread(schaefer_file)
feature_df_schaefer = feature_dt_schaefer.to_pandas()
feature_df_schaefer.set_index('SubjectID', inplace=True)

tian_file = os.path.join(
    jay_path,
    f"{brain_correlation_type.upper()}_Tian_Mean.jay")
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
# Find the intersection of indexes
intersection_index = df.index.intersection(df_brain_correlation.index)

df_intersected = df[df.index.isin(intersection_index)]
df_brain_correlation_overlap = df_brain_correlation[df_brain_correlation.index.isin(intersection_index)]

intersection_index_female = df_intersected[df_intersected["gender"]==0].index
intersection_index_male = df_intersected[df_intersected["gender"]==1].index

df_intersected_female = df_intersected[df_intersected.index.isin(intersection_index_female)]
df_intersected_male = df_intersected[df_intersected.index.isin(intersection_index_male)]

# Calculate the number of rows to take for the top 10%
n_f = int(len(df_intersected_female) * 0.1)
n_m = int(len(df_intersected_male) * 0.1)

# Sort the dataframe by the column in descending order
sorted_df_female = df_intersected_female.sort_values(by=f'1st_scan_{target}_(actual-predicted)', ascending=True)
sorted_df_male = df_intersected_male.sort_values(by=f'1st_scan_{target}_(actual-predicted)', ascending=True)


# Take the top 10% highest values
top_10_percent_female = sorted_df_female.head(n_f)
top_10_percent_male = sorted_df_male.head(n_m)

df_brain_correlation_overlap_female = df_brain_correlation_overlap[df_brain_correlation_overlap.index.isin(top_10_percent_female.index)]
df_brain_correlation_overlap_male = df_brain_correlation_overlap[df_brain_correlation_overlap.index.isin(top_10_percent_male.index)]

##############################################################################
n_regions = df_brain_correlation_overlap.shape[1]
y_axis = ["actual", "predicted", "actual-predicted"]
x_axis = df_brain_correlation_overlap.columns.tolist()[:n_regions]

df_corr, df_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap, df_intersected, y_axis, x_axis)
df_female_corr, df_female_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_female, top_10_percent_female, y_axis, x_axis)
df_male_corr, df_male_pvalue = calculate_spearman_hgs_correlation_on_brain_correlations(df_brain_correlation_overlap_male, top_10_percent_male, y_axis, x_axis)
print(df_corr)
print(df_female_corr)
print(df_male_corr)
print("===== Done! =====")
embed(globals(), locals())
##############################################################################
# Perform FDR correction on p-values
p_values_flat_female = df_female_pvalue.values.flatten()
reject_female, corrected_p_values_female, _, _ = sm.multipletests(p_values_flat_female, alpha=0.05, method='fdr_bh')
corrected_p_values_matrix_female = corrected_p_values_female.reshape(df_female_pvalue.shape)

p_values_flat_male = df_male_pvalue.values.flatten()
reject_male, corrected_p_values_male, _, _ = sm.multipletests(p_values_flat_male, alpha=0.05, method='fdr_bh')
corrected_p_values_matrix_male = corrected_p_values_male.reshape(df_male_pvalue.shape)

# Check if any corrected p-value is less than the significance level (e.g., 0.05)
is_significant_female = any(corrected_p_values_matrix_female.flatten() < 0.05)
is_significant_male = any(corrected_p_values_matrix_male.flatten() < 0.05)

if is_significant_female or is_significant_male:
    print("At least one correlation is significant after FDR correction.")
else:
    print("None of the correlations are significant after FDR correction.")

##############################################################################
df_female_corr =df_female_corr.apply(pd.to_numeric, errors='coerce')
df_male_corr =df_male_corr.apply(pd.to_numeric, errors='coerce')
# Add gender column to each DataFrame
df_female_corr = df_female_corr.T
df_female_corr['gender'] = 0

df_male_corr = df_male_corr.T
df_male_corr['gender'] = 1

# Combine DataFrames
df_combined = pd.concat([df_female_corr, df_male_corr])

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

 # Plot regression line
g.ax_joint.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')
plt.tight_layout()

plt.show()
plt.savefig(f"correlate_mri_delta_true_{target}_lowest.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())
##############################################################################
fig = plt.figure(figsize=(12,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 12,
                     "xtick.labelsize": 12,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

g = sns.jointplot(data=df_combined, x=f'1st_scan_{target}_actual', y=f'1st_scan_{target}_predicted', hue="gender", palette=custom_palette,  marker="$\circ$", s=120)

for gender_type, gr in df_combined.groupby('gender'):
    # slope, intercept, r_value, p_value, std_err = linregress(gr[f'1st_scan_{target}_actual'], gr[f'1st_scan_{target}_predicted'])
    # if gr['gender'].any() == 0:
    #     female_corr = pearsonr(gr[f'1st_scan_{target}_predicted'], gr[f'1st_scan_{target}_actual'])[0]
    #     female_R2 = r2_score(gr[f'1st_scan_{target}_actual'], gr[f'1st_scan_{target}_predicted'])
    #     print(female_corr)
    #     print("female_r2=", female_R2)
    # elif gr['gender'].any() == 1:
    #     male_corr = pearsonr(gr[f'1st_scan_{target}_predicted'], gr[f'1st_scan_{target}_actual'])[0]
    #     male_R2 = r2_score(gr[f'1st_scan_{target}_actual'], gr[f'1st_scan_{target}_predicted'])
    #     print(male_corr)
    #     print("male_r2=", male_R2)
        
    sns.regplot(x=gr[f'1st_scan_{target}_actual'], y=gr[f'1st_scan_{target}_predicted'], ax=g.ax_joint, scatter=False, color=custom_palette[gender_type])
   
# remove the legend from ax_joint
g.ax_joint.legend_.remove()

# g.fig.suptitle(f"{population} {mri_status}: {target}", fontsize=10, fontweight="bold")
# g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 

g.ax_joint.set_xlabel("Correlation of True HGS and GMV", fontsize=12, fontweight="bold")
g.ax_joint.set_ylabel("Correlation of Predicted HGS and GMV correlation", fontsize=12, fontweight="bold")

xmin, xmax = g.ax_joint.get_xlim()
ymin, ymax = g.ax_joint.get_ylim()
# g.ax_joint.set_xticks(np.arange(0, round(xmax), 30))

 # Plot regression line
g.ax_joint.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')
plt.tight_layout()

plt.show()
plt.savefig(f"correlate_mri_predicted_true_{target}.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())