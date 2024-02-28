import os
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score 
from hgsprediction.load_results.healthy import load_hgs_predicted_results
from hgsprediction.define_features import define_features
from scipy.stats import linregress, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
model_name = sys.argv[5]
y_axis = sys.argv[6]
session = sys.argv[7]
confound_status = sys.argv[8]
n_repeats = sys.argv[9]
n_folds = sys.argv[10]
stats_correlation_type = sys.argv[11]
###############################################################################
df_female = load_hgs_predicted_results(
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
df_male = load_hgs_predicted_results(
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

if y_axis == "delta":
    y_axis = "delta(true-predicted)"
    
features, extend_features = define_features(feature_type)

###############################################################################
###############################################################################
##############################################################################
def calculate_behavioral_hgs(df, y_axis, x_axis, stats_correlation_type):
    
    correlation_values = pd.DataFrame(columns=["feature_name", "correlations", "p_values"])

    for i, feature in enumerate(x_axis):
        # Compute correlations and p-values for all, female, and male datasets
        if stats_correlation_type == "pearson":
            corr, p_value = pearsonr(df.loc[:, feature], df.loc[:, f"{target}_{y_axis}"])
        elif stats_correlation_type == "spearman":
            corr, p_value = spearmanr(df.loc[:, feature], df.loc[:, f"{target}_{y_axis}"])
        correlation_values.loc[i, "feature_name"] = feature
        correlation_values.loc[i, "correlations"] = corr
        correlation_values.loc[i, "p_values"] = p_value

    # Perform FDR correction on p-values
    reject, p_corrected, _, _ = multipletests(correlation_values.loc[:, 'p_values'], method='fdr_bh')
    
    # Add corrected p-values and significance indicator columns to dataframes
    correlation_values.loc[:, 'p_corrected'] = p_corrected
    correlation_values.loc[:, 'significant'] = reject

    correlation_values_significant = correlation_values[correlation_values.loc[:, 'significant']==True]
    n_features_survived = len(correlation_values_significant)
    
    return correlation_values, correlation_values_significant, n_features_survived

corr_female, corr_significant_female, n_features_survived_female = calculate_behavioral_hgs(df_female, y_axis, features, stats_correlation_type)
corr_male, corr_significant_male, n_features_survived_male = calculate_behavioral_hgs(df_male, y_axis, features, stats_correlation_type)

print("===== Done! =====")
embed(globals(), locals())
# Plotting
def plot_bar_with_scatter(data, x, y, corr_target, gender, n_features_survived, color):
    plt.figure(figsize=(10, 40))
    sns.barplot(data=data, x=y, y=x, color='darkgrey', errorbar=None, width=0.3)
    sns.scatterplot(data=data, x=y, y=x, color=color, zorder=5, s=100)
    plt.xlabel(x.capitalize(), fontsize=20, fontweight='bold')
    plt.ylabel(f'{corr_target.capitalize()} HGS', fontsize=20, fontweight='bold')
    plt.title(f"{gender.capitalize()} - {corr_target.capitalize()} {target.replace('hgs_', '')} HGS vs brain features - survived features({n_features_survived}/150)", fontsize=20, fontweight='bold')
    plt.xticks(rotation=90, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=15, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xlim(-0.5, len(data[x]) - 0.5)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"corr_features_{stats_correlation_type}_{model_name}_{corr_target}_{target}_{gender}.png")  # Save the plot as a PNG file

sorted_p_values_female = corr_female.sort_values(by='correlations', ascending=True)
sorted_p_values_male = corr_male.sort_values(by='correlations', ascending=True)

plot_bar_with_scatter(sorted_p_values_female, 'feature_name', 'correlations', "delta", 'female', n_features_survived_female, color="red")
plot_bar_with_scatter(sorted_p_values_male, 'feature_name', 'correlations', "delta", 'male', n_features_survived_female, color="#069AF3")


def plot_custom_palette(df_sorted, target):
    custom_palette = ["#eb0917", "#86AD21", "#5ACACA", "#B382D6"]
    plt.figure(figsize=(20,30))
    plt.rcParams.update({"font.weight": "bold", 
                        "axes.labelweight": "bold",
                        "ytick.labelsize": 25,
                        "xtick.labelsize": 25})
    ax = sns.barplot(x='significance', y='feature_name', data=df_sorted, hue="cognitive_type", palette=custom_palette, width=0.5)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=25, color='black')
    plt.xlabel('-log(p-value)', weight="bold", fontsize=30)
    plt.ylabel('')
    plt.xticks(range(0, 25, 5))
    plt.title(f'non-MRI Controls (N={len(df)})', weight="bold", fontsize=30)
    plt.legend(fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"both_gender_cognitive_{target}.png")
    plt.close()


plot_custom_palette()
###############################################################################
###############################################################################
def plot_correlation(df, population, mri_status, target, model_name, x_axis, y_axis):
    # Convert numeric gender values to categorical labels
    df["gender"] = df["gender"].replace({0: "Female", 1: "Male"})
    
    custom_palette = {"Female": 'red', "Male": '#069AF3'}
    
    plt.rcParams.update({"font.weight": "bold", 
                         "axes.labelweight": "bold",
                         "ytick.labelsize": 12,
                         "xtick.labelsize": 12,
                         })
    
    sns.set_style("whitegrid", {'axes.grid' : False})
    
    fig = plt.figure(figsize=(12,12))
    
    g = sns.jointplot(data=df, x=f"{target}_{x_axis}", y=f"{target}_{y_axis}", hue="gender", palette=custom_palette, marker="$\circ$", s=50)

    for gender_type, gr in df.groupby(df['gender']):
        slope, intercept, r_value, p_value, std_err = linregress(gr[f"{target}_{y_axis}"], gr[f"{target}_{x_axis}"])
        if gr['gender'].any() == "Female":
            female_corr = pearsonr(gr[f"{target}_{y_axis}"], gr[f"{target}_{x_axis}"])[0]
            female_R2 = r2_score(gr[f"{target}_{y_axis}"], gr[f"{target}_{x_axis}"])
            print(female_corr)
            print("female_r2=", female_R2)
            g.ax_joint.text(0.05, 0.95, f'r = {female_corr:.2f}', ha='left', va='top', fontsize=10)
            
        elif gr['gender'].any() == "Male":
            male_corr = pearsonr(gr[f"{target}_{y_axis}"], gr[f"{target}_{x_axis}"])[0]
            male_R2 = r2_score(gr[f"{target}_{y_axis}"], gr[f"{target}_{x_axis}"])
            print(male_corr)
            print("male_r2=", male_R2)
            g.ax_joint.text(0.05, 0.95, f'r = {male_corr:.2f}', ha='left', va='top', fontsize=10)

        color = custom_palette[gender_type]
        sns.regplot(x=f"{target}_{x_axis}", y=f"{target}_{y_axis}", data=gr, scatter=False, ax=g.ax_joint, color=color, line_kws={'label': f'{gender_type} Regression (r={r_value:.2f})'})
        print(r_value)
    
    # Remove the legend from ax_joint
    g.ax_joint.legend_.remove()
    
    g.fig.suptitle(f"{population} {mri_status}: {target}", fontsize=10, fontweight="bold")
    g.fig.subplots_adjust(top=0.95) # Reduce plot to make room 
    
    g.ax_joint.set_xlabel(f"{x_axis.capitalize()} HGS", fontsize=12, fontweight="bold")
    g.ax_joint.set_ylabel(f"{y_axis.capitalize()} HGS", fontsize=12, fontweight="bold")
    
    xmin, xmax = g.ax_joint.get_xlim()
    ymin, ymax = g.ax_joint.get_ylim()
    g.ax_joint.set_xticks(np.arange(0, round(xmax), 30))
    
    # Plot regression line
    g.ax_joint.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')
    
    plt.show()
    plt.savefig(f"{y_axis}_{x_axis}_{population}_{mri_status}_session{session}_{model_name}_{feature_type}_{target}.png")
    plt.close()

# Call the function
plot_correlation(df, population, mri_status, target, model_name, x_axis, y_axis)

print("===== Done! =====")
embed(globals(), locals())

###############################################################################
###############################################################################


