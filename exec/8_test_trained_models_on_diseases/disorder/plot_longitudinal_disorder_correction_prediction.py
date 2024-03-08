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


from hgsprediction.load_results.load_disorder_corrected_prediction_results import load_disorder_corrected_prediction_results
from hgsprediction.load_results.load_disorder_corrected_prediction_correlation_results import load_disorder_corrected_prediction_correlation_results

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
disorder_cohort = sys.argv[3]
visit_session = sys.argv[4]
feature_type = sys.argv[5]
target = sys.argv[6]
model_name = sys.argv[7]
confound_status = sys.argv[8]
n_repeats = sys.argv[9]
n_folds = sys.argv[10]
###############################################################################
# load data
disorder_cohort = f"{disorder_cohort}-{population}"
if visit_session == "1":
    session_column = f"1st_{disorder_cohort}_session"

df_female = load_disorder_corrected_prediction_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "female",
    confound_status,
    n_repeats,
    n_folds,
)

df_male = load_disorder_corrected_prediction_results(
    population,
    mri_status,
    session_column,
    model_name,
    feature_type,
    target,
    "male",
    confound_status,
    n_repeats,
    n_folds,
)

df_female_correlations, df_female_p_values, df_female_r2_values = load_disorder_corrected_prediction_correlation_results(   
                                                                                        population,
                                                                                        mri_status,
                                                                                        session_column,
                                                                                        model_name,
                                                                                        feature_type,
                                                                                        target,
                                                                                        "female",
                                                                                        confound_status,
                                                                                        n_repeats,
                                                                                        n_folds,    
                                                                                    )

df_male_correlations, df_male_p_values, df_male_r2_values = load_disorder_corrected_prediction_correlation_results(   
                                                                                        population,
                                                                                        mri_status,
                                                                                        session_column,
                                                                                        model_name,
                                                                                        feature_type,
                                                                                        target,
                                                                                        "male",
                                                                                        confound_status,
                                                                                        n_repeats,
                                                                                        n_folds,    
                                                                                    )

###############################################################################
# Predicted vs True HGS
# Raw predicted HGS vs True HGS
# Corrected predicted HGS vs True HGS
fig, axes = plt.subplots(2, 2, figsize=(30, 30))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 30,
                     "xtick.labelsize": 30,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i, disorder_subgroup in enumerate([f"pre-{population}", f"post-{population}"]):
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}"
    for j in range(2):
        ax = axes[j][i]

        if j == 0:
            sns.regplot(data=df_female, x=f"{prefix}_{target}", y=f"{prefix}_{target}_predicted", color='lightcoral', marker="$\circ$", scatter_kws={'s': 40, 'linewidths': 15}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male, x=f"{prefix}_{target}", y=f"{prefix}_{target}_predicted", color='#069AF3', marker="$\circ$", scatter_kws={'s': 40, 'linewidths': 15}, line_kws={"color": "blue"}, ax=ax)
            ax.set_ylabel("Predicted HGS", fontsize=30, fontweight="bold")
            ax.set_xlabel("")    
            ax.set_title(f"{disorder_subgroup}(females={len(df_female)}, males={len(df_male)})", fontsize=30, fontweight="bold")            

            r_text_female = f"r:{df_female_correlations.loc[0, f'{prefix}_r_values_true_predicted']:.3f}\nR2:{df_female_r2_values.loc[0, f'{prefix}_r2_values_true_predicted']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[0, f'{prefix}_r_values_true_predicted']:.3f}\nR2:{df_male_r2_values.loc[0, f'{prefix}_r2_values_true_predicted']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
        elif j == 1:
            sns.regplot(data=df_female, x=f"{prefix}_{target}", y=f"{prefix}_{target}_corrected_predicted", color='lightcoral', marker="$\circ$", scatter_kws={'s': 40, 'linewidths': 15}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male, x=f"{prefix}_{target}", y=f"{prefix}_{target}_corrected_predicted", color='#069AF3', marker="$\circ$", scatter_kws={'s': 40, 'linewidths': 15}, line_kws={"color": "blue"}, ax=ax)
            ax.set_ylabel("Corrected predicted HGS", fontsize=30, fontweight="bold")
            ax.set_xlabel("True HGS", fontsize=30, fontweight="bold")
            
            r_text_female = f"r:{df_female_correlations.loc[0, f'{prefix}_r_values_true_corrected_predicted']:.3f}\nR2:{df_female_r2_values.loc[0, f'{prefix}_r2_values_true_corrected_predicted']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[0, f'{prefix}_r_values_true_corrected_predicted']:.3f}\nR2:{df_male_r2_values.loc[0, f'{prefix}_r2_values_true_corrected_predicted']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
            
    
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=8)

plt.tight_layout()
plt.show()
plt.savefig(f"true_predicted_{population}_{feature_type}_{target}_{model_name}_{n_repeats}_{n_folds}_.png")
plt.close()
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# Delta vs True HGS
# Raw delta HGS vs True HGS
# Corrected delta HGS vs True HGS
fig, axes = plt.subplots(2, 2, figsize=(30, 30))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 30,
                     "xtick.labelsize": 30,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i, disorder_subgroup in enumerate([f"pre-{population}", f"post-{population}"]):
    if visit_session == "1":
        prefix = f"1st_{disorder_subgroup}"
    elif visit_session == "2":
        prefix = f"2nd_{disorder_subgroup}"
    elif visit_session == "3":
        prefix = f"3rd_{disorder_subgroup}"
    elif visit_session == "4":
        prefix = f"4th_{disorder_subgroup}"
    for j in range(2):
        ax = axes[j][i]

        if j == 0:
            sns.regplot(data=df_female, x=f"{prefix}_{target}", y=f"{prefix}_{target}_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 40, 'linewidths': 15}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male, x=f"{prefix}_{target}", y=f"{prefix}_{target}_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 40, 'linewidths': 15}, line_kws={"color": "blue"}, ax=ax)
            ax.set_ylabel("Predicted HGS", fontsize=30, fontweight="bold")
            ax.set_xlabel("")    
            ax.set_title(f"{disorder_subgroup}(females={len(df_female)}, males={len(df_male)})", fontsize=30, fontweight="bold")            

            r_text_female = f"r:{df_female_correlations.loc[0, f'{prefix}_r_values_true_delta(true-predicted)']:.3f}\nR2:{df_female_r2_values.loc[0, f'{prefix}_r2_values_true_delta(true-predicted)']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[0, f'{prefix}_r_values_true_delta(true-predicted)']:.3f}\nR2:{df_male_r2_values.loc[0, f'{prefix}_r2_values_true_delta(true-predicted)']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
        elif j == 1:
            sns.regplot(data=df_female, x=f"{prefix}_{target}", y=f"{prefix}_{target}_corrected_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 40, 'linewidths': 15}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male, x=f"{prefix}_{target}", y=f"{prefix}_{target}_corrected_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 40, 'linewidths': 15}, line_kws={"color": "blue"}, ax=ax)
            ax.set_ylabel("Corrected predicted HGS", fontsize=30, fontweight="bold")
            ax.set_xlabel("True HGS", fontsize=30, fontweight="bold")
            
            r_text_female = f"r:{df_female_correlations.loc[0, f'{prefix}_r_values_true_corrected_delta(true-predicted)']:.3f}\nR2:{df_female_r2_values.loc[0, f'{prefix}_r2_values_true_corrected_delta(true-predicted)']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[0, f'{prefix}_r_values_true_corrected_delta(true-predicted)']:.3f}\nR2:{df_male_r2_values.loc[0, f'{prefix}_r2_values_true_corrected_delta(true-predicted)']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
            
    
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=8)

plt.tight_layout()
plt.show()
plt.savefig(f"{population}_true_delta_{population}_{feature_type}_{target}_{model_name}_{n_repeats}_{n_folds}_.png")
plt.close()