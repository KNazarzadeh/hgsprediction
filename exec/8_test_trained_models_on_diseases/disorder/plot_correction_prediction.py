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


from hgsprediction.prediction_corrector_model import prediction_corrector_model
from hgsprediction.load_results.healthy.load_hgs_predicted_results import load_hgs_predicted_results
from hgsprediction.load_results.load_corrected_prediction_results import load_corrected_prediction_results
from hgsprediction.load_results.load_corrected_prediction_correlation_results import load_corrected_prediction_correlation_results

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

df_female_correlations, df_female_p_values, df_female_r2_values = load_corrected_prediction_correlation_results(   
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

df_male_correlations, df_male_p_values, df_male_r2_values = load_corrected_prediction_correlation_results(   
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


print("===== Done! =====")
embed(globals(), locals())

###############################################################################
# Predicted vs True HGS
# Raw predicted HGS vs True HGS
# Corrected predicted HGS vs True HGS
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
        sns.regplot(data=df_female, x=f"{target}", y=f"{target}_predicted", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male, x=f"{target}", y=f"{target}_predicted", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Predicted HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("")            
        # ax.set_title(f"Fold:{fold}", fontsize=40, fontweight="bold")            
        
        r_text_female = f"r:{df_female_correlations.loc[0, 'r_values_true_predicted']:.3f}\nR2:{df_female_r2_values.loc[0, 'r2_values_true_predicted']:.3f}"
        r_text_male = f"r:{df_male_correlations.loc[0, 'r_values_true_predicted']:.3f}\nR2:{df_male_r2_values.loc[0, 'r2_values_true_predicted']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    elif i == 1:
        sns.regplot(data=df_female, x=f"{target}", y=f"{target}_corrected_predicted", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male, x=f"{target}", y=f"{target}_corrected_predicted", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Corrected predicted HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

        r_text_female = f"r:{df_female_correlations.loc[0, 'r_values_true_corrected_predicted']:.3f}\nR2:{df_female_r2_values.loc[0, 'r2_values_true_corrected_predicted']:.3f}"
        r_text_male = f"r:{df_male_correlations.loc[0, 'r_values_true_corrected_predicted']:.3f}\nR2:{df_male_r2_values.loc[0, 'r2_values_true_corrected_predicted']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=12)

plt.tight_layout()
plt.show()
plt.savefig(f"true_predicted_10_10.png")
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
        sns.regplot(data=df_female, x=f"{target}", y=f"{target}_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male, x=f"{target}", y=f"{target}_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Delta(true-predicted) HGS", fontsize=40, fontweight="bold")
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

plt.tight_layout()
plt.show()
plt.savefig(f"true_delta_10_10.png")
plt.close()