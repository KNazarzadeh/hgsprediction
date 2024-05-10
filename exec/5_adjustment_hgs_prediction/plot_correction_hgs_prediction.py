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
if confound_status == '0':
        confound = "without_confound_removal"
else:
    confound = "with_confound_removal"

for gender in ["female", "male"]:
    main_folder_path = os.path.join(
                "/data",
                "project",
                "stroke_ukb",
                "knazarzadeh",
                "project_hgsprediction",  
                "results_hgsprediction",
                f"{population}",
                f"{mri_status}",
                f"{feature_type}",
                f"{target}",
                f"{confound}",
                f"{model_name}",
                f"{n_repeats}_repeats_{n_folds}_folds",
                f"{gender}",
            )

    subfolders = ["corrected_predictions", "corrected_correlations"]

    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder_path, subfolder)
    
        file_path = os.path.join(subfolder_path, f"{gender}_{subfolder}.csv")
        
        if subfolder == "corrected_predictions":
            df_corrected = pd.read_csv(file_path, sep=',', index_col=0)
            
        elif subfolder == "corrected_correlations":
            df_correlations = pd.read_csv(file_path, sep=',', index_col=0)
    if gender == "female":
        df_female_corrected = df_corrected.copy()
        df_female_correlations = df_correlations.copy()
    
    elif gender == "male":
        df_male_corrected = df_corrected.copy()
        df_male_correlations = df_correlations.copy()
# print("===== Done! =====")
# embed(globals(), locals())
###############################################################################
# Predicted vs True HGS
# Raw predicted HGS vs True HGS
# Corrected predicted HGS vs True HGS
fig, axes = plt.subplots(2, 5, figsize=(100, 25))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 40,
                     "xtick.labelsize": 40,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i in range(2):
    for j in range(5):
        fold = j
        ax = axes[i][j]
        
        if i == 0:
            sns.regplot(data=df_female_corrected[df_female_corrected['cv_fold']==fold], x=f"{target}", y=f"{target}_predicted", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male_corrected[df_male_corrected['cv_fold']==fold], x=f"{target}", y=f"{target}_predicted", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
            ax.set_ylabel("Raw predicted HGS", fontsize=40, fontweight="bold")
            ax.set_xlabel("")            
            ax.set_title(f"Fold:{fold}", fontsize=40, fontweight="bold")            
            
            r_text_female = f"r:{df_female_correlations.loc[fold, 'r_values_true_predicted']:.3f}\nR2:{df_female_correlations.loc[fold, 'r2_values_true_predicted']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[fold, 'r_values_true_predicted']:.3f}\nR2:{df_male_correlations.loc[fold, 'r2_values_true_predicted']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
        elif i == 1:
            sns.regplot(data=df_female_corrected[df_female_corrected['cv_fold']==fold], x=f"{target}", y=f"hgs_L+R_corrected_predicted", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male_corrected[df_male_corrected['cv_fold']==fold], x=f"{target}", y=f"hgs_L+R_corrected_predicted", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
            ax.set_ylabel("Corrected predicted HGS", fontsize=40, fontweight="bold")
            ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

            r_text_female = f"r:{df_female_correlations.loc[fold, 'r_values_true_corrected_predicted']:.3f}\nR2:{df_female_correlations.loc[fold, 'r2_values_true_corrected_predicted']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[fold, 'r_values_true_corrected_predicted']:.3f}\nR2:{df_male_correlations.loc[fold, 'r2_values_true_corrected_predicted']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
        
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=12)

plt.tight_layout()
plt.show()
plt.savefig(f"true_predicted.png")
plt.close()

###############################################################################
# Delta vs True HGS
# Raw delta HGS vs True HGS
# Corrected delta HGS vs True HGS
fig, axes = plt.subplots(2, 5, figsize=(100, 25))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 40,
                     "xtick.labelsize": 40,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i in range(2):
    for j in range(5):
        fold = j
        ax = axes[i][j]
        
        if i == 0:
            sns.regplot(data=df_female_corrected[df_female_corrected['cv_fold']==fold], x=f"{target}", y=f"{target}_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male_corrected[df_male_corrected['cv_fold']==fold], x=f"{target}", y=f"{target}_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
            ax.set_ylabel("Raw delta HGS", fontsize=40, fontweight="bold")
            ax.set_xlabel("")                        
            ax.set_title(f"Fold:{fold}", fontsize=40, fontweight="bold")            
            
            r_text_female = f"r:{df_female_correlations.loc[fold, 'r_values_true_delta']:.3f}\nR2:{df_female_correlations.loc[fold, 'r2_values_true_delta']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[fold, 'r_values_true_delta']:.3f}\nR2:{df_male_correlations.loc[fold, 'r2_values_true_delta']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
        elif i == 1:
            sns.regplot(data=df_female_corrected[df_female_corrected['cv_fold']==fold], x=f"{target}", y=f"hgs_L+R_corrected_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male_corrected[df_male_corrected['cv_fold']==fold], x=f"{target}", y=f"hgs_L+R_corrected_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
            ax.set_ylabel("Corrected delta HGS", fontsize=40, fontweight="bold")
            ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

            r_text_female = f"r:{df_female_correlations.loc[fold, 'r_values_true_corrected_delta']:.3f}\nR2:{df_female_correlations.loc[fold, 'r2_values_true_corrected_delta']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[fold, 'r_values_true_corrected_delta']:.3f}\nR2:{df_male_correlations.loc[fold, 'r2_values_true_corrected_delta']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
        
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=12)

plt.tight_layout()
plt.show()
plt.savefig(f"true_delta.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())