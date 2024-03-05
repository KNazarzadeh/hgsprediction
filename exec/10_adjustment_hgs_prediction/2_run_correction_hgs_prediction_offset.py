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
folder_path = os.path.join(
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
            f"female",
            "prediction_hgs_on_validation_set_trained",
        )

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"prediction_hgs_on_validation_set_trained_trained.pkl")

df_female = pd.read_pickle(file_path)

folder_path = os.path.join(
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
            f"male",
            "prediction_hgs_on_validation_set_trained",
        )

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    f"prediction_hgs_on_validation_set_trained_trained.pkl")

df_male = pd.read_pickle(file_path)

# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
def calculate_correlations(df, n_folds, target):
    df_corrected = pd.DataFrame()
    df_correlations = pd.DataFrame(columns=["cv_fold", 
                                                    "r_values_true_raw_predicted", "r2_values_true_raw_predicted",
                                                    "r_values_true_raw_delta", "r2_values_true_raw_delta",
                                                    "r_values_true_corrected_predicted", "r2_values_true_corrected_predicted",
                                                    "r_values_true_corrected_delta", "r2_values_true_corrected_delta"])

    model = LinearRegression()
    for fold in range(int(n_folds)):
        df_tmp = df[df['cv_fold']==fold]
        df_half = df_tmp.sample(frac=0.5, random_state=42) 
        model.fit(df_half.loc[:, f"{target}"].values.reshape(-1, 1), df_half.loc[:, f"{target}_delta(true-predicted)"])
        slope = model.coef_[0]
        intercept = model.intercept_
        df_half_rest = df_tmp[~df_tmp.index.isin(df_half.index)]
        
        # df_half_rest.loc[:, "corrected_predicted_hgs"] = (df_half_rest.loc[:, f"{target}_predicted"] - intercept) / slope
        df_half_rest.loc[:, "corrected_predicted_hgs"] = (df_half_rest.loc[:, f"{target}"] * slope) + intercept
        df_half_rest.loc[:, "corrected_delta_hgs"] =  df_half_rest.loc[:, f"{target}"] - df_half_rest.loc[:, "corrected_predicted_hgs"]

        r_values_true_raw_predicted = pearsonr(df_half_rest.loc[:, f"{target}"],df_half_rest.loc[:,f"{target}_predicted"])[0]
        r2_values_true_raw_predicted = r2_score(df_half_rest.loc[:, f"{target}"],df_half_rest.loc[:,f"{target}_predicted"])

        r_values_true_raw_delta = pearsonr(df_half_rest.loc[:, f"{target}"],df_half_rest.loc[:,f"{target}_delta(true-predicted)"])[0]
        r2_values_true_raw_delta = r2_score(df_half_rest.loc[:, f"{target}"],df_half_rest.loc[:,f"{target}_delta(true-predicted)"])

        r_values_true_corrected_predicted = pearsonr(df_half_rest.loc[:, f"{target}"],df_half_rest.loc[:,"corrected_predicted_hgs"])[0]
        r2_values_true_corrected_predicted = r2_score(df_half_rest.loc[:, f"{target}"],df_half_rest.loc[:,"corrected_predicted_hgs"])

        r_values_true_corrected_delta = pearsonr(df_half_rest.loc[:, f"{target}"],df_half_rest.loc[:,"corrected_delta_hgs"])[0]
        r2_values_true_corrected_delta = r2_score(df_half_rest.loc[:, f"{target}"],df_half_rest.loc[:,"corrected_delta_hgs"])

        df_correlations.loc[fold, "cv_fold"] = fold
        df_correlations.loc[fold, "r_values_true_raw_predicted"] = r_values_true_raw_predicted
        df_correlations.loc[fold, "r2_values_true_raw_predicted"] = r2_values_true_raw_predicted
        df_correlations.loc[fold, "r_values_true_raw_delta"] = r_values_true_raw_delta
        df_correlations.loc[fold, "r2_values_true_raw_delta"] = r2_values_true_raw_delta
        df_correlations.loc[fold, "r_values_true_corrected_predicted"] = r_values_true_corrected_predicted
        df_correlations.loc[fold, "r2_values_true_corrected_predicted"] = r2_values_true_corrected_predicted
        df_correlations.loc[fold, "r_values_true_corrected_delta"] = r_values_true_corrected_delta
        df_correlations.loc[fold, "r2_values_true_corrected_delta"] = r2_values_true_corrected_delta

        df_corrected = pd.concat([df_corrected, df_half_rest], axis=0)
    
    df_correlations = df_correlations.set_index("cv_fold")
    return df_corrected, df_correlations


###############################################################################

df_female_corrected, df_female_correlations = calculate_correlations(df_female, n_folds, target)
df_male_corrected, df_male_correlations = calculate_correlations(df_male, n_folds, target)

df_female_correlations.to_csv("female_corrected_correlations.csv", sep=',')
df_female_corrected.to_csv("female_corrected_predictions.csv", sep=',')
df_male_correlations.to_csv("male_corrected_correlations.csv", sep=',')
df_male_corrected.to_csv("male_corrected_predictions.csv", sep=',')

print("===== Done! =====")
embed(globals(), locals())
###############################################################################

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
            
            r_text_female = f"r:{df_female_correlations.loc[fold, 'r_values_true_raw_predicted']:.3f}\nR2:{df_female_correlations.loc[fold, 'r2_values_true_raw_predicted']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[fold, 'r_values_true_raw_predicted']:.3f}\nR2:{df_male_correlations.loc[fold, 'r2_values_true_raw_predicted']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
        elif i == 1:
            sns.regplot(data=df_female_corrected[df_female_corrected['cv_fold']==fold], x=f"{target}", y=f"corrected_predicted_hgs", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male_corrected[df_male_corrected['cv_fold']==fold], x=f"{target}", y=f"corrected_predicted_hgs", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
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
plt.savefig(f"true_predicted_offset.png")
plt.close()


###############################################################################

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
            
            r_text_female = f"r:{df_female_correlations.loc[fold, 'r_values_true_raw_delta']:.3f}\nR2:{df_female_correlations.loc[fold, 'r2_values_true_raw_delta']:.3f}"
            r_text_male = f"r:{df_male_correlations.loc[fold, 'r_values_true_raw_delta']:.3f}\nR2:{df_male_correlations.loc[fold, 'r2_values_true_raw_delta']:.3f}"
            ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
            ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
        elif i == 1:
            sns.regplot(data=df_female_corrected[df_female_corrected['cv_fold']==fold], x=f"{target}", y=f"corrected_delta_hgs", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
            sns.regplot(data=df_male_corrected[df_male_corrected['cv_fold']==fold], x=f"{target}", y=f"corrected_delta_hgs", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
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
plt.savefig(f"true_delta_offset.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())

