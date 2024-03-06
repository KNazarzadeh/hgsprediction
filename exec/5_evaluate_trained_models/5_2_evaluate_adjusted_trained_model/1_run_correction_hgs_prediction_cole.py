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
# gender = sys.argv[10]
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

###############################################################################

model_female = LinearRegression()
model_female.fit(df_female.loc[:, f"{target}"].values.reshape(-1, 1), df_female.loc[:, f"{target}_predicted"])
slope_female = model_female.coef_[0]
intercept_female = model_female.intercept_

model_male = LinearRegression()
model_male.fit(df_male.loc[:, f"{target}"].values.reshape(-1, 1), df_male.loc[:, f"{target}_predicted"])
slope_male = model_male.coef_[0]
intercept_male = model_male.intercept_

###############################################################################
# Assuming that you have already trained and instantiated the model as `model`
folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",  
        "results_hgsprediction",
        f"{population}",
        "mri",
        "2_session_ukb",
        f"{feature_type}",
        f"{target}",
        f"{confound}",
        f"{model_name}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        "hgs_predicted_results",
    )

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    "female_hgs_predicted_data.csv")

df_female_mri = pd.read_csv(file_path, sep=',', index_col=0)
    

# Assuming that you have already trained and instantiated the model as `model`
folder_path = os.path.join(
        "/data",
        "project",
        "stroke_ukb",
        "knazarzadeh",
        "project_hgsprediction",  
        "results_hgsprediction",
        f"{population}",
        "mri",
        "2_session_ukb",
        f"{feature_type}",
        f"{target}",
        f"{confound}",
        f"{model_name}",
        f"{n_repeats}_repeats_{n_folds}_folds",
        "hgs_predicted_results",
    )

# Define the csv file path to save
file_path = os.path.join(
    folder_path,
    "male_hgs_predicted_data.csv")

df_male_mri = pd.read_csv(file_path, sep=',', index_col=0)

###############################################################################
print("===== Done! =====")
embed(globals(), locals())
df_female_mri_correlations = pd.DataFrame(columns=["r_values_true_raw_predicted", "r2_values_true_raw_predicted",
                                            "r_values_true_raw_delta", "r2_values_true_raw_delta",
                                            "r_values_true_corrected_predicted", "r2_values_true_corrected_predicted",
                                            "r_values_true_corrected_delta", "r2_values_true_corrected_delta"])

df_female_mri.loc[:, "corrected_predicted_hgs"] = (df_female_mri.loc[:, f"{target}_predicted"] - intercept_female) / slope_female
df_female_mri.loc[:, "corrected_delta_hgs"] =  df_female_mri.loc[:, f"{target}"] - df_female_mri.loc[:, "corrected_predicted_hgs"]

r_values_true_raw_predicted = pearsonr(df_female_mri.loc[:, f"{target}"],df_female_mri.loc[:,f"{target}_predicted"])[0]
r2_values_true_raw_predicted = r2_score(df_female_mri.loc[:, f"{target}"],df_female_mri.loc[:,f"{target}_predicted"])

r_values_true_raw_delta = pearsonr(df_female_mri.loc[:, f"{target}"],df_female_mri.loc[:,f"{target}_delta(true-predicted)"])[0]
r2_values_true_raw_delta = r2_score(df_female_mri.loc[:, f"{target}"],df_female_mri.loc[:,f"{target}_delta(true-predicted)"])

r_values_true_corrected_predicted = pearsonr(df_female_mri.loc[:, f"{target}"],df_female_mri.loc[:,"corrected_predicted_hgs"])[0]
r2_values_true_corrected_predicted = r2_score(df_female_mri.loc[:, f"{target}"],df_female_mri.loc[:,"corrected_predicted_hgs"])

r_values_true_corrected_delta = pearsonr(df_female_mri.loc[:, f"{target}"],df_female_mri.loc[:,"corrected_delta_hgs"])[0]
r2_values_true_corrected_delta = r2_score(df_female_mri.loc[:, f"{target}"],df_female_mri.loc[:,"corrected_delta_hgs"])

df_female_mri_correlations.loc[0, "r_values_true_raw_predicted"] = r_values_true_raw_predicted
df_female_mri_correlations.loc[0, "r2_values_true_raw_predicted"] = r2_values_true_raw_predicted
df_female_mri_correlations.loc[0, "r_values_true_raw_delta"] = r_values_true_raw_delta
df_female_mri_correlations.loc[0, "r2_values_true_raw_delta"] = r2_values_true_raw_delta
df_female_mri_correlations.loc[0, "r_values_true_corrected_predicted"] = r_values_true_corrected_predicted
df_female_mri_correlations.loc[0, "r2_values_true_corrected_predicted"] = r2_values_true_corrected_predicted
df_female_mri_correlations.loc[0, "r_values_true_corrected_delta"] = r_values_true_corrected_delta
df_female_mri_correlations.loc[0, "r2_values_true_corrected_delta"] = r2_values_true_corrected_delta


###############################################################################

df_male_mri_correlations = pd.DataFrame(columns=["r_values_true_raw_predicted", "r2_values_true_raw_predicted",
                                            "r_values_true_raw_delta", "r2_values_true_raw_delta",
                                            "r_values_true_corrected_predicted", "r2_values_true_corrected_predicted",
                                            "r_values_true_corrected_delta", "r2_values_true_corrected_delta"])

df_male_mri.loc[:, "corrected_predicted_hgs"] = (df_male_mri.loc[:, f"{target}_predicted"] - intercept_male) / slope_male
df_male_mri.loc[:, "corrected_delta_hgs"] =  df_male_mri.loc[:, f"{target}"] - df_male_mri.loc[:, "corrected_predicted_hgs"]

r_values_true_raw_predicted = pearsonr(df_male_mri.loc[:, f"{target}"],df_male_mri.loc[:,f"{target}_predicted"])[0]
r2_values_true_raw_predicted = r2_score(df_male_mri.loc[:, f"{target}"],df_male_mri.loc[:,f"{target}_predicted"])

r_values_true_raw_delta = pearsonr(df_male_mri.loc[:, f"{target}"],df_male_mri.loc[:,f"{target}_delta(true-predicted)"])[0]
r2_values_true_raw_delta = r2_score(df_male_mri.loc[:, f"{target}"],df_male_mri.loc[:,f"{target}_delta(true-predicted)"])

r_values_true_corrected_predicted = pearsonr(df_male_mri.loc[:, f"{target}"],df_male_mri.loc[:,"corrected_predicted_hgs"])[0]
r2_values_true_corrected_predicted = r2_score(df_male_mri.loc[:, f"{target}"],df_male_mri.loc[:,"corrected_predicted_hgs"])

r_values_true_corrected_delta = pearsonr(df_male_mri.loc[:, f"{target}"],df_male_mri.loc[:,"corrected_delta_hgs"])[0]
r2_values_true_corrected_delta = r2_score(df_male_mri.loc[:, f"{target}"],df_male_mri.loc[:,"corrected_delta_hgs"])

df_male_mri_correlations.loc[0, "r_values_true_raw_predicted"] = r_values_true_raw_predicted
df_male_mri_correlations.loc[0, "r2_values_true_raw_predicted"] = r2_values_true_raw_predicted
df_male_mri_correlations.loc[0, "r_values_true_raw_delta"] = r_values_true_raw_delta
df_male_mri_correlations.loc[0, "r2_values_true_raw_delta"] = r2_values_true_raw_delta
df_male_mri_correlations.loc[0, "r_values_true_corrected_predicted"] = r_values_true_corrected_predicted
df_male_mri_correlations.loc[0, "r2_values_true_corrected_predicted"] = r2_values_true_corrected_predicted
df_male_mri_correlations.loc[0, "r_values_true_corrected_delta"] = r_values_true_corrected_delta
df_male_mri_correlations.loc[0, "r2_values_true_corrected_delta"] = r2_values_true_corrected_delta

###############################################################################
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
        sns.regplot(data=df_female_mri, x=f"{target}", y=f"{target}_predicted", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male_mri, x=f"{target}", y=f"{target}_predicted", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Raw predicted HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("")            
        # ax.set_title(f"Fold:{fold}", fontsize=40, fontweight="bold")            
        
        r_text_female = f"r:{df_female_mri_correlations.loc[0, 'r_values_true_raw_predicted']:.3f}\nR2:{df_female_mri_correlations.loc[0, 'r2_values_true_raw_predicted']:.3f}"
        r_text_male = f"r:{df_male_mri_correlations.loc[0, 'r_values_true_raw_predicted']:.3f}\nR2:{df_male_mri_correlations.loc[0, 'r2_values_true_raw_predicted']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    elif i == 1:
        sns.regplot(data=df_female_mri, x=f"{target}", y=f"corrected_predicted_hgs", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male_mri, x=f"{target}", y=f"corrected_predicted_hgs", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Corrected predicted HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

        r_text_female = f"r:{df_female_mri_correlations.loc[0, 'r_values_true_corrected_predicted']:.3f}\nR2:{df_female_mri_correlations.loc[0, 'r2_values_true_corrected_predicted']:.3f}"
        r_text_male = f"r:{df_male_mri_correlations.loc[0, 'r_values_true_corrected_predicted']:.3f}\nR2:{df_male_mri_correlations.loc[0, 'r2_values_true_corrected_predicted']:.3f}"
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
        sns.regplot(data=df_female_mri, x=f"{target}", y=f"{target}_delta(true-predicted)", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male_mri, x=f"{target}", y=f"{target}_delta(true-predicted)", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 12}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Raw delta HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("")                        
        
        r_text_female = f"r:{df_female_mri_correlations.loc[0, 'r_values_true_raw_delta']:.3f}\nR2:{df_female_mri_correlations.loc[0, 'r2_values_true_raw_delta']:.3f}"
        r_text_male = f"r:{df_male_mri_correlations.loc[0, 'r_values_true_raw_delta']:.3f}\nR2:{df_male_mri_correlations.loc[0, 'r2_values_true_raw_delta']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    elif i == 1:
        sns.regplot(data=df_female_mri, x=f"{target}", y=f"corrected_delta_hgs", color='lightcoral', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "red"}, ax=ax)
        sns.regplot(data=df_male_mri, x=f"{target}", y=f"corrected_delta_hgs", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, line_kws={"color": "blue"}, ax=ax)
        ax.set_ylabel("Corrected delta HGS", fontsize=40, fontweight="bold")
        ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")

        r_text_female = f"r:{df_female_mri_correlations.loc[0, 'r_values_true_corrected_delta']:.3f}\nR2:{df_female_mri_correlations.loc[0, 'r2_values_true_corrected_delta']:.3f}"
        r_text_male = f"r:{df_male_mri_correlations.loc[0, 'r_values_true_corrected_delta']:.3f}\nR2:{df_male_mri_correlations.loc[0, 'r2_values_true_corrected_delta']:.3f}"
        ax.annotate(r_text_female, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=30, fontweight="bold", color='red')
        ax.annotate(r_text_male, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=30, fontweight="bold", color='#069AF3')
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=12)

plt.tight_layout()
plt.show()
plt.savefig(f"true_delta.png")
plt.close()
    