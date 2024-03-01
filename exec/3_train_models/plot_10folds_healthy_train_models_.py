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
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
fig, axes = plt.subplots(1, 5, figsize=(80, 40))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 60,
                     "xtick.labelsize": 60,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for i in range(1):
    for j in range(5):
        fold = i * 5 + j
        ax = axes[i][j]
        
        sns.regplot(data=df_female[df_female['fold']==fold], x="hgs_L+R", y="hgs_pred", color='red', marker="$\circ$", scatter_kws={'s': 50}, ax=ax)
        sns.regplot(data=df_male[df_male['fold']==fold], x="hgs_L+R", y="hgs_pred", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50}, ax=ax)

        ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")
        ax.set_ylabel("Predicted HGS", fontsize=40, fontweight="bold")
        ax.set_title(f"Fold {fold}", fontsize=40, fontweight="bold")

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

plt.tight_layout()
plt.show()
plt.savefig(f"true_predicted_5folds.png")
plt.close()
###############################################################################
df_female.loc[:, "delta"] = df_female.loc[:, "hgs_L+R"] - df_female.loc[:, "hgs_pred"]
df_male.loc[:, "delta"] = df_male.loc[:, "hgs_L+R"] - df_male.loc[:, "hgs_pred"]

print("===== Done! =====")
embed(globals(), locals())
###############################################################################
from sklearn.linear_model import LinearRegression
df_male_delta_pred = pd.DataFrame()
df_female_delta_pred = pd.DataFrame()
# Assuming X_train and y_train contain the training data for the delta feature and HGS respectively
model = LinearRegression()
for fold in range(5):
    # Select data for the current fold
    df_tmp = df_female[df_female['fold']==fold]
    # Take a random sample of 50% of data for training, setting random_state for reproducibility    
    df_female_half = df_tmp.sample(frac=0.5, random_state=42)
    # Select the remaining data for testing    
    df_female_half_rest = df_tmp[~df_tmp.index.isin(df_female_half.index)]
    # Fit the model on the 50% of data of the fold   
    model.fit(df_female_half.loc[:,"delta"].values.reshape(-1, 1), df_female_half.loc[:,"hgs_L+R"])
    # Predict HGS for the remaining data
    df_female_half_rest.loc[:, "hgs_delta_pred"] = model.predict(df_female_half_rest.loc[:,"delta"].values.reshape(-1, 1))
    # Concatenate the predicted on remaining data   
    df_female_delta_pred = pd.concat([df_female_delta_pred, df_female_half_rest], axis=0)
    
for fold in range(5):
    df_tmp = df_male[df_male['fold']==fold]
    df_male_half = df_tmp.sample(frac=0.5, random_state=42)  # Set random_state for reproducibility
    df_male_half_rest = df_tmp[~df_tmp.index.isin(df_male_half.index)]
    model.fit(df_male_half.loc[:,"delta"].values.reshape(-1, 1), df_male_half.loc[:,"hgs_L+R"])
    df_male_half_rest.loc[:, "hgs_delta_pred"] = model.predict(df_male_half_rest.loc[:,"delta"].values.reshape(-1, 1))
    df_male_delta_pred = pd.concat([df_male_delta_pred, df_male_half_rest], axis=0)

###############################################################################
fig, axes = plt.subplots(1, 5, figsize=(80, 40))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 60,
                     "xtick.labelsize": 60,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for j in range(5):
    fold = j
    ax = axes[j]
    
    sns.regplot(data=df_female_delta_pred[df_female_delta_pred['fold']==fold], x="hgs_L+R", y="hgs_delta_pred", color='red', marker="$\circ$", scatter_kws={'s': 50}, ax=ax)
    sns.regplot(data=df_male_delta_pred[df_male_delta_pred['fold']==fold], x="hgs_L+R", y="hgs_delta_pred", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50}, ax=ax)

    ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")
    ax.set_ylabel("Predicted HGS with delta feature", fontsize=40, fontweight="bold")
    ax.set_title(f"Fold {fold}", fontsize=40, fontweight="bold")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

plt.tight_layout()
plt.show()
plt.savefig(f"true_predicted_with delta_feature_5folds.png")
plt.close()
###############################################################################
from sklearn.linear_model import LinearRegression
df_male_pred_pred = pd.DataFrame()
df_female_pred_pred = pd.DataFrame()
# Assuming X_train and y_train contain the training data for the delta feature and HGS respectively
model = LinearRegression()
for fold in range(5):
    # Select data for the current fold
    df_tmp = df_female[df_female['fold']==fold]
    # Take a random sample of 50% of data for training, setting random_state for reproducibility    
    df_female_half = df_tmp.sample(frac=0.5, random_state=42)
    # Select the remaining data for testing    
    df_female_half_rest = df_tmp[~df_tmp.index.isin(df_female_half.index)]
    # Fit the model on the 50% of data of the fold   
    model.fit(df_female_half.loc[:,"hgs_pred"].values.reshape(-1, 1), df_female_half.loc[:,"hgs_L+R"])
    # Predict HGS for the remaining data
    df_female_half_rest.loc[:, "hgs_pred_pred"] = model.predict(df_female_half_rest.loc[:,"hgs_pred"].values.reshape(-1, 1))
    # Concatenate the predicted on remaining data   
    df_female_pred_pred = pd.concat([df_female_pred_pred, df_female_half_rest], axis=0)
    
for fold in range(5):
    df_tmp = df_male[df_male['fold']==fold]
    df_male_half = df_tmp.sample(frac=0.5, random_state=42)  # Set random_state for reproducibility
    df_male_half_rest = df_tmp[~df_tmp.index.isin(df_male_half.index)]
    model.fit(df_male_half.loc[:,"hgs_pred"].values.reshape(-1, 1), df_male_half.loc[:,"hgs_L+R"])
    df_male_half_rest.loc[:, "hgs_pred_pred"] = model.predict(df_male_half_rest.loc[:,"hgs_pred"].values.reshape(-1, 1))
    df_male_pred_pred = pd.concat([df_male_pred_pred, df_male_half_rest], axis=0)
    
############################################################################### 
fig, axes = plt.subplots(1, 5, figsize=(100, 20))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 40,
                     "xtick.labelsize": 40,
                     })

sns.set_style("whitegrid", {'axes.grid' : False})

for j in range(5):
    fold = j
    ax = axes[j]
    
    sns.regplot(data=df_female_pred_pred[df_female_pred_pred['fold']==fold], x="hgs_L+R", y="hgs_pred_pred", color='red', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, ax=ax)
    sns.regplot(data=df_male_pred_pred[df_male_pred_pred['fold']==fold], x="hgs_L+R", y="hgs_pred_pred", color='#069AF3', marker="$\circ$", scatter_kws={'s': 50, 'linewidths': 5}, ax=ax)

    ax.set_xlabel("True HGS", fontsize=40, fontweight="bold")
    ax.set_ylabel("Predicted HGS(used perdicted hgs as feature)", fontsize=40, fontweight="bold")
    ax.set_title(f"Fold {fold}", fontsize=40, fontweight="bold")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--', linewidth=5)

plt.tight_layout()
plt.show()
plt.savefig(f"true_predicted_with_pred_feature_5folds.png")
plt.close()