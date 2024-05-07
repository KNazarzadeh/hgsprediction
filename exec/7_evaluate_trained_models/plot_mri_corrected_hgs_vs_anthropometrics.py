import math
import sys
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from hgsprediction.load_results.load_corrected_prediction_results import load_corrected_prediction_results
from sklearn.metrics import r2_score

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
session = sys.argv[6]
confound_status = sys.argv[7]
n_repeats = sys.argv[8]
n_folds = sys.argv[9]
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

for corr_target in ["bmi", "height", "waist_to_hip_ratio", "age"]:
    fig= plt.figure(figsize=(8,8))
    sns.set_style("whitegrid", {'axes.grid' : False})
    
    sns.regplot(x=df_female[f"{corr_target}"], y=df_female[f"{target}_corrected_delta(true-predicted)"], color='red',  marker="$\circ$", scatter_kws={'s': 50})
    sns.regplot(x=df_male[f"{corr_target}"], y=df_male[f"{target}_corrected_delta(true-predicted)"], color='#069AF3',  marker="$\circ$", scatter_kws={'s': 50})
    
    female_corr = pearsonr(df_female[f"{corr_target}"], df_female[f"{target}_corrected_delta(true-predicted)"])[0]
    male_corr = pearsonr(df_male[f"{corr_target}"], df_male[f"{target}_corrected_delta(true-predicted)"])[0]
    r2_female = r2_score(df_female[f"{corr_target}"], df_female[f"{target}_corrected_delta(true-predicted)"])
    r2_male = r2_score(df_male[f"{corr_target}"], df_male[f"{target}_corrected_delta(true-predicted)"])

    r_text_female = f"r:{female_corr:.3f}\nR2:{r2_female:.3f}"
    r_text_male = f"r:{male_corr:.3f}\nR2:{r2_male:.3f}"
    plt.annotate(r_text_female, xy=(0.5, 0.9), xycoords='axes fraction', fontsize=12, fontweight="bold", color='red')
    plt.annotate(r_text_male, xy=(0.5, 0.8), xycoords='axes fraction', fontsize=12, fontweight="bold", color='#069AF3')    

    plt.xlabel(f"{corr_target}", fontsize=12, fontweight="bold")
    plt.ylabel("Corrected delta HGS", fontsize=12, fontweight="bold")
    plt.title(f"{target}",fontsize=14, fontweight="bold")
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    # Plot regression line
    plt.plot([xmin, xmax], [ymin, ymax], color='darkgrey', linestyle='--')

    plt.show()
    plt.savefig(f"correlation_corrected_delta_with_features_{corr_target}_{population}_{mri_status}_{target}.png")
    plt.close()

print("===== Done! =====")
embed(globals(), locals())