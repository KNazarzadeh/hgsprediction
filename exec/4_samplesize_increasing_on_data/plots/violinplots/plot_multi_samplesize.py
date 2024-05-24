

import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
from hgsprediction.load_results import load_scores_trained

from ptpython.repl import embed
# print("===== Done! =====")
# embed(globals(), locals())


###############################################################################
# Inputs : Required inputs
filename = sys.argv[0]
population = sys.argv[1]
mri_status = sys.argv[2]
feature_type = sys.argv[3]
target = sys.argv[4]
confound_status = sys.argv[5]
model_name = sys.argv[6]
n_repeats = sys.argv[7]
n_folds, = sys.argv[8]
gender = sys.argv[9]

###############################################################################
###############################################################################
samplesize_list = ["10_percent", "20_percent", "40_percent", "60_percent", "80_percent", "100_percent"]

if model_name == "both_models":
    df_combined_models_scores = pd.DataFrame()
    for model_name in ["linear_svm", "random_forest"]:
        df_scores = pd.DataFrame()
        for samplesize in samplesize_list:
            print(samplesize)
            df_tmp1 = load_scores_trained(
                population,
                mri_status,
                confound_status,
                gender,
                feature_type,
                target,
                model_name,
                n_repeats,
                n_folds,
                samplesize,
                )
            df_tmp1.loc[:, 'samplesize_percent'] = samplesize
            df_scores = pd.concat ([df_scores, df_tmp1[['test_score', 'samplesize_percent']]], axis=0)
        
        df_scores.loc[:, 'model'] = model_name
        df_combined_models_scores = pd.concat ([df_combined_models_scores, df_scores], axis=0)
        
    df_combined_models_scores['samplesize_percent'] = df_combined_models_scores['samplesize_percent'].str.replace('_', ' ')
    df_combined_models_scores['model'] = df_combined_models_scores['model'].str.replace('_', ' ').str.capitalize()
    
# else:
#     df_scores = pd.DataFrame()
#     for samplesize in samplesize_list:
#         print(samplesize)
#         df_tmp1 = load_scores_trained(
#             population,
#             mri_status,
#             confound_status,
#             gender,
#             feature_type,
#             target,
#             model_name,
#             n_repeats,
#             n_folds,
#             samplesize,
#             )
#         df_tmp1.loc[:, 'samplesize_percent'] = samplesize
#         df_scores = pd.concat ([df_scores, df_tmp1[['test_score', 'samplesize_percent']]], axis=0)
#         df_scores.loc[:, 'model'] = model_name
        
#     df_scores['samplesize_percent'] = df_scores['samplesize_percent'].str.replace('_', ' ')
#     df_scores['model'] = df_scores['model'].str.replace('_', ' ').str.capitalize()

print("===== Done! =====")
embed(globals(), locals())

###############################################################################
###############################################################################
# Create a custom color palette dictionary
# Define custom palettes
# custom_palette_svm = sns.color_palette("Blues")
# custom_palette_rf = sns.color_palette("YlOrBr")

# custom_palette = {'linear_svm': custom_palette_svm, 'random_forest': custom_palette_rf}

fig = plt.figure(figsize=(18,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 25,
                     "xtick.labelsize": 25})

ax = sns.set_style("whitegrid")
ax = sns.violinplot(data=df_combined_models_scores, x="model", y="test_score", hue='samplesize_percent',
               palette="Pastel1", linewidth=3)

plt.title(f"Samplesize increasing for Anthropometrics and Age features {target}", fontsize=20, fontweight="bold")

plt.xlabel("Model", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")

ymin, ymax = plt.ylim()
y_step_value = 0.01
plt.yticks(np.arange(round(ymin/0.01)*.01-y_step_value, round(ymax/0.01)*.01, 0.01), fontsize=18, weight='bold')


# Place legend outside the plot
plt.legend(title="Samples", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()  # Adjust layout to prevent cropping

plt.show()
plt.savefig(f"{gender}_{model_name}_{target}_violin.png")

# plot_folder_path = os.path.join(
#     "/data",
#     "project",
#     "stroke_ukb",
#     "knazarzadeh",
#     "project_hgsprediction",
#     "plots",
#     f"{population}",
#     f"{mri_status}",
#     f"{feature_type}",
#     f"{target}",
#     "multi_samplesize",
#     f"violin_r2_scores_plots",
# )

# if(not os.path.isdir(plot_folder_path)):
#     os.makedirs(plot_folder_path)

# # Define the csv file path to save
# plot_file_path = os.path.join(plot_folder_path, f"{gender}_{model_name}_violin.png")
    
# plt.savefig(plot_file_path)
plt.close()