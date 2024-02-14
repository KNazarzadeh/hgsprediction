

import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
            df_tmp_female = load_scores_trained(
                population,
                mri_status,
                confound_status,
                "female",
                feature_type,
                target,
                model_name,
                n_repeats,
                n_folds,
                samplesize,
                )
            df_tmp_male = load_scores_trained(
                population,
                mri_status,
                confound_status,
                "male",
                feature_type,
                target,
                model_name,
                n_repeats,
                n_folds,
                samplesize,
                )
            
            df_tmp_female.loc[:, 'samplesize_percent'] = samplesize
            df_tmp_female.loc[:, 'gender'] = "Female"
            df_tmp_male.loc[:, 'samplesize_percent'] = samplesize
            df_tmp_male.loc[:, 'gender'] = "Male"
            
            df_tmp = pd.concat ([df_tmp_female[['test_score', 'samplesize_percent', 'gender']], df_tmp_male[['test_score', 'samplesize_percent', 'gender']]], axis=0)

            df_scores = pd.concat ([df_scores, df_tmp], axis=0)
        
        df_scores.loc[:, 'model'] = model_name
        df_combined_models_scores = pd.concat ([df_combined_models_scores, df_scores], axis=0)
        
    df_combined_models_scores['samplesize_percent'] = df_combined_models_scores['samplesize_percent'].str.replace('_', ' ')
    df_combined_models_scores['model'] = df_combined_models_scores['model'].str.replace('_', ' ').str.capitalize()
    df_combined_models_scores['model_sample'] = df_combined_models_scores['model'] + " " + df_combined_models_scores['samplesize_percent']

    print(df_combined_models_scores)
    
# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
###############################################################################
# Create a custom color palette dictionary
# Define custom palettes

custom_palette = {'Female': 'red', 'Male': '#069AF3'}

fig = plt.figure(figsize=(18,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 25,
                     "xtick.labelsize": 25})

ax = sns.set_style("whitegrid")
ax = sns.violinplot(data=df_combined_models_scores, x="model_sample", y="test_score", hue='gender', palette=custom_palette, linewidth=3, inner="box")

# Adjust transparency of violins and set edge color to match face color
for collection in ax.collections:
    if isinstance(collection, matplotlib.collections.PolyCollection):
        collection.set_edgecolor(collection.get_facecolor())
        collection.set_alpha(0.5)

        
plt.title(f"Samplesize increasing for Anthropometrics and Age features {target}", fontsize=20, fontweight="bold")

plt.xlabel("Model", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")

ymin, ymax = plt.ylim()
y_step_value = 0.01
plt.yticks(np.arange(round(ymin/0.01)*.01-y_step_value, round(ymax/0.01)*.01, 0.01), fontsize=18, weight='bold')

# Change x-axis tick labels
new_xticklabels = ["10%", "20%", "40%", "60%", "80%", "100%", "10%", "20%", "40%", "60%", "80%", "100%"]  # Replace with your desired labels
ax.set_xticklabels(new_xticklabels, fontsize=18, weight='bold')

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')


plt.tight_layout()  # Adjust layout to prevent cropping

plt.show()
plt.savefig(f"gender_specified_{model_name}_{target}_violin.png")
plt.close()

# print("===== Done! =====")
# embed(globals(), locals())

###############################################################################
###############################################################################
custom_palette = {'Female': 'red', 'Male': '#069AF3'}

fig = plt.figure(figsize=(18,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 25,
                     "xtick.labelsize": 25})

ax = sns.set_style("whitegrid")
ax = sns.boxplot(data=df_combined_models_scores, x="model_sample", y="test_score", hue='gender', palette=custom_palette, linewidth=2, boxprops=dict(alpha=.5))
        
plt.title(f"Samplesize increasing for Anthropometrics and Age features {target}", fontsize=20, fontweight="bold", y=1.01)

plt.xlabel("Model", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")

ymin, ymax = plt.ylim()
y_step_value = 0.01
plt.yticks(np.arange(round(ymin/0.01)*.01-y_step_value, round(ymax/0.01)*.01, 0.01), fontsize=18, weight='bold')

# Change x-axis tick labels
new_xticklabels = ["10%", "20%", "40%", "60%", "80%", "100%", "10%", "20%", "40%", "60%", "80%", "100%"]  # Replace with your desired labels
ax.set_xticklabels(new_xticklabels, fontsize=18, weight='bold')

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')


plt.tight_layout()  # Adjust layout to prevent cropping

plt.show()
plt.savefig(f"boxplot_gender_specified_{model_name}_{target}.png")
plt.close()

###############################################################################
###############################################################################
custom_palette = {'Female': 'red', 'Male': '#069AF3'}

fig = plt.figure(figsize=(18,12))

plt.rcParams.update({"font.weight": "bold", 
                     "axes.labelweight": "bold",
                     "ytick.labelsize": 25,
                     "xtick.labelsize": 25})

ax = sns.set_style("whitegrid")
ax = sns.barplot(data=df_combined_models_scores, x="model_sample", y="test_score", hue='gender', palette=custom_palette, linewidth=2)

# Adjust transparency of bars and set edge color to match face color
for container in ax.containers:
    for bar in container:
        bar.set_alpha(0.5)
        bar.set_edgecolor(bar.get_facecolor())


# Set the color of the plot's spines to black
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black') 

plt.title(f"Samplesize increasing for Anthropometrics and Age features {target}", fontsize=20, fontweight="bold", y=1.01)

plt.xlabel("Model", fontsize=40, fontweight="bold")
plt.ylabel("R2 score", fontsize=40, fontweight="bold")

ymin, ymax = plt.ylim()
ymin=0
y_step_value = 0.01
plt.yticks(np.arange(round(ymin/0.01)*.01, round(ymax/0.01)*.01, 0.01), fontsize=18, weight='bold')

# Change x-axis tick labels
new_xticklabels = ["10%", "20%", "40%", "60%", "80%", "100%", "10%", "20%", "40%", "60%", "80%", "100%"]  # Replace with your desired labels
ax.set_xticklabels(new_xticklabels, fontsize=18, weight='bold')

# Place legend outside the plot
plt.legend(title="Gender", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')


plt.tight_layout()  # Adjust layout to prevent cropping

plt.show()
plt.savefig(f"barplot_gender_specified_{model_name}_{target}.png")
plt.close()

print("===== Done! =====")
embed(globals(), locals())