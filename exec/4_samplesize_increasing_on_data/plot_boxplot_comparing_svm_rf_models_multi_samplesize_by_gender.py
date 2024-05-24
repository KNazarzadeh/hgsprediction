
import os
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from hgsprediction.load_results.load_multi_samples_trained_models_results import load_scores_trained
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
n_repeats = sys.argv[6]
n_folds = sys.argv[7]
score_type = sys.argv[8]
###############################################################################
if score_type == "r_score":
    test_score = "test_pearson_corr"
elif score_type == "r2_score":
    test_score = "test_r2"
###############################################################################
# samplesize_list = ["10_percent", "20_percent", "40_percent", "60_percent", "80_percent", "100_percent"]
samplesize_list = ["10_percent", "20_percent", "40_percent"]

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
        
        df_tmp = pd.concat ([df_tmp_female[[test_score, 'samplesize_percent', 'gender']], df_tmp_male[[test_score, 'samplesize_percent', 'gender']]], axis=0)

        df_scores = pd.concat ([df_scores, df_tmp], axis=0)
    
    df_scores.loc[:, 'model'] = model_name
    df_combined_models_scores = pd.concat ([df_combined_models_scores, df_scores], axis=0)
    
df_combined_models_scores['samplesize_percent'] = df_combined_models_scores['samplesize_percent'].str.replace('_', ' ')
df_combined_models_scores['model'] = df_combined_models_scores['model'].str.replace('_', ' ').str.capitalize()
df_combined_models_scores['model_sample'] = df_combined_models_scores['model'] + " " + df_combined_models_scores['samplesize_percent']

print(df_combined_models_scores)

df_linear_svm = df_combined_models_scores[df_combined_models_scores['model']==' Linear svm']
df_random_forest = df_combined_models_scores[df_combined_models_scores['model']==' Random forest']

print("===== Done! =====")
embed(globals(), locals())

###############################################################################
plot_folder = os.path.join(os.getcwd(), f"plots/boxplots/{target}/{n_repeats}_repeats_{n_folds}_folds/{score_type}")
if(not os.path.isdir(plot_folder)):
        os.makedirs(plot_folder)
plot_file = os.path.join(plot_folder, f"comparing_SVM_RF_models_multi_samplesize_by_gender_{target}.png")
###############################################################################
# Create a custom color palette dictionary
# Define custom palettes

palette_male = sns.color_palette("Blues")
palette_female = sns.color_palette(palette='PuRd')
custom_palette = {'Female': palette_female[1], 'Male': palette_male[1]}
###############################################################################
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,12))

ax[0] = sns.set_style("darkgrid")
sns.boxplot(data=df_linear_svm,
                 x="model_sample",
                 y=test_score,
                 hue='gender',
                 palette=custom_palette,
                 linewidth=2,
                 showcaps=False,
                #  boxprops=dict(alpha=.5),
                ax=ax[0]
                )
ax[1] = sns.set_style("darkgrid")
sns.boxplot(data=df_random_forest,
                 x="model_sample",
                 y=test_score,
                 hue='gender',
                 palette=custom_palette,
                 linewidth=2,
                 showcaps=False,
                #  boxprops=dict(alpha=.5),
                ax=ax[1]
                )

# plt.title(f"Samplesize increasing for Anthropometrics and Age features {target}", fontsize=20, fontweight="bold", y=1.01)

# ax[0].set_xlabel("Model", fontsize=40, fontweight="bold")
# if score_type == "r_score":
#     y_lable = "r value"
# elif score_type == "r2_score":
#     y_lable = "$R^2$ value"
# y_lable = "accuracy"
# ax[0].set_ylabel(y_lable, fontsize=40, fontweight="bold")

ymin, ymax = plt.ylim()
y_step_value = 0.01
ax[0].set_yticks(np.arange(round(ymin/0.01)*.01-y_step_value, round(ymax/0.01)*.01, 0.01), fontsize=18)

# Change x-axis tick labels
new_xticklabels = ["10%", "20%", "40%", "60%", "80%", "100%", "10%", "20%", "40%", "60%", "80%", "100%"]  # Replace with your desired labels

ax[0].set_xticklabels(new_xticklabels, fontsize=18)

# Set the color of the plot's spines to black for the first subplot
for spine in ax[0].spines.values():
    spine.set_color('black')
    
# Set the color of the plot's spines to black for the second subplot
for spine in ax[1].spines.values():
    spine.set_color('black')
    
    
# Place legend outside the plot
legend = plt.legend(title="Gender", title_fontsize='24', fontsize='20', bbox_to_anchor=(1.05, 1), loc='upper left')
# Adjust transparency of legend markers
# for handle in legend.legend_handles:
#     handle.set_alpha(0.5)  # Set the transparency here as desired

plt.tight_layout()  # Adjust layout to prevent cropping

plt.show()
plt.savefig(plot_file)
plt.close()

###############################################################################
###############################################################################

print("===== Done! =====")
embed(globals(), locals())

