
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
samplesize_list = ["10_percent", "20_percent", "40_percent", "60_percent", "80_percent", "100_percent"]

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

        df_tmp_female.loc[:, 'samplesize_percent'] = samplesize.replace('_', ' ')
        df_tmp_female.loc[:, 'gender'] = "Female"
        df_tmp_male.loc[:, 'samplesize_percent'] = samplesize.replace('_', ' ')
        df_tmp_male.loc[:, 'gender'] = "Male"
        
        df_tmp = pd.concat ([df_tmp_female[[test_score, 'samplesize_percent', 'gender']], df_tmp_male[[test_score, 'samplesize_percent', 'gender']]], axis=0)

        df_scores = pd.concat ([df_scores, df_tmp], axis=0)
    
    df_scores.loc[:, 'model'] = model_name.replace('_', ' ').capitalize()
    df_scores['model_sample'] = df_scores['model'] + " " + df_scores['samplesize_percent']

    if model_name == "linear_svm":
        df_linear_svm = df_scores
    elif model_name == "random_forest":
        df_random_forest = df_scores


###############################################################################
plot_folder = os.path.join(os.getcwd(), f"plots/boxplots/{target}/{n_repeats}_repeats_{n_folds}_folds/{score_type}")
if(not os.path.isdir(plot_folder)):
        os.makedirs(plot_folder)
plot_file = os.path.join(plot_folder, f"comparing_SVM_RF_models_multi_samplesize_by_gender_{target}.png")
###############################################################################
# Create a custom color palette dictionary
# Define custom palettes
palette_male = sns.color_palette("Blues")
# palette_female = sns.color_palette(palette='RdPu')
palette_female = sns.cubehelix_palette()
custom_palette = {'Female': palette_female[1], 'Male': palette_male[2]}
print("===== Done! =====")
embed(globals(), locals())
###############################################################################
# Set the style once for all plots

sns.set_style("whitegrid")
fig, ax = plt.subplots(ncols=2)
# Plot the first boxplot on ax[0]
sns.boxplot(data=df_linear_svm,
            x="samplesize_percent",
            y=test_score,  # Ensure y is passed as a string if it is a column name
            hue="gender",
            palette=custom_palette,
            linewidth=1,
            showcaps=False,
            ax=ax[0],
           )

# Plot the second boxplot on ax[1]
sns.boxplot(data=df_random_forest,
            x="samplesize_percent",
            y=test_score,  # Ensure y is passed as a string if it is a column name
            hue="gender",
            palette=custom_palette,
            linewidth=1,
            showcaps=False,
            ax=ax[1],
           )

# Set the y-axis label with specified properties
# Set y-ticks based on score type
if score_type == "r2_score":
    y_step_value = 0.025
    yticks = np.arange(0, 0.25 + y_step_value, y_step_value)
    ax[0].set_yticks(yticks)
    ax[1].set_yticks(yticks)
    y_lable = "R2 (CV)"
    ax[0].set_ylabel(y_lable, fontsize=16)
    ax[1].set_ylabel("")
elif score_type == "r_score":
    y_step_value = 0.05
    yticks = np.arange(0.2, 0.5 + y_step_value, y_step_value)
    ax[0].set_yticks(yticks)
    ax[1].set_yticks(yticks)
    y_lable = "r (CV)"
    ax[0].set_ylabel(y_lable, fontsize=16)
    ax[1].set_ylabel("")
# Customize y-tick labels properties and set tick direction to 'out'
ax[0].tick_params(axis='y', labelsize=14, direction='out')
ax[1].tick_params(axis='y', labelsize=14, direction='out')

# Format y-tick labels to one decimal place if they are round
ax[0].set_yticklabels([f'{tick:.1f}' if tick % 1 == 0 else f'{tick:.2f}' for tick in yticks])
ax[1].set_yticklabels([f'{tick:.1f}' if tick % 1 == 0 else f'{tick:.2f}' for tick in yticks])
  
# set style for the axes
new_xticks = ["10", "20", "40", "60", "80", "100"]
for axes in [ax[0], ax[1]]:
    axes.set_xticks(range(len(new_xticks)))  # Set tick positions
    axes.set_xticklabels(new_xticks, fontsize=14)  # Set tick labels

# Plot linear svm title
ax[0].set_title('Linear SVM', fontsize=12)
ax[1].set_title('Random Forest',fontsize=12)

# Set the color of the plot's spines to black for both subplots
for ax_subplot in ax:
    for spine in ax_subplot.spines.values():
        spine.set_color('darkgrey')

# Place legend outside the plot
legend = fig.legend(title="Gender", title_fontsize='12', fontsize='10', bbox_to_anchor=(1.05, 1), loc='upper left')

# Remove legend from the axes
for ax_subplot in ax:
    ax_subplot.legend().remove()
    
# Hide y-ticks on the second subplot (using ax[1].set_yticklabels([]) to keep the style)
ax[1].set_yticklabels([])
# Hide x-labels on the second subplot (using ax[1].set_yticklabels([]) to keep the style)
ax[0].set_xlabel("")
ax[1].set_xlabel("")

# Set a common x-label
fig.text(0.5, 0.12, 'Sample size (%)', ha='center')

ax[0].set_box_aspect(1)
ax[1].set_box_aspect(1)

plt.tight_layout()  # Adjust layout to fit x-label

plt.show()
plt.savefig(plot_file)
plt.close()

###############################################################################
###############################################################################

print("===== Done! =====")
embed(globals(), locals())

